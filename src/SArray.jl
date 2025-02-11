
@noinline function generator_too_short_error(inds::CartesianIndices, i::CartesianIndex)
    error("Generator produced too few elements: Expected exactly $(shape_string(inds)) elements, but generator stopped at $(shape_string(i))")
end
@noinline function generator_too_long_error(inds::CartesianIndices)
    error("Generator produced too many elements: Expected exactly $(shape_string(inds)) elements, but generator yields more")
end

shape_string(inds::CartesianIndices) = join(length.(inds.indices), '×')
shape_string(inds::CartesianIndex) = join(Tuple(inds), '×')

@inline throw_if_nothing(x, inds, i) =
    (x === nothing && generator_too_short_error(inds, i); x)

@generated function sacollect(::Type{SA}, gen) where {SA <: StaticArray{S}} where {S <: Tuple}
    stmts = [:(@_inline_meta)]
    args = []
    iter = :(iterate(gen))
    inds = CartesianIndices(size_to_tuple(S))
    for i in inds
        el = Symbol(:el, i)
        push!(stmts, :(($el,st) = throw_if_nothing($iter, $inds, $i)))
        push!(args, el)
        iter = :(iterate(gen,st))
    end
    push!(stmts, :($iter === nothing || generator_too_long_error($inds)))
    push!(stmts, :(SA(($(args...),))))
    Expr(:block, stmts...)
end
"""
    sacollect(SA, gen)

Construct a statically-sized vector of type `SA`.from a generator
`gen`. `SA` needs to have a size parameter since the length of `vec`
is unknown to the compiler. `SA` can optionally specify the element
type as well.

Example:
```julia
sacollect(SVector{3, Int}, 2i+1 for i in 1:3)
sacollect(SMatrix{2, 3}, i+j for i in 1:2, j in 1:3)
sacollect(SArray{2, 3}, i+j for i in 1:2, j in 1:3)
```

This creates the same statically-sized vector as if the generator were
collected in an array, but is more efficient since no array is
allocated.

Equivalent:

```julia
SVector{3, Int}([2i+1 for i in 1:3])
```
"""
sacollect

@inline (::Type{SA})(gen::Base.Generator) where {SA <: StaticArray} =
    sacollect(SA, gen)

####################
## SArray methods ##
####################

@propagate_inbounds function getindex(v::SArray, i::Int)
    getfield(v,:data)[i]
end

@inline Base.Tuple(v::SArray) = getfield(v,:data)

Base.dataids(::SArray) = ()

# See #53
Base.cconvert(::Type{Ptr{T}}, a::SArray) where {T} = Base.RefValue(a)
Base.unsafe_convert(::Type{Ptr{T}}, a::Base.RefValue{SA}) where {S,T,D,L,SA<:SArray{S,T,D,L}} =
    Ptr{T}(Base.unsafe_convert(Ptr{SArray{S,T,D,L}}, a))

# Handle nested cat ast.
_cat_ndims(x) = 0
_cat_ndims(x::AbstractArray) = ndims(x)
_cat_size(x, _) = 1
_cat_size(x::AbstractArray, i) = size(x, i)
_cat_sizes(x, dims) = ntuple(i -> _cat_size(x, i), dims)

function cat_any(::Val{maxdim}, ::Val{catdim}, args::Vector{Any}) where {maxdim,catdim}
    szs = Dims{maxdim}[_cat_sizes(a, Val(maxdim)) for a in args]
    out = Array{Any}(undef, check_cat_size(szs, catdim))
    dims_before = ntuple(_ -> (:), Val(catdim-1))
    dims_after = ntuple(_ -> (:), Val(maxdim-catdim))
    cat_any!(out, dims_before, dims_after, args)
end

function cat_any!(out, dims_before, dims_after, args::Vector{Any})
    catdim = length(dims_before) + 1
    i = 0
    @views for arg in args
        len = _cat_size(arg, catdim)
        dest = out[dims_before..., i+1:i+len, dims_after...]
        if arg isa AbstractArray
            copyto!(dest, arg)
        else
            dest[] = arg
        end
        i += len
    end
    out
end

@noinline cat_mismatch(j,sz,nsz) = throw(DimensionMismatch("mismatch in dimension $j (expected $sz got $nsz)"))
function check_cat_size(szs::Vector{Dims{maxdim}}, catdim) where {maxdim}
    isempty(szs) && return ntuple(_ -> 0, Val(maxdim))
    sz = szs[1]
    catsz = sz[catdim]
    for i in 2:length(szs)
        for j in 1:maxdim
            nszⱼ = szs[i][j]
            if j == catdim
                catsz += nszⱼ
            elseif sz[j] != nszⱼ
                cat_mismatch(j, sz[j], nszⱼ)
            end
        end
    end
    return Base.setindex(sz, catsz, catdim)
end

parse_cat_ast(x) = x
function parse_cat_ast(ex::Expr)
    head, args = ex.head, ex.args
    head === :vect && return args
    i = 0
    if head === :typed_vcat || head === :typed_hcat || head === :typed_ncat
        i += 1 # skip Type arg
    end
    if head === :vcat || head === :typed_vcat
        catdim = 1
    elseif head === :hcat || head === :row || head === :typed_hcat
        catdim = 2
    elseif head === :ncat || head === :typed_ncat || head === :nrow
        catdim = args[i+=1]::Int
    else
        return ex
    end
    nargs = Any[parse_cat_ast(args[k]) for k in i+1:length(args)]
    maxdim = maximum(_cat_ndims, nargs; init = catdim)
    cat_any(Val(maxdim), Val(catdim), nargs)
end

#=
For example,
* `@SArray rand(2, 3, 4)`
* `@SArray rand(rng, 3, 4)`
will be expanded to the following.
* `_rand_with_Val(SArray, 2, 3, _int2val(2),   _int2val(3), Val((4,)))`
* `_rand_with_Val(SArray, 2, 3, _int2val(rng), _int2val(3), Val((4,)))`
The function `_int2val` is required to avoid the following case.
* `_rand_with_Val(SArray, 2, 3, Val(2),   Val(3), Val((4,)))`
* `_rand_with_Val(SArray, 2, 3, Val(rng), Val(3), Val((4,)))`
Mutable object such as `rng` cannot be type parameter, and `Val(rng)` throws an error.
=#
_int2val(x::Int) = Val(x)
_int2val(::Any) = nothing
# @SArray zeros(...)
_zeros_with_Val(::Type{SA}, ::Int,       ::Val{n1}, ::Val{ns}) where {SA, n1, ns} = zeros(SA{Tuple{n1, ns...}})
_zeros_with_Val(::Type{SA}, T::DataType, ::Val,     ::Val{ns}) where {SA, ns} = zeros(SA{Tuple{ns...}, T})
# @SArray ones(...)
_ones_with_Val(::Type{SA}, ::Int,       ::Val{n1}, ::Val{ns}) where {SA, n1, ns} = ones(SA{Tuple{n1, ns...}})
_ones_with_Val(::Type{SA}, T::DataType, ::Val,     ::Val{ns}) where {SA, ns} = ones(SA{Tuple{ns...}, T})
# @SArray rand(...)
@inline _rand_with_Val(::Type{SA}, ::Int,            ::Int,       ::Val{n1}, ::Val{n2}, ::Val{ns}) where {SA, n1, n2, ns} = rand(SA{Tuple{n1,n2,ns...}})
@inline _rand_with_Val(::Type{SA}, T::DataType,      ::Int,       ::Nothing, ::Val{n1}, ::Val{ns}) where {SA, n1, ns} = _rand(Random.GLOBAL_RNG, T, Size(n1, ns...), SA{Tuple{n1, ns...}, T})
@inline _rand_with_Val(::Type{SA}, sampler,          ::Int,       ::Nothing, ::Val{n1}, ::Val{ns}) where {SA, n1, ns} = _rand(Random.GLOBAL_RNG, sampler, Size(n1, ns...), SA{Tuple{n1, ns...}, Random.gentype(sampler)})
@inline _rand_with_Val(::Type{SA}, rng::AbstractRNG, ::Int,       ::Nothing, ::Val{n1}, ::Val{ns}) where {SA, n1, ns} = _rand(rng, Float64, Size(n1, ns...), SA{Tuple{n1, ns...}, Float64})
@inline _rand_with_Val(::Type{SA}, rng::AbstractRNG, T::DataType, ::Nothing, ::Nothing, ::Val{ns}) where {SA, ns} = _rand(rng, T, Size(ns...), SA{Tuple{ns...}, T})
@inline _rand_with_Val(::Type{SA}, rng::AbstractRNG, sampler,     ::Nothing, ::Nothing, ::Val{ns}) where {SA, ns} = _rand(rng, sampler, Size(ns...), SA{Tuple{ns...}, Random.gentype(sampler)})
# @SArray randn(...)
@inline _randn_with_Val(::Type{SA}, ::Int,            ::Int,       ::Val{n1}, ::Val{n2}, ::Val{ns}) where {SA, n1, n2, ns} = randn(SA{Tuple{n1,n2,ns...}})
@inline _randn_with_Val(::Type{SA}, T::DataType,      ::Int,       ::Nothing, ::Val{n1}, ::Val{ns}) where {SA, n1, ns} = _randn(Random.GLOBAL_RNG, Size(n1, ns...), SA{Tuple{n1, ns...}, T})
@inline _randn_with_Val(::Type{SA}, rng::AbstractRNG, ::Int,       ::Nothing, ::Val{n1}, ::Val{ns}) where {SA, n1, ns} = _randn(rng, Size(n1, ns...), SA{Tuple{n1, ns...}, Float64})
@inline _randn_with_Val(::Type{SA}, rng::AbstractRNG, T::DataType, ::Nothing, ::Nothing, ::Val{ns}) where {SA, ns} = _randn(rng, Size(ns...), SA{Tuple{ns...}, T})
# @SArray randexp(...)
@inline _randexp_with_Val(::Type{SA}, ::Int,            ::Int,       ::Val{n1}, ::Val{n2}, ::Val{ns}) where {SA, n1, n2, ns} = randexp(SA{Tuple{n1,n2,ns...}})
@inline _randexp_with_Val(::Type{SA}, T::DataType,      ::Int,       ::Nothing, ::Val{n1}, ::Val{ns}) where {SA, n1, ns} = _randexp(Random.GLOBAL_RNG, Size(n1, ns...), SA{Tuple{n1, ns...}, T})
@inline _randexp_with_Val(::Type{SA}, rng::AbstractRNG, ::Int,       ::Nothing, ::Val{n1}, ::Val{ns}) where {SA, n1, ns} = _randexp(rng, Size(n1, ns...), SA{Tuple{n1, ns...}, Float64})
@inline _randexp_with_Val(::Type{SA}, rng::AbstractRNG, T::DataType, ::Nothing, ::Nothing, ::Val{ns}) where {SA, ns} = _randexp(rng, Size(ns...), SA{Tuple{ns...}, T})

escall(args) = Iterators.map(esc, args)
function _isnonnegvec(args)
    length(args) == 0 && return false
    all(isa.(args, Integer)) && return all(args .≥ 0)
    return false
end
function static_array_gen(::Type{SA}, @nospecialize(ex), mod::Module) where {SA}
    if !isa(ex, Expr)
        error("Bad input for @$SA")
    end
    head = ex.head
    if head === :vect    # vector
        return :($SA{Tuple{$(length(ex.args))}}($tuple($(escall(ex.args)...))))
    elseif head === :ref # typed, vector
        return :($SA{Tuple{$(length(ex.args)-1)},$(esc(ex.args[1]))}($tuple($(escall(ex.args[2:end])...))))
    elseif head === :typed_vcat || head === :typed_hcat || head === :typed_ncat # typed, cat
        args = parse_cat_ast(ex)
        return :($SA{Tuple{$(size(args)...)},$(esc(ex.args[1]))}($tuple($(escall(args)...))))
    elseif head === :vcat || head === :hcat || head === :ncat # untyped, cat
        args = parse_cat_ast(ex)
        return :($SA{Tuple{$(size(args)...)}}($tuple($(escall(args)...))))
    elseif head === :comprehension
        if length(ex.args) != 1
            error("Expected generator in comprehension, e.g. [f(i,j) for i = 1:3, j = 1:3]")
        end
        ex = ex.args[1]
        if !isa(ex, Expr) || (ex::Expr).head != :generator
            error("Expected generator in comprehension, e.g. [f(i,j) for i = 1:3, j = 1:3]")
        end
        n_rng = length(ex.args) - 1
        rng_args = (ex.args[i+1].args[1] for i = 1:n_rng)
        rngs = Any[Core.eval(mod, ex.args[i+1].args[2]) for i = 1:n_rng]
        exprs = (:(f($(j...))) for j in Iterators.product(rngs...))
        return quote
            let
                f($(escall(rng_args)...)) = $(esc(ex.args[1]))
                $SA{Tuple{$(size(exprs)...)}}($tuple($(exprs...)))
            end
        end
    elseif head === :typed_comprehension
        if length(ex.args) != 2
            error("Expected generator in typed comprehension, e.g. Float64[f(i,j) for i = 1:3, j = 1:3]")
        end
        T = esc(ex.args[1])
        ex = ex.args[2]
        if !isa(ex, Expr) || (ex::Expr).head != :generator
            error("Expected generator in typed comprehension, e.g. Float64[f(i,j) for i = 1:3, j = 1:3]")
        end
        n_rng = length(ex.args) - 1
        rng_args = (ex.args[i+1].args[1] for i = 1:n_rng)
        rngs = Any[Core.eval(mod, ex.args[i+1].args[2]) for i = 1:n_rng]
        exprs = (:(f($(j...))) for j in Iterators.product(rngs...))
        return quote
            let
                f($(escall(rng_args)...)) = $(esc(ex.args[1]))
                $SA{Tuple{$(size(exprs)...)},$T}($tuple($(exprs...)))
            end
        end
    elseif head === :call
        f = ex.args[1]
        fargs = ex.args[2:end]
        if f === :zeros || f === :ones
            _f_with_Val = Symbol(:_, f, :_with_Val)
            if length(fargs) == 0
                # for calls like `zeros()`
                return :($f($SA{Tuple{},$Float64}))
            elseif _isnonnegvec(fargs)
                # for calls like `zeros(dims...)`
                return :($f($SA{Tuple{$(escall(fargs)...)}}))
            else
                # for calls like `zeros(type)`
                # for calls like `zeros(type, dims...)`
                return :($_f_with_Val($SA, $(esc(fargs[1])), Val($(esc(fargs[1]))), Val(tuple($(escall(fargs[2:end])...)))))
            end
        elseif f === :fill
            # for calls like `fill(value, dims...)`
            return :($f($(esc(fargs[1])), $SA{Tuple{$(escall(fargs[2:end])...)}}))
        elseif f === :rand || f === :randn || f === :randexp
            _f_with_Val = Symbol(:_, f, :_with_Val)
            if length(fargs) == 0
                # No support for `@SArray rand()`
                error("@$SA got bad expression: $(ex)")
            elseif _isnonnegvec(fargs)
                # for calls like `rand(dims...)`
                return :($f($SA{Tuple{$(escall(fargs)...)}}))
            elseif length(fargs) ≥ 2
                # for calls like `rand(dim1,    dim2,    dims...)`
                # for calls like `rand(type,    dim1,    dims...)`
                # for calls like `rand(sampler, dim1,    dims...)`
                # for calls like `rand(rng,     dim1,    dims...)`
                # for calls like `rand(rng,     type,    dims...)`
                # for calls like `rand(rng,     sampler, dims...)`
                # for calls like `randn(dim1, dim2, dims...)`
                # for calls like `randn(type, dim1, dims...)`
                # for calls like `randn(rng,  dim1, dims...)`
                # for calls like `randn(rng,  type, dims...)`
                # for calls like `randexp(dim1, dim2, dims...)`
                # for calls like `randexp(type, dim1, dims...)`
                # for calls like `randexp(rng,  dim1, dims...)`
                # for calls like `randexp(rng,  type, dims...)`
                return :($_f_with_Val($SA, $(esc(fargs[1])), $(esc(fargs[2])), _int2val($(esc(fargs[1]))), _int2val($(esc(fargs[2]))), Val(tuple($(escall(fargs[3:end])...)))))
            elseif length(fargs) == 1
                # for calls like `rand(dim)`
                return :($f($SA{Tuple{$(escall(fargs)...)}}))
            else
                error("@$SA got bad expression: $(ex)")
            end
        else
            error("@$SA only supports the zeros(), ones(), fill(), rand(), randn(), and randexp() functions.")
        end
    else
        error("Bad input for @$SA")
    end
end

"""
    @SArray [a b; c d]
    @SArray [[a, b];[c, d]]
    @SArray [i+j for i in 1:2, j in 1:2]
    @SArray ones(2, 2, 2)

A convenience macro to construct `SArray` with arbitrary dimension.
It supports:
1. (typed) array literals.
   !!! note
       Every argument inside the square brackets is treated as a scalar during expansion.
       Thus `@SArray[a; b]` always forms a `SVector{2}` and `@SArray [a b; c]` always throws
       an error.

2. comprehensions
   !!! note
       The range of a comprehension is evaluated at global scope by the macro, and must be
       made of combinations of literal values, functions, or global variables.

3. initialization functions
   !!! note
       Only support `zeros()`, `ones()`, `fill()`, `rand()`, `randn()`, and `randexp()`
"""
macro SArray(ex)
    static_array_gen(SArray, ex, __module__)
end

function promote_rule(::Type{<:SArray{S,T,N,L}}, ::Type{<:SArray{S,U,N,L}}) where {S,T,U,N,L}
    SArray{S,promote_type(T,U),N,L}
end
