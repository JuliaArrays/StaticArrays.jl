#####################
## SVector methods ##
#####################

# Converting a CartesianIndex to an SVector
convert(::Type{SVector}, I::CartesianIndex) = SVector(I.I)
convert(::Type{SVector{N}}, I::CartesianIndex{N}) where {N} = SVector{N}(I.I)
convert(::Type{SVector{N,T}}, I::CartesianIndex{N}) where {N,T} = SVector{N,T}(I.I)

Base.promote_rule(::Type{SVector{N,T}}, ::Type{CartesianIndex{N}}) where {N,T} = SVector{N,promote_type(T,Int)}

function check_vector_length(x::Tuple, T = :S)
    if length(x) > 1
        all(isone, Base.tail(x)) || error("Bad input for @$(T)Vector, must be vector like")
    end
    length(x) >= 1 ? x[1] : 1
end

# @SVector rand(...)
@inline _rand_with_Val(::Type{SV}, rng::AbstractRNG, ::Val{n}) where {SV, n} = rand(rng, SV{n})
@inline _rand_with_Val(::Type{SV}, T::DataType,      ::Val{n}) where {SV, n} = _rand(Random.GLOBAL_RNG, T, Size(n), SV{n, T})
@inline _rand_with_Val(::Type{SV}, sampler,          ::Val{n}) where {SV, n} = _rand(Random.GLOBAL_RNG, sampler, Size(n), SV{n, Random.gentype(sampler)})
@inline _rand_with_Val(::Type{SV}, rng::AbstractRNG, T::DataType, ::Val{n}) where {SV, n} = rand(rng, SV{n, T})
@inline _rand_with_Val(::Type{SV}, rng::AbstractRNG, sampler,     ::Val{n}) where {SV, n} = _rand(rng, sampler, Size(n), SV{n, Random.gentype(sampler)})
# @SVector randn(...)
@inline _randn_with_Val(::Type{SV}, rng::AbstractRNG, ::Val{n}) where {SV, n} = randn(rng, SV{n})
@inline _randn_with_Val(::Type{SV}, T::DataType,      ::Val{n}) where {SV, n} = _randn(Random.GLOBAL_RNG, Size(n), SV{n, T})
@inline _randn_with_Val(::Type{SV}, rng::AbstractRNG, T::DataType, ::Val{n}) where {SV, n} = randn(rng, SV{n, T})
# @SVector randexp(...)
@inline _randexp_with_Val(::Type{SV}, rng::AbstractRNG, ::Val{n}) where {SV, n} = randexp(rng, SV{n})
@inline _randexp_with_Val(::Type{SV}, T::DataType,      ::Val{n}) where {SV, n} = _randexp(Random.GLOBAL_RNG, Size(n), SV{n, T})
@inline _randexp_with_Val(::Type{SV}, rng::AbstractRNG, T::DataType, ::Val{n}) where {SV, n} = randexp(rng, SV{n, T})

function static_vector_gen(::Type{SV}, @nospecialize(ex), mod::Module) where {SV}
    if !isa(ex, Expr)
        error("Bad input for @$SV")
    end
    head = ex.head
    if head === :vect
        return :($SV{$(length(ex.args))}($tuple($(escall(ex.args)...))))
    elseif head === :ref
        return :($SV{$(length(ex.args)-1),$(esc(ex.args[1]))}($tuple($(escall(ex.args[2:end])...))))
    elseif head === :typed_vcat || head === :typed_hcat || head === :typed_ncat # typed, cat
        args = parse_cat_ast(ex)
        len = check_vector_length(size(args))
        return :($SV{$len,$(esc(ex.args[1]))}($tuple($(escall(args)...))))
    elseif head === :vcat || head === :hcat || head === :ncat # untyped, cat
        args = parse_cat_ast(ex)
        len = check_vector_length(size(args))
        return :($SV{$len}($tuple($(escall(args)...))))
    elseif head === :comprehension
        if length(ex.args) != 1
            error("Expected generator in comprehension, e.g. [f(i) for i = 1:3]")
        end
        ex = ex.args[1]
        if !isa(ex, Expr) || (ex::Expr).head != :generator
            error("Expected generator in comprehension, e.g. [f(i) for i = 1:3]")
        end
        if length(ex.args) != 2
            error("Use a one-dimensional comprehension for @$SV")
        end
        rng = Core.eval(mod, ex.args[2].args[2])
        exprs = (:(f($j)) for j in rng)
        return quote
            let
                f($(esc(ex.args[2].args[1]))) = $(esc(ex.args[1]))
                $SV{$(length(rng))}($tuple($(exprs...)))
            end
        end
    elseif head === :typed_comprehension
        if length(ex.args) != 2
            error("Expected generator in typed comprehension, e.g. Float64[f(i) for i = 1:3]")
        end
        T = esc(ex.args[1])
        ex = ex.args[2]
        if !isa(ex, Expr) || (ex::Expr).head != :generator
            error("Expected generator in typed comprehension, e.g. Float64[f(i) for i = 1:3]")
        end
        if length(ex.args) != 2
            error("Use a one-dimensional comprehension for @$SV")
        end
        rng = Core.eval(mod, ex.args[2].args[2])
        exprs = (:(f($j)) for j in rng)
        return quote
            let 
                f($(esc(ex.args[2].args[1]))) = $(esc(ex.args[1]))
                $SV{$(length(rng)),$T}($tuple($(exprs...)))
            end
        end
    elseif head === :call
        f = ex.args[1]
        fargs = ex.args[2:end]
        if f === :zeros || f === :ones
            if length(fargs) == 1
                # for calls like `zeros(dim)`
                return :($f($SV{$(esc(fargs[1]))}))
            elseif length(fargs) == 2
                # for calls like `zeros(type, dim)`
                return :($f($SV{$(esc(fargs[2])), $(esc(fargs[1]))}))
            else
                error("@$SV got bad expression: $(ex)")
            end
        elseif f === :fill
            # for calls like `fill(value, dim)`
            if length(fargs) == 2
                return :($f($(esc(fargs[1])), $SV{$(esc(fargs[2]))}))
            else
                error("@$SV expected a 1-dimensional array expression")
            end
        elseif f === :rand || f === :randn || f === :randexp
            _f_with_Val = Symbol(:_, f, :_with_Val)
            if length(fargs) == 1
                # for calls like `rand(dim)`
                # for calls like `randn(dim)`
                # for calls like `randexp(dim)`
                return :($f($SV{$(escall(fargs)...)}))
            elseif length(fargs) == 2
                # for calls like `rand(rng, dim)`
                # for calls like `rand(type, dim)`
                # for calls like `rand(sampler, dim)`
                # for calls like `randn(rng, dim)`
                # for calls like `randn(type, dim)`
                # for calls like `randexp(rng, dim)`
                # for calls like `randexp(type, dim)`
                return :($_f_with_Val($SV, $(esc(fargs[1])), Val($(esc(fargs[2])))))
            elseif length(fargs) == 3
                # for calls like `rand(rng, type, dim)`
                # for calls like `rand(rng, sampler, dim)`
                # for calls like `randn(rng, type, dim)`
                # for calls like `randexp(rng, type, dim)`
                return :($_f_with_Val($SV, $(esc(fargs[1])), $(esc(fargs[2])), Val($(esc(fargs[3])))))
            else
                error("@$SV got bad expression: $(ex)")
            end
        else
            error("@$SV only supports the zeros(), ones(), fill(), rand(), randn(), and randexp() functions.")
        end
    else
        error("Use @$SV [a,b,c], @$SV Type[a,b,c] or a comprehension like @$SV [f(i) for i = i_min:i_max]")
    end
end

"""
    @SVector [a, b, c, d]
    @SVector [i for i in 1:2]
    @SVector ones(2)

A convenience macro to construct `SVector`.
See [`@SArray`](@ref) for detailed features.
"""
macro SVector(ex)
    static_vector_gen(SVector, ex, __module__)
end

