@inline _first(a1, as...) = a1

################
## map / map! ##
################

# In 0.6 the three methods below could be replaced with
# `map(f, as::Union{<:StaticArray,AbstractArray}...)` which included at least one `StaticArray`
# this is not the case on 0.7 and we instead hope to find a StaticArray in the first two arguments.
@inline function map(f, a1::StaticArray, as::AbstractArray...)
    _map(f, a1, as...)
end
@inline function map(f, a1::AbstractArray, a2::StaticArray, as::AbstractArray...)
    _map(f, a1, a2, as...)
end
@inline function map(f, a1::StaticArray, a2::StaticArray, as::AbstractArray...)
    _map(f, a1, a2, as...)
end

@generated function _map(f, a::AbstractArray...)
    i = findfirst(ai -> ai <: StaticArray, a)
    if i === nothing
        return :(throw(ArgumentError("No StaticArray found in argument list")))
    end
    # Passing the Size as an argument to _map leads to inference issues when
    # recursively mapping over nested StaticArrays (see issue #593). Calling
    # Size in the generator here is valid because a[i] is known to be a
    # StaticArray for which the default Size method is correct. If wrapped
    # StaticArrays (with a custom Size method) are to be supported, this will
    # no longer be valid.
    S = Size(a[i])
    exprs = Vector{Expr}(undef, prod(S))
    for i ∈ 1:prod(S)
        tmp = [:(a[$j][$i]) for j ∈ 1:length(a)]
        exprs[i] = :(f($(tmp...)))
    end

    return quote
        @_inline_meta
        S = same_size(a...)
        @inbounds elements = tuple($(exprs...))
        @inbounds return similar_type(typeof(_first(a...)), eltype(elements), S)(elements)
    end
end

@inline function map!(f, dest::StaticArray, a::StaticArray...)
    _map!(f, dest, same_size(dest, a...), a...)
end

# Ambiguities with Base:
@inline function map!(f, dest::StaticArray, a::StaticArray)
    _map!(f, dest, same_size(dest, a), a)
end
@inline function map!(f, dest::StaticArray, a::StaticArray, b::StaticArray)
    _map!(f, dest, same_size(dest, a, b), a, b)
end


@generated function _map!(f, dest, ::Size{S}, a::StaticArray...) where {S}
    exprs = Vector{Expr}(undef, prod(S))
    for i ∈ 1:prod(S)
        tmp = [:(a[$j][$i]) for j ∈ 1:length(a)]
        exprs[i] = :(dest[$i] = f($(tmp...)))
    end
    return quote
        @_inline_meta
        @inbounds $(Expr(:block, exprs...))
    end
end

###############
## mapreduce ##
###############

@inline function mapreduce(f, op, a::StaticArray, b::StaticArray...; dims=:,kw...)
    _mapreduce(f, op, dims, kw.data, same_size(a, b...), a, b...)
end

@generated function _mapreduce(f, op, dims::Colon, nt::NamedTuple{()},
                               ::Size{S}, a::StaticArray...) where {S}
    tmp = [:(a[$j][1]) for j ∈ 1:length(a)]
    expr = :(f($(tmp...)))
    for i ∈ 2:prod(S)
        tmp = [:(a[$j][$i]) for j ∈ 1:length(a)]
        expr = :(op($expr, f($(tmp...))))
    end
    return quote
        @_inline_meta
        @inbounds return $expr
    end
end
    
@generated function _mapreduce(f, op, dims::Colon, nt::NamedTuple{(:init,)},
                               ::Size{S}, a::StaticArray...) where {S}    
    expr = :(nt.init)
    for i ∈ 1:prod(S)
        tmp = [:(a[$j][$i]) for j ∈ 1:length(a)]
        expr = :(op($expr, f($(tmp...))))
    end
    return quote
        @_inline_meta
        @inbounds return $expr
    end
end

@inline function _mapreduce(f, op, ::Type{Val{D}}, nt::NamedTuple,sz::Size{S}, a::StaticArray) where {S,D}
    Base.depwarn("`Val{D}` as dims argument is deprecated, use `D` or `Val(D)` instead.")
    _mapreduce(f, op, Val(D), nt, sz, a)
end
@inline _mapreduce(f, op, D::Int, nt::NamedTuple, sz::Size{S}, a::StaticArray) where {S} =
    _mapreduce(f, op, Val(D), nt, sz, a)

    
@generated function _mapreduce(f, op, dims::Val{D}, nt::NamedTuple{()},
                               ::Size{S}, a::StaticArray) where {S,D}
    N = length(S)
    Snew = ([n==D ? 1 : S[n] for n = 1:N]...,)
    
    exprs = Array{Expr}(undef, Snew)
    itr = [1:n for n ∈ Snew]
    for i ∈ Base.product(itr...)
        expr = :(f(a[$(i...)]))
        for k = 2:S[D]
            ik = collect(i)
            ik[D] = k
            expr = :(op($expr, f(a[$(ik...)])))
        end

        exprs[i...] = expr
    end

    return quote
        @_inline_meta
        @inbounds elements = tuple($(exprs...))
        @inbounds return similar_type(a, eltype(elements), Size($Snew))(elements)
    end
end

@generated function _mapreduce(f, op, dims::Val{D}, nt::NamedTuple{(:init,)},
                                  ::Size{S}, a::StaticArray) where {S,D}
    N = length(S)
    Snew = ([n==D ? 1 : S[n] for n = 1:N]...,)

    exprs = Array{Expr}(undef, Snew)
    itr = [1:n for n = Snew]
    for i ∈ Base.product(itr...)
        expr = :(nt.init)
        for k = 1:S[D]
            ik = collect(i)
            ik[D] = k
            expr = :(op($expr, f(a[$(ik...)])))
        end

        exprs[i...] = expr
    end

    return quote
        @_inline_meta
        @inbounds elements = tuple($(exprs...))
        @inbounds return similar_type(a, eltype(elements), Size($Snew))(elements)
    end
end

############
## reduce ##
############

@inline reduce(op, a::StaticArray; kw...) = mapreduce(identity, op, a; kw...)

#######################
## related functions ##
#######################

# These are all similar in Base but not @inline'd
#
# Implementation notes:
#
# 1. mapreduce and mapreducedim usually do not take initial value, because we don't
# always know the return type of an arbitrary mapping function f.  (We usually want to use
# some initial value such as one(T) or zero(T), where T is the return type of f, but
# if users provide type-unstable f, its return type cannot be known.)  Therefore, mapped
# versions of the functions implemented below usually require the collection to have at
# least two entries.
#
# 2. Exceptions are the ones that require Boolean mapping functions.  For example, f in
# all and any must return Bool, so we know the appropriate initial value is true and false,
# respectively.  Therefore, all(f, ...) and any(f, ...) are implemented by mapreduce(f, ...)
# with an initial value v0 = true and false.
#
# TODO: change to use Base.reduce_empty/Base.reduce_first
@inline iszero(a::StaticArray{<:Tuple,T}) where {T} = reduce((x,y) -> x && iszero(y), a, init=true)

@inline sum(a::StaticArray{<:Tuple,T}; dims=:) where {T} = reduce(+, a; dims=dims)
@inline sum(f, a::StaticArray{<:Tuple,T}; dims=:) where {T} = mapreduce(f, +, a; dims=dims)
@inline sum(f::Union{Function, Type}, a::StaticArray{<:Tuple,T}; dims=:) where {T} = mapreduce(f, +, a; dims=dims) # avoid ambiguity

@inline prod(a::StaticArray{<:Tuple,T}; dims=:) where {T} = reduce(*, a; dims=dims)
@inline prod(f, a::StaticArray{<:Tuple,T}; dims=:) where {T} = mapreduce(f, *, a; dims=dims)
@inline prod(f::Union{Function, Type}, a::StaticArray{<:Tuple,T}; dims=:) where {T} = mapreduce(f, *, a; dims=dims)

@inline count(a::StaticArray{<:Tuple,Bool}; dims=:) = reduce(+, a; dims=dims)
@inline count(f, a::StaticArray; dims=:) = mapreduce(x->f(x)::Bool, +, a; dims=dims)

@inline all(a::StaticArray{<:Tuple,Bool}; dims=:) = reduce(&, a; dims=dims, init=true)  # non-branching versions
@inline all(f::Function, a::StaticArray; dims=:) = mapreduce(x->f(x)::Bool, &, a; dims=dims, init=true)

@inline any(a::StaticArray{<:Tuple,Bool}; dims=:) = reduce(|, a; dims=dims, init=false) # (benchmarking needed)
@inline any(f::Function, a::StaticArray; dims=:) = mapreduce(x->f(x)::Bool, |, a; dims=dims, init=false) # (benchmarking needed)

_mean_denom(a, dims::Colon) = length(a)
_mean_denom(a, dims::Int) = size(a, dims)
_mean_denom(a, ::Val{D}) where {D} = size(a, D)
_mean_denom(a, ::Type{Val{D}}) where {D} = size(a, D)

@inline mean(a::StaticArray; dims=:) = sum(a; dims=dims) / _mean_denom(a,dims)
@inline mean(f::Function, a::StaticArray;dims=:) = sum(f, a; dims=dims) / _mean_denom(a,dims)

@inline minimum(a::StaticArray; dims=:) = reduce(min, a; dims=dims) # base has mapreduce(idenity, scalarmin, a)
@inline minimum(f::Function, a::StaticArray; dims=:) = mapreduce(f, min, a; dims=dims)

@inline maximum(a::StaticArray; dims=:) = reduce(max, a; dims=dims) # base has mapreduce(idenity, scalarmax, a)
@inline maximum(f::Function, a::StaticArray; dims=:) = mapreduce(f, max, a; dims=dims)

# Diff is slightly different
@inline diff(a::StaticArray; dims) = _diff(Size(a), a, dims)
@inline diff(a::StaticVector) = diff(a;dims=Val(1))

@inline function _diff(sz::Size{S}, a::StaticArray, D::Int) where {S}
    _diff(sz,a,Val(D))
end
@inline function _diff(sz::Size{S}, a::StaticArray, ::Type{Val{D}}) where {S,D}
    Base.depwarn("`Val{D}` as dims argument is deprecated, use `D` or `Val(D)` instead.")
    _diff(sz,a,Val(D))
end
@generated function _diff(::Size{S}, a::StaticArray, ::Val{D}) where {S,D}
    N = length(S)
    Snew = ([n==D ? S[n]-1 : S[n] for n = 1:N]...,)

    exprs = Array{Expr}(undef, Snew)
    itr = [1:n for n = Snew]

    for i1 = Base.product(itr...)
        i2 = copy([i1...])
        i2[D] = i1[D] + 1
        exprs[i1...] = :(a[$(i2...)] - a[$(i1...)])
    end

    return quote
        @_inline_meta
        T = typeof(one(eltype(a)) - one(eltype(a)))
        @inbounds return similar_type(a, T, Size($Snew))(tuple($(exprs...)))
    end
end
