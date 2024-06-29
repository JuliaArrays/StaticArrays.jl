@inline zeros(::Type{SA}) where {SA <: StaticArray{<:Tuple}} = zeros(typeintersect(SA, AbstractArray{Float64}))
@inline zeros(::Type{SA}) where {SA <: StaticArray{<:Tuple, T}} where T = _zeros(Size(SA), SA)
@generated function _zeros(::Size{s}, ::Type{SA}) where {s, SA <: StaticArray}
    T = eltype(SA)
    v = [:(zero($T)) for i = 1:prod(s)]
    if SA <: SArray
        SA = SArray{Tuple{s...}, T, length(s), prod(s)}
    elseif SA <: MArray
        SA = MArray{Tuple{s...}, T, length(s), prod(s)}
    elseif SA <: SizedArray
        SA = SizedArray{Tuple{s...}, T, length(s)}
    end
    return quote
        @_inline_meta
        $SA(tuple($(v...)))
    end
end

@inline ones(::Type{SA}) where {SA <: StaticArray{<:Tuple}} = ones(typeintersect(SA, AbstractArray{Float64}))
@inline ones(::Type{SA}) where {SA <: StaticArray{<:Tuple, T}} where T = _ones(Size(SA), SA)
@generated function _ones(::Size{s}, ::Type{SA}) where {s, SA <: StaticArray}
    T = eltype(SA)
    v = [:(one($T)) for i = 1:prod(s)]
    if SA <: SArray
        SA = SArray{Tuple{s...}, T, length(s), prod(s)}
    elseif SA <: MArray
        SA = MArray{Tuple{s...}, T, length(s), prod(s)}
    elseif SA <: SizedArray
        SA = SizedArray{Tuple{s...}, T, length(s)}
    end
    return quote
        @_inline_meta
        $SA(tuple($(v...)))
    end
end

@inline fill(val, ::SA) where {SA <: StaticArray{<:Tuple}} = _fill(val, Size(SA), SA)
@inline fill(val::U, ::Type{SA}) where {SA <: StaticArray} where U = fill(val, Base.typeintersect(SA, AbstractArray{U}))
@inline fill(val, ::Type{SA}) where {SA <: StaticArray{<:Tuple, T}} where T = _fill(val, Size(SA), SA)
@generated function _fill(val, ::Size{s}, ::Type{SA}) where {s, SA <: StaticArray}
    T = eltype(SA)
    v = [:val for i = 1:prod(s)]
    if SA <: SArray
        SA = SArray{Tuple{s...}, T, length(s), prod(s)}
    elseif SA <: MArray
        SA = MArray{Tuple{s...}, T, length(s), prod(s)}
    elseif SA <: SizedArray
        SA = SizedArray{Tuple{s...}, T, length(s)}
    end
    return quote
        @_inline_meta
        $SA(tuple($(v...)))
    end
end

# Also consider randcycle, randperm? Also faster rand!(staticarray, collection)

using Random: SamplerType
@inline rand(rng::AbstractRNG, ::Type{SA}, dims::Dims) where {SA <: StaticArray} = rand!(rng, Array{SA}(undef, dims), SA)
@inline rand(rng::AbstractRNG, ::SamplerType{SA}) where {SA <: StaticArray} = _rand(rng, Size(SA), SA)

@generated function _rand(rng::AbstractRNG, ::Size{s}, ::Type{SA}) where {s, SA <: StaticArray}
    T = eltype(SA)
    if T == Any
        T = Float64
    end
    v = [:(rand(rng, $T)) for i = 1:prod(s)]
    if SA <: SArray
        SA = SArray{Tuple{s...}, T, length(s), prod(s)}
    elseif SA <: MArray
        SA = MArray{Tuple{s...}, T, length(s), prod(s)}
    elseif SA <: SizedArray
        SA = SizedArray{Tuple{s...}, T, length(s)}
    end
    return quote
        @_inline_meta
        $SA(tuple($(v...)))
    end
end

@inline rand(rng::AbstractRNG, range::AbstractArray, ::Type{SA}) where {SA <: StaticArray} = _rand(rng, range, Size(SA), SA)
@inline rand(range::AbstractArray, ::Type{SA}) where {SA <: StaticArray} = _rand(Random.GLOBAL_RNG, range, Size(SA), SA)
@generated function _rand(rng::AbstractRNG, X, ::Size{s}, ::Type{SA}) where {s, SA <: StaticArray}
    v = [:(rand(rng, X)) for i = 1:prod(s)]
    return quote
        @_inline_meta
        $SA(tuple($(v...)))
    end
end

#@inline rand(rng::MersenneTwister, range::AbstractArray, ::Type{SA}) where {SA <: StaticArray} = _rand(rng, range, Size(SA), SA)

@inline randn(rng::AbstractRNG, ::Type{SA}) where {SA <: StaticArray} = _randn(rng, Size(SA), SA)
@generated function _randn(rng::AbstractRNG, ::Size{s}, ::Type{SA}) where {s, SA <: StaticArray}
    T = eltype(SA)
    if T == Any
        T = Float64
    end
    v = [:(randn(rng, $T)) for i = 1:prod(s)]
    if SA <: SArray
        SA = SArray{Tuple{s...}, T, length(s), prod(s)}
    elseif SA <: MArray
        SA = MArray{Tuple{s...}, T, length(s), prod(s)}
    elseif SA <: SizedArray
        SA = SizedArray{Tuple{s...}, T, length(s)}
    end
    return quote
        @_inline_meta
        $SA(tuple($(v...)))
    end
end

@inline randexp(rng::AbstractRNG, ::Type{SA}) where {SA <: StaticArray} = _randexp(rng, Size(SA), SA)
@generated function _randexp(rng::AbstractRNG, ::Size{s}, ::Type{SA}) where {s, SA <: StaticArray}
    T = eltype(SA)
    if T == Any
        T = Float64
    end
    v = [:(randexp(rng, $T)) for i = 1:prod(s)]
    if SA <: SArray
        SA = SArray{Tuple{s...}, T, length(s), prod(s)}
    elseif SA <: MArray
        SA = MArray{Tuple{s...}, T, length(s), prod(s)}
    elseif SA <: SizedArray
        SA = SizedArray{Tuple{s...}, T, length(s)}
    end
    return quote
        @_inline_meta
        $SA(tuple($(v...)))
    end
end

# Mutable versions

# Why don't these two exist in Base?
# @generated function Base.zeros!{SA <: StaticArray}(a::SA)
# @generated function Base.ones!{SA <: StaticArray}(a::SA)

@inline fill!(a::SA, val) where {SA <: StaticArray} = _fill!(Size(SA), a, val)
@generated function _fill!(::Size{s}, a::SA, val) where {s, SA <: StaticArray}
    exprs = [:(a[$i] = valT) for i = 1:prod(s)]
    return quote
        @_inline_meta
        valT = convert(eltype(SA), val)
        @inbounds $(Expr(:block, exprs...))
        return a
    end
end

@inline rand!(rng::AbstractRNG, a::SA) where {SA <: StaticArray} = _rand!(rng, Size(SA), a)
@generated function _rand!(rng::AbstractRNG, ::Size{s}, a::SA) where {s, SA <: StaticArray}
    exprs = [:(a[$i] = rand(rng, eltype(SA))) for i = 1:prod(s)]
    return quote
        @_inline_meta
        @inbounds $(Expr(:block, exprs...))
        return a
    end
end

@inline rand!(rng::MersenneTwister, a::SA) where {SA <: StaticArray{<:Tuple, Float64}} = _rand!(rng, Size(SA), a)

@inline randn!(rng::AbstractRNG, a::SA) where {SA <: StaticArray} = _randn!(rng, Size(SA), a)
@generated function _randn!(rng::AbstractRNG, ::Size{s}, a::SA) where {s, SA <: StaticArray}
    exprs = [:(a[$i] = randn(rng, eltype(SA))) for i = 1:prod(s)]
    return quote
        @_inline_meta
        @inbounds $(Expr(:block, exprs...))
        return a
    end
end

@inline randexp!(rng::AbstractRNG, a::SA) where {SA <: StaticArray} = _randexp!(rng, Size(SA), a)
@generated function _randexp!(rng::AbstractRNG, ::Size{s}, a::SA) where {s, SA <: StaticArray}
    exprs = [:(a[$i] = randexp(rng, eltype(SA))) for i = 1:prod(s)]
    return quote
        @_inline_meta
        @inbounds $(Expr(:block, exprs...))
        return a
    end
end

"""
    arithmetic_closure(T)

Return the type which values of type `T` will promote to under a combination of the arithmetic operations `+`, `-`, `*` and `/`.

```jldoctest
julia> import StaticArrays.arithmetic_closure

julia> arithmetic_closure(Bool)
Float64

julia> arithmetic_closure(Int32)
Float64

julia> arithmetic_closure(BigFloat)
BigFloat

julia> arithmetic_closure(BigInt)
BigFloat
```
"""
arithmetic_closure(::Type{T}) where T = typeof((one(T)*zero(T) + zero(T))/one(T))
