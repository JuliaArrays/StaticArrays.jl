@inline zeros(::SA) where {SA <: StaticArray} = zeros(SA)
@generated function zeros(::Type{SA}) where {SA <: StaticArray}
    T = eltype(SA)
    if T === Any
        return quote
            @_inline_meta
            _fill(zero(Float64), Size(SA), SA)
        end
    else
        return quote
            @_inline_meta
            _fill(zero($T), Size(SA), SA)
        end
    end
end

@inline ones(::SA) where {SA <: StaticArray} = ones(SA)
@generated function ones(::Type{SA}) where {SA <: StaticArray}
    T = eltype(SA)
    if T === Any
        return quote
            @_inline_meta
            _fill(one(Float64), Size(SA), SA)
        end
    else
        return quote
            @_inline_meta
            _fill(one($T), Size(SA), SA)
        end
    end
end

@inline fill(val, ::SA) where {SA <: StaticArray} = _fill(val, Size(SA), SA)
@inline fill(val, ::Type{SA}) where {SA <: StaticArray} = _fill(val, Size(SA), SA)
@generated function _fill(val, ::Size{s}, ::Type{SA}) where {s, SA <: StaticArray}
    v = [:val for i = 1:prod(s)]
    return quote
        @_inline_meta
        $SA(tuple($(v...)))
    end
end

# Also consider randcycle, randperm? Also faster rand!(staticarray, collection)

@inline rand(rng::AbstractRNG, ::SA) where {SA <: StaticArray} = rand(rng, SA)
@inline rand(rng::AbstractRNG, ::Type{SA}) where {SA <: StaticArray} = _rand(rng, Size(SA), SA)
@generated function _rand(rng::AbstractRNG, ::Size{s}, ::Type{SA}) where {s, SA <: StaticArray}
    T = eltype(SA)
    if T == Any
        T = Float64
    end
    v = [:(rand(rng, $T)) for i = 1:prod(s)]
    return quote
        @_inline_meta
        $SA(tuple($(v...)))
    end
end

@inline rand(rng::AbstractRNG, range::AbstractArray, ::SA) where {SA <: StaticArray} = rand(rng, range, SA)
@inline rand(rng::AbstractRNG, range::AbstractArray, ::Type{SA}) where {SA <: StaticArray} = _rand(rng, range, Size(SA), SA)
@generated function _rand(rng::AbstractRNG, range::AbstractArray, ::Size{s}, ::Type{SA}) where {s, SA <: StaticArray}
    v = [:(rand(rng, range)) for i = 1:prod(s)]
    return quote
        @_inline_meta
        $SA(tuple($(v...)))
    end
end

@inline randn(rng::AbstractRNG, ::SA) where {SA <: StaticArray} = randn(rng, SA)
@inline randn(rng::AbstractRNG, ::Type{SA}) where {SA <: StaticArray} = _randn(rng, Size(SA), SA)
@generated function _randn(rng::AbstractRNG, ::Size{s}, ::Type{SA}) where {s, SA <: StaticArray}
    T = eltype(SA)
    if T == Any
        T = Float64
    end
    v = [:(randn(rng, $T)) for i = 1:prod(s)]
    return quote
        @_inline_meta
        $SA(tuple($(v...)))
    end
end

@inline randexp(rng::AbstractRNG, ::SA) where {SA <: StaticArray} = randexp(rng, SA)
@inline randexp(rng::AbstractRNG, ::Type{SA}) where {SA <: StaticArray} = _randexp(rng, Size(SA), SA)
@generated function _randexp(rng::AbstractRNG, ::Size{s}, ::Type{SA}) where {s, SA <: StaticArray}
    T = eltype(SA)
    if T == Any
        T = Float64
    end
    v = [:(randexp(rng, $T)) for i = 1:prod(s)]
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

# ambiguity with AbstractRNG and non-Float64... possibly an optimized form in Base?
@inline rand!(rng::MersenneTwister, a::SA) where {SA <: StaticArray{<:Any, Float64}} = _rand!(rng, Size(SA), a)

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
