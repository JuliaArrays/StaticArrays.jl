import Base: .+, .-, .*, ./

# Support for elementwise ops on AbstractArray{S<:StaticArray} with Number
Base.promote_op{Op,A<:StaticArray,T<:Number}(op::Op, ::Type{A}, ::Type{T}) = similar_type(A, promote_op(op, eltype(A), T))
Base.promote_op{Op,T<:Number,A<:StaticArray}(op::Op, ::Type{T}, ::Type{A}) = similar_type(A, promote_op(op, T, eltype(A)))


# TODO lots more operators

@inline .-(a1::StaticArray) = broadcast(-, a1)

@inline .+(a1::StaticArray, a2::StaticArray) = broadcast(+, a1, a2)
@inline .-(a1::StaticArray, a2::StaticArray) = broadcast(-, a1, a2)
@inline .*(a1::StaticArray, a2::StaticArray) = broadcast(*, a1, a2)
@inline ./(a1::StaticArray, a2::StaticArray) = broadcast(/, a1, a2)

@inline .+(a1::StaticArray, a2::Number) = broadcast(+, a1, a2)
@inline .-(a1::StaticArray, a2::Number) = broadcast(-, a1, a2)
@inline .*(a1::StaticArray, a2::Number) = broadcast(*, a1, a2)
@inline ./(a1::StaticArray, a2::Number) = broadcast(/, a1, a2)

@inline .+(a1::Number, a2::StaticArray) = broadcast(+, a1, a2)
@inline .-(a1::Number, a2::StaticArray) = broadcast(-, a1, a2)
@inline .*(a1::Number, a2::StaticArray) = broadcast(*, a1, a2)
@inline ./(a1::Number, a2::StaticArray) = broadcast(/, a1, a2)

@generated function Base.zeros{SA <: StaticArray}(::Union{SA,Type{SA}})
    s = size(SA)
    T = eltype(SA)
    if T == Any
        T = Float64
    end
    v = zeros(T, s...)
    return quote
        $(Expr(:meta, :inline))
        $(Expr(:call, SA, Expr(:tuple, v...)))
    end
end
@inline Base.zero{SA <: StaticArray}(a::Union{SA,Type{SA}}) = zeros(a)

@generated function Base.ones{SA <: StaticArray}(::Union{SA,Type{SA}})
    s = size(SA)
    T = eltype(SA)
    if T == Any
        T = Float64
    end
    v = ones(T, s...)
    return quote
        $(Expr(:meta, :inline))
        $(Expr(:call, SA, Expr(:tuple, v...)))
    end
end

@generated function Base.fill{SA <: StaticArray}(val, ::Union{SA,Type{SA}})
    l = length(SA)
    T = eltype(SA)
    expr = [:valT for i = 1:l]
    return quote
        $(Expr(:meta, :inline))
        valT = convert($T, val)
        SA($(expr...))
    end
end

# TODO allow ranges/collections as inputs...
# Signatures = rand{SA <: StaticArray}(::AbstractRNG, range::AbstractArray, dims::Union{SA, Type{SA}})
#              rand{SA <: StaticArray}(range::AbstractArray, dims::Union{SA, Type{SA}})
# Also consider randcycle, randperm?
@generated function Base.rand{SA <: StaticArray}(rng::AbstractRNG, ::Union{SA,Type{SA}})
    s = size(SA)
    T = eltype(SA)
    if T == Any
        T = Float64
    end
    v = [:(rand(rng, $T)) for i = 1:prod(s)]
    return quote
        $(Expr(:meta, :inline))
        $(Expr(:call, SA, Expr(:tuple, v...)))
    end
end

@generated function Base.randn{SA <: StaticArray}(rng::AbstractRNG, ::Union{SA,Type{SA}})
    s = size(SA)
    T = eltype(SA)
    if T == Any
        T = Float64
    end
    v = [:(randn(rng, $T)) for i = 1:prod(s)]
    return quote
        $(Expr(:meta, :inline))
        $(Expr(:call, SA, Expr(:tuple, v...)))
    end
end

@generated function Base.randexp{SA <: StaticArray}(rng::AbstractRNG, ::Union{SA,Type{SA}})
    s = size(SA)
    T = eltype(SA)
    if T == Any
        T = Float64
    end
    v = [:(randexp(rng, $T)) for i = 1:prod(s)]
    return quote
        $(Expr(:meta, :inline))
        $(Expr(:call, SA, Expr(:tuple, v...)))
    end
end

# Why don't these two exist in Base?
# @generated function Base.zeros!{SA <: StaticArray}(a::SA)
# @generated function Base.ones!{SA <: StaticArray}(a::SA)

@generated function Base.fill!{SA <: StaticArray}(a::SA, val)
    l = length(SA)
    T = eltype(SA)
    exprs = [:(@inbounds a[$i] = valT) for i = 1:l]
    return quote
        $(Expr(:meta, :inline))
        valT = convert($T, val)
        $(Expr(:block, exprs...))
        return a
    end
end

@generated function Base.rand!{SA <: StaticArray}(rng::AbstractRNG, a::SA)
    l = length(SA)
    T = eltype(SA)
    exprs = [:(@inbounds a[$i] = rand(rng, $T)) for i = 1:l]
    return quote
        $(Expr(:meta, :inline))
        $(Expr(:block, exprs...))
        return a
    end
end

@generated function Base.rand!{N}(rng::MersenneTwister, a::StaticArray{Float64,N}) # ambiguity with AbstractRNG and non-Float64... possibly an optimized form in Base?
    l = length(a)
    exprs = [:(@inbounds a[$i] = rand(rng, Float64)) for i = 1:l]
    return quote
        $(Expr(:meta, :inline))
        $(Expr(:block, exprs...))
        return a
    end
end

@generated function Base.randn!{SA <: StaticArray}(rng::AbstractRNG, a::SA)
    l = length(SA)
    T = eltype(SA)
    exprs = [:(@inbounds a[$i] = randn(rng, $T)) for i = 1:l]
    return quote
        $(Expr(:meta, :inline))
        $(Expr(:block, exprs...))
        return a
    end
end

@generated function Base.randexp!{SA <: StaticArray}(rng::AbstractRNG, a::SA)
    l = length(SA)
    T = eltype(SA)
    exprs = [:(@inbounds a[$i] = randexp(rng, $T)) for i = 1:l]
    return quote
        $(Expr(:meta, :inline))
        $(Expr(:block, exprs...))
        return a
    end
end
