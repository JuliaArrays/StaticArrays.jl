import Base: .+, .-, .*, ./

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

@generated function Base.zeros{SA <: StaticArray}(::Type{SA})
    s = size(SA)
    T = eltype(SA)
    v = zeros(T, s...)
    return quote
        $(Expr(:meta, :inline))
        $(Expr(:call, SA, Expr(:tuple, v...)))
    end
end

@generated function Base.ones{SA <: StaticArray}(::Type{SA})
    s = size(SA)
    T = eltype(SA)
    v = ones(T, s...)
    return quote
        $(Expr(:meta, :inline))
        $(Expr(:call, SA, Expr(:tuple, v...)))
    end
end
