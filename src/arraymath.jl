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
