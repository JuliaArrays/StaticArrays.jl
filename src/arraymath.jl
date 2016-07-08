import Base: .+, .-, .*, ./

# TODO lots more operators

@inline .-(a1::Union{StaticArray, Number}) = broadcast(-, a1)

@inline .+(a1::Union{StaticArray, Number}, a2::Union{StaticArray, Number}) = broadcast(+, a1, a2)
@inline .-(a1::Union{StaticArray, Number}, a2::Union{StaticArray, Number}) = broadcast(-, a1, a2)
@inline .*(a1::Union{StaticArray, Number}, a2::Union{StaticArray, Number}) = broadcast(*, a1, a2)
@inline ./(a1::Union{StaticArray, Number}, a2::Union{StaticArray, Number}) = broadcast(/, a1, a2)
