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
    expr = Expr(:tuple, fill(:val, l)...)
    return quote
        $(Expr(:meta, :inline))
        $(Expr(:call, SA, expr))
    end
end

@generated function Base.fill!{SA <: StaticArray}(a::SA, val)
    l = length(SA)
    exprs = [:(@inbounds a[$i] = val) for i = 1:l]
    return quote
        $(Expr(:meta, :inline))
        $(Expr(:block, exprs...))
        return a
    end
end

@generated function Base.rand{SA <: StaticArray}(::Union{SA,Type{SA}})
    s = size(SA)
    T = eltype(SA)
    if T == Any
        T = Float64
    end
    v = [:(rand($T)) for i = 1:prod(s)]
    return quote
        $(Expr(:meta, :inline))
        $(Expr(:call, SA, Expr(:tuple, v...)))
    end
end

@generated function Base.randn{SA <: StaticArray}(::Union{SA,Type{SA}})
    s = size(SA)
    T = eltype(SA)
    if T == Any
        T = Float64
    end
    v = [:(randn($T)) for i = 1:prod(s)]
    return quote
        $(Expr(:meta, :inline))
        $(Expr(:call, SA, Expr(:tuple, v...)))
    end
end
