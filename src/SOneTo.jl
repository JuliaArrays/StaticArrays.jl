"""
    SOneTo(n)

Return a statically-sized `AbstractUnitRange` starting at `1`, functioning as the `axes` of
a `StaticArray`.
"""
struct SOneTo{n} <: AbstractUnitRange{Int}
end

SOneTo(n::Int) = SOneTo{n}()

Base.axes(s::SOneTo) = (s,)
Base.size(s::SOneTo) = (length(s),)
Base.length(s::SOneTo{n}) where {n} = n

function Base.getindex(s::SOneTo, i::Int) 
    @boundscheck checkbounds(s, i)
    return i
end
function Base.getindex(s::SOneTo, s2::SOneTo)
    @boundscheck checkbounds(s, s2)
    return s2
end

Base.first(::SOneTo) = 1
Base.last(::SOneTo{n}) where {n} = n::Int

@pure function Base.iterate(::SOneTo{n}) where {n}
    if n::Int < 1
        return nothing
    else
        (1, 1)
    end
end
function Base.iterate(::SOneTo{n}, s::Int) where {n}
    if s < n::Int
        s2 = s + 1
        return (s2, s2)
    else
        return nothing
    end
end

function Base.getproperty(::SOneTo{n}, s::Symbol) where {n}
    if s === :start
        return 1
    elseif s === :stop
        return n::Int
    else
        error("type SOneTo has no property $s")
    end
end

function Base.show(io::IO, ::SOneTo{n}) where {n}
    print(io, "SOneTo(", n::Int, ")")
end

Base.@pure function Base.checkindex(::Type{Bool}, ::SOneTo{n1}, ::SOneTo{n2}) where {n1, n2}
    return n1::Int >= n2::Int
end

Base.promote_rule(a::Type{Base.OneTo{T}}, ::Type{SOneTo{n}}) where {T,n} =
    Base.OneTo{promote_type(T, Int)}
