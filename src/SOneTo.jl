"""
    SOneTo(n)

Return a statically-sized `AbstractUnitRange` starting at `1`, functioning as the `axes` of
a `StaticArray`.
"""
struct SOneTo{n} <: AbstractUnitRange{Int}
end

SOneTo(n::Int) = SOneTo{n}()
function SOneTo{n}(r::AbstractUnitRange) where n
    ((first(r) == 1) & (last(r) == n)) && return SOneTo{n}()

    errmsg(r) = throw(DimensionMismatch("$r is inconsistent with SOneTo{$n}")) # avoid GC frame
    errmsg(r)
end

Base.Tuple(::SOneTo{N}) where N = ntuple(identity, Val(N))

Base.axes(s::SOneTo) = (s,)
Base.size(s::SOneTo) = (length(s),)
Base.length(s::SOneTo{n}) where {n} = n

# The axes of a Slice'd SOneTo use the SOneTo itself
Base.axes(S::Base.Slice{<:SOneTo}) = (S.indices,)
Base.unsafe_indices(S::Base.Slice{<:SOneTo}) = (S.indices,)
Base.axes1(S::Base.Slice{<:SOneTo}) = S.indices

@propagate_inbounds function Base.getindex(s::SOneTo, i::Int)
    @boundscheck checkbounds(s, i)
    return i
end
@propagate_inbounds function Base.getindex(s::SOneTo, s2::SOneTo)
    @boundscheck checkbounds(s, s2)
    return s2
end

Base.first(::SOneTo) = 1
Base.last(::SOneTo{n}) where {n} = n::Int

function Base.iterate(::SOneTo{n}) where {n}
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

function Base.show(io::IO, @nospecialize(x::SOneTo))
    print(io, "SOneTo(", length(x)::Int, ")")
end

Base.@pure function Base.checkindex(::Type{Bool}, ::SOneTo{n1}, ::SOneTo{n2}) where {n1, n2}
    return n1::Int >= n2::Int
end

Base.promote_rule(a::Type{Base.OneTo{T}}, ::Type{SOneTo{n}}) where {T,n} =
    Base.OneTo{promote_type(T, Int)}

function Base.reduced_indices(inds::Tuple{SOneTo,Vararg{SOneTo}}, d::Int)
    Base.reduced_indices(map(Base.OneTo, inds), d)
end

Base.intersect(r::SOneTo{n1}, s::SOneTo{n2}) where {n1,n2} = SOneTo(min(n1, n2))
Base.union(r::SOneTo{n1}, s::SOneTo{n2}) where {n1,n2} = SOneTo(max(n1, n2))
