# `Start` and `End` should be `Int`
struct SUnitRange{Start,End} <: StaticVector{Int}
end

@pure Size(::SUnitRange{Start, End}) where {Start,End} = Size(End-Start+1)

@pure @propagate_inbounds function getindex(x::SUnitRange{Start,End}, i::Int) where {Start, End}
    @boundscheck if i < Start || i > End
        throw(BoundsError(x, i))
    end
    return i
end
