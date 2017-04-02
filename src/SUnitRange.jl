# `Start` and `End` should be `Int`
struct SUnitRange{Start, End, L} <: StaticVector{L, Int}
    function SUnitRange{Start, End, L}() where {Start, End, L}
        check_sunitrange_params(Start, End, L)
        new{Start, End, L}()
    end
end

@pure function check_sunitrange_params(a::Int, b::Int, c::Int)
    if max(0, b - a + 1) != c
        throw(DimensionMismatch("Static unit range $a:$b does not have length $c"))
    end
end

function check_sunitrange_params(a, b, c)
    throw(TypeError(:SUnitRange, "type parameters must be `Int`s", Tuple{Int, Int, Int}, Tuple{typeof(a), typeof(b), typeof(c)}))
end

@pure SUnitRange{Start, End}() where {Start, End} = SUnitRange{Start, End, max(0, End - Start + 1)}()
@pure SUnitRange(a::Int, b::Int) = SUnitRange{a, b}()

@pure @propagate_inbounds function getindex(x::SUnitRange{Start,End}, i::Int) where {Start, End}
    @boundscheck if i < Start || i > End
        throw(BoundsError(x, i))
    end
    return i
end
