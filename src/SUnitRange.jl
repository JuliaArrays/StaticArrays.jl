# `Start` and `End` should be `Int`
struct SUnitRange{Start, L} <: StaticVector{L, Int}
    function SUnitRange{Start, L}() where {Start, L}
        check_sunitrange_params(L)
        new{Start, L}()
    end
end

function check_sunitrange_params(L::Int)
    if L < 0
        error("Static unit range length is negative")
    end
end

function check_sunitrange_params(L)
    throw(TypeError(:SUnitRange, "type parameters must be `Int`", Tuple{Int,}, Tuple{typeof(L),}))
end

SUnitRange(a::Int, b::Int) = SUnitRange{a, max(0, b - a + 1)}()

@propagate_inbounds function getindex(x::SUnitRange{Start, L}, i::Int) where {Start, L}
    @boundscheck if i < 1 || i > L
        throw(BoundsError(x, i))
    end
    return Start + i - 1
end

function show(io::IO, ::SUnitRange{Start, L}) where {Start, L}
    print(io, "SUnitRange($Start,$(Start + L - 1))")
end

# Shorten show for REPL use.
show(io::IO, ::MIME"text/plain", su::SUnitRange) = show(io, su)
