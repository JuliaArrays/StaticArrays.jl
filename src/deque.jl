"""
    push(vec::StaticVector, item)

Return a new `StaticVector` with `item` inserted on the end of `vec`.

# Examples
```jldoctest
julia> push(@SVector[1, 2, 3], 4)
4-element SVector{4, Int64} with indices SOneTo(4):
 1
 2
 3
 4
```
"""
@inline push(vec::StaticVector, x) = _push(Size(vec), vec, x)
@generated function _push(::Size{s}, vec::StaticVector, x) where {s}
    newlen = s[1] + 1
    exprs = vcat([:(vec[$i]) for i = 1:s[1]], :x)
    return quote
        @_inline_meta
        @inbounds return similar_type(vec, Size($newlen))(tuple($(exprs...)))
    end
end

"""
    pushfirst(vec::StaticVector, item)

Return a new `StaticVector` with `item` inserted at the beginning of `vec`.

# Examples
```jldoctest
julia> pushfirst(@SVector[1, 2, 3, 4], 5)
5-element SVector{5, Int64} with indices SOneTo(5):
 5
 1
 2
 3
 4
```
"""
@inline pushfirst(vec::StaticVector, x) = _pushfirst(Size(vec), vec, x)
@generated function _pushfirst(::Size{s}, vec::StaticVector, x) where {s}
    newlen = s[1] + 1
    exprs = vcat(:x, [:(vec[$i]) for i = 1:s[1]])
    return quote
        @_inline_meta
        @inbounds return similar_type(vec, Size($newlen))(tuple($(exprs...)))
    end
end

"""
    insert(vec::StaticVector, index::Integer, item)

Return a new vector with `item` inserted into `vec` at the given `index`.

# Examples
```jldoctest
julia> insert(@SVector[6, 5, 4, 2, 1], 4, 3)
6-element SVector{6, Int64} with indices SOneTo(6):
 6
 5
 4
 3
 2
 1
```
"""
@propagate_inbounds insert(vec::StaticVector, index, x) = _insert(Size(vec), vec, index, x)
@generated function _insert(::Size{s}, vec::StaticVector, index, x) where {s}
    newlen = s[1] + 1
    exprs = [(i == 1 ? :(if $i < index; vec[$i] else x; end) :
              i == newlen ? :(ifelse($i == index, x, vec[$i-1])) :
              :(ifelse($i < index, vec[$i], ifelse($i == index, x, vec[$i-1])))) for i = 1:newlen]
    return quote
        @_propagate_inbounds_meta
        @boundscheck if (index < 1 || index > $newlen)
            throw(BoundsError(vec, index))
        end
        @inbounds return similar_type(vec, Size($newlen))(tuple($(exprs...)))
    end
end

"""
    pop(vec::StaticVector)

Return a new vector with the last item in `vec` removed.

# Examples
```jldoctest
julia> pop(@SVector[1,2,3])
2-element SVector{2, Int64} with indices SOneTo(2):
 1
 2
```
"""
@inline pop(vec::StaticVector) = _pop(Size(vec), vec)
@generated function _pop(::Size{s}, vec::StaticVector) where {s}
    newlen = s[1] - 1
    exprs = [:(vec[$i]) for i = 1:s[1]-1]
    return quote
        @_inline_meta
        @inbounds return similar_type(vec, Size($newlen))(tuple($(exprs...)))
    end
end

"""
    popfirst(vec::StaticVector)

Return a new vector with the first item in `vec` removed.

# Examples
```jldoctest
julia> popfirst(@SVector[1,2,3])
2-element SVector{2, Int64} with indices SOneTo(2):
 2
 3
```
"""
@inline popfirst(vec::StaticVector) = _popfirst(Size(vec), vec)
@generated function _popfirst(::Size{s}, vec::StaticVector) where {s}
    newlen = s[1] - 1
    exprs = [:(vec[$i]) for i = 2:s[1]]
    return quote
        @_inline_meta
        @inbounds return similar_type(vec, Size($newlen))(tuple($(exprs...)))
    end
end

"""
    deleteat(vec::StaticVector, index::Integer)

Return a new vector with the item at the given `index` removed.

# Examples
```jldoctest
julia> deleteat(@SVector[6, 5, 4, 3, 2, 1], 2)
5-element SVector{5, Int64} with indices SOneTo(5):
 6
 4
 3
 2
 1
```
"""
@propagate_inbounds deleteat(vec::StaticVector, index) = _deleteat(Size(vec), vec, index)
@generated function _deleteat(::Size{s}, vec::StaticVector, index) where {s}
    newlen = s[1] - 1
    exprs = [:(ifelse($i < index, vec[$i], vec[$i+1])) for i = 1:newlen]
    return quote
        @_propagate_inbounds_meta
        @boundscheck if (index < 1 || index > $(s[1]))
            throw(BoundsError(vec, index))
        end
        @inbounds return similar_type(vec, Size($newlen))(tuple($(exprs...)))
    end
end

# TODO consider prepend, append (can use vcat, but eltype might change), and
# maybe splice (a bit hard to get statically sized without a "static" range)


# Immutable version of setindex!(). Seems similar in nature to the above, but
# could also be justified to live in src/indexing.jl
import Base.setindex
"""
    setindex(vec::StaticArray, x, index::Int)

Return a new array with the item at `index` replaced by `x`.

# Examples
```jldoctest
julia> setindex(@SVector[1,2,3], 4, 2)
3-element SVector{3, Int64} with indices SOneTo(3):
 1
 4
 3

julia> setindex(@SMatrix[2 4; 6 8], 1, 2)
2×2 SMatrix{2, 2, Int64, 4} with indices SOneTo(2)×SOneTo(2):
 2  4
 1  8
```
"""
@propagate_inbounds setindex(a::StaticArray, x, index::Int) = _setindex(Length(a), a, convert(eltype(typeof(a)), x), index)
@generated function _setindex(::Length{L}, a::StaticArray{<:Tuple,T}, x::T, index::Int) where {L, T}
    exprs = [:(ifelse($i == index, x, a[$i])) for i = 1:L]
    return quote
        @_propagate_inbounds_meta
        @boundscheck if (index < 1 || index > $(L))
            throw(BoundsError(a, index))
        end
        @inbounds return typeof(a)(tuple($(exprs...)))
    end
end

# TODO proper multidimension boundscheck
@propagate_inbounds setindex(a::StaticArray, x, inds...) = setindex(a, x, LinearIndices(a)[inds...])
