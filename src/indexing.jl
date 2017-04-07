# Default error messages to help users with new types and to avoid subsequent stack overflows
getindex(a::StaticArray, i::Int) = error("getindex(::$typeof(a), ::Int) is not defined.")
setindex!(a::StaticArray, value, i::Int) = error("setindex!(::$(typeof(a)), value, ::Int) is not defined.")

#######################################
## Multidimensional scalar indexing  ##
#######################################

# Note: all indexing behavior defaults to dense, linear indexing

@propagate_inbounds function getindex(a::StaticArray, inds::Int...)
    _getindex_scalar(Size(a), a, inds...)
end

@generated function _getindex_scalar(::Size{S}, a::StaticArray, inds::Int...) where S
    stride = 1
    for i ∈ 1:length(inds)
        if i == 1
            ind_expr = :(inds[1])
        else
            ind_expr = :($ind_expr + $stride * (inds[$i] - 1))
        end
        stride *= Size(S)[i]
    end
    return quote
        @_propagate_inbounds_meta
        a[$ind_expr]
    end
end

@propagate_inbounds function setindex!(a::StaticArray, value, inds::Int...)
    _setindex!_scalar(Size(a), a, value, inds...)
end

@generated function _setindex!_scalar(::Size{S}, a::StaticArray, value, inds::Int...) where S
    stride = 1
    for i ∈ 1:length(inds)
        if i == 1
            ind_expr = :(inds[1])
        else
            ind_expr = :($ind_expr + $stride * (inds[$i] - 1))
        end
        stride *= S[i]
    end
    return quote
        @_propagate_inbounds_meta
        a[$ind_expr] = value
    end
end

#########################
## Indexing utilities  ##
#########################

@pure tail(::Type{Size{S}}) where {S} = Size{Base.tail(S)}
@inline tail(::S) where {S<:Size} = tail(S)()

@inline index_sizes(s::Size) = ()
@inline index_sizes(s::Size, ::Int, inds...) = (Size(), index_sizes(tail(s), inds...)...)
@inline index_sizes(s::Size, a::StaticArray, inds...) = (Size(a), index_sizes(tail(s), inds...)...)
@inline index_sizes(s::Size, ::Colon, inds...) = (Size(s[1]), index_sizes(tail(s), inds...)...)

@inline index_sizes() = ()
@inline index_sizes(::Int, inds...) = (Size(), index_sizes(inds...)...)
@inline index_sizes(a::StaticArray, inds...) = (Size(a), index_sizes(inds...)...)

out_index_size(ind_sizes::Type{<:Size}...) = Size(_out_index_size((), ind_sizes...))
@inline _out_index_size(t::Tuple) = t
@inline _out_index_size(t::Tuple, ::Type{Size{S}}, ind_sizes...) where {S} = _out_index_size((t..., S...), ind_sizes...)

linear_index_size(ind_sizes::Type{<:Size}...) = _linear_index_size((), ind_sizes...)
@inline _linear_index_size(t::Tuple) = t
@inline _linear_index_size(t::Tuple, ::Type{Size{S}}, ind_sizes...) where {S} = _linear_index_size((t..., prod(S)), ind_sizes...)

_ind(i::Int, ::Int, ::Type{Int}) = :(inds[$i])
_ind(i::Int, j::Int, ::Type{<:StaticArray}) = :(inds[$i][$j])
_ind(i::Int, j::Int, ::Type{Colon}) = j

################################
## Non-scalar linear indexing ##
################################

@inline function getindex(a::StaticArray, ::Colon)
    _getindex(a::StaticArray, Length(a), :)
end

@generated function _getindex(a::StaticArray, ::Length{L}, ::Colon) where {L}
    exprs = [:(a[$i]) for i = 1:L]
    return quote
        @_inline_meta
        @inbounds return similar_type(a, Size(L))(tuple($(exprs...)))
    end
end

@propagate_inbounds function getindex(a::StaticArray, inds::StaticArray{Int})
    _getindex(a, Length(inds), inds)
end

@generated function _getindex(a::StaticArray, ::Length{L}, inds::StaticArray{Int}) where {L}
    exprs = [:(a[inds[$i]]) for i = 1:L]
    return quote
        @_propagate_inbounds_meta
        similar_type(a, Size(L))(tuple($(exprs...)))
    end
end

@inline function setindex!(a::StaticArray, v, ::Colon)
    _setindex!(a::StaticArray, v, Length(a), :)
    return v
end

@generated function _setindex!(a::StaticArray, v, ::Length{L}, ::Colon) where {L}
    exprs = [:(a[$i] = v) for i = 1:L]
    return quote
        @_inline_meta
        @inbounds $(Expr(:block, exprs...))
    end
end

@generated function _setindex!(a::StaticArray, v::AbstractArray, ::Length{L}, ::Colon) where {L}
    exprs = [:(a[$i] = v[$i]) for i = 1:L]
    return quote
        @_propagate_inbounds_meta
        if length(v) != L
            throw(DimensionMismatch("tried to assign $(length(v))-element array to length-$L destination"))
        end
        @inbounds $(Expr(:block, exprs...))
    end
end

@generated function _setindex!(a::StaticArray, v::StaticArray, ::Length{L}, ::Colon) where {L}
    exprs = [:(a[$i] = v[$i]) for i = 1:L]
    return quote
        @_propagate_inbounds_meta
        if Length(typeof(v)) != L
            throw(DimensionMismatch("tried to assign $(length(v))-element array to length-$L destination"))
        end
        $(Expr(:block, exprs...))
    end
end

@propagate_inbounds function setindex!(a::StaticArray, v, inds::StaticArray{Int})
    _setindex!(a, v, Length(inds), inds)
    return v
end

@generated function _setindex!(a::StaticArray, v, ::Length{L}, inds::StaticArray{Int}) where {L}
    exprs = [:(a[inds[$i]] = v) for i = 1:L]
    return quote
        @_propagate_inbounds_meta
        similar_type(a, Size(L))(tuple($(exprs...)))
    end
end

@generated function _setindex!(a::StaticArray, v::AbstractArray, ::Length{L}, inds::StaticArray{Int}) where {L}
    exprs = [:(a[$i] = v[$i]) for i = 1:L]
    return quote
        @_propagate_inbounds_meta
        if length(v) != L
            throw(DimensionMismatch("tried to assign $(length(v))-element array to length-$L destination"))
        end
        $(Expr(:block, exprs...))
    end
end

@generated function _setindex!(a::StaticArray, v::StaticArray, ::Length{L}, inds::StaticArray{Int}) where {L}
    exprs = [:(a[$i] = v[$i]) for i = 1:L]
    return quote
        @_propagate_inbounds_meta
        if Length(typeof(v)) != L
            throw(DimensionMismatch("tried to assign $(length(v))-element array to length-$L destination"))
        end
        $(Expr(:block, exprs...))
    end
end

###########################################
## Multidimensional non-scalar indexing  ##
###########################################

# getindex

@propagate_inbounds function getindex(a::StaticArray, inds::Union{Int, StaticArray{Int}, Colon}...)
    _getindex(a, index_sizes(Size(a), inds...), inds)
end

# Hard to describe "Union{Int, StaticArray{Int}} with at least one StaticArray{Int}"
# Here we require the first StaticArray{Int} to be within the first four dimensions
@propagate_inbounds function getindex(a::AbstractArray, i1::StaticArray{Int}, inds::Union{Int, StaticArray{Int}}...)
    _getindex(a, index_sizes(i1, inds...), (i1, inds...))
end

@propagate_inbounds function getindex(a::AbstractArray, i1::Int, i2::StaticArray{Int}, inds::Union{Int, StaticArray{Int}}...)
    _getindex(a, index_sizes(i1, i2, inds...), (i1, i2, inds...))
end

@propagate_inbounds function getindex(a::AbstractArray, i1::Int, i2::Int, i3::StaticArray{Int}, inds::Union{Int, StaticArray{Int}}...)
    _getindex(a, index_sizes(i1, i2, i3, inds...), (i1, i2, i3, inds...))
end

@propagate_inbounds function getindex(a::AbstractArray, i1::Int, i2::Int, i3::Int, i4::StaticArray{Int}, inds::Union{Int, StaticArray{Int}}...)
    _getindex(a, index_sizes(i1, i2, i3, i4, inds...), (i1, i2, i3, i4, inds...))
end

# Disambuguity methods for the above
@propagate_inbounds function getindex(a::StaticArray, i1::StaticArray{Int}, inds::Union{Int, StaticArray{Int}}...)
    _getindex(a, index_sizes(i1, inds...), (i1, inds...))
end

@propagate_inbounds function getindex(a::StaticArray, i1::Int, i2::StaticArray{Int}, inds::Union{Int, StaticArray{Int}}...)
    _getindex(a, index_sizes(i1, i2, inds...), (i1, i2, inds...))
end

@propagate_inbounds function getindex(a::StaticArray, i1::Int, i2::Int, i3::StaticArray{Int}, inds::Union{Int, StaticArray{Int}}...)
    _getindex(a, index_sizes(i1, i2, i3, inds...), (i1, i2, i3, inds...))
end

@propagate_inbounds function getindex(a::StaticArray, i1::Int, i2::Int, i3::Int, i4::StaticArray{Int}, inds::Union{Int, StaticArray{Int}}...)
    _getindex(a, index_sizes(i1, i2, i3, i4, inds...), (i1, i2, i3, i4, inds...))
end

@generated function _getindex(a::AbstractArray, ind_sizes::Tuple{Vararg{Size}}, inds)
    newsize = out_index_size(ind_sizes.parameters...)
    linearsizes = linear_index_size(ind_sizes.parameters...)
    exprs = Array{Expr}(linearsizes)

    # Iterate over input indices
    ind_types = inds.parameters
    current_ind = ones(Int,length(linearsizes))
    more = linearsizes[1] != 0
    while more
        exprs_tmp = [_ind(i, current_ind[i], ind_types[i]) for i = 1:length(linearsizes)]
        exprs[current_ind...] = :(getindex(a, $(exprs_tmp...)))

        # increment current_ind
        current_ind[1] += 1
        for i ∈ 1:length(linearsizes)
            if current_ind[i] > linearsizes[i]
                if i == length(linearsizes)
                    more = false
                    break
                else
                    current_ind[i] = 1
                    current_ind[i+1] += 1
                end
            else
                break
            end
        end
    end

    quote
        @_propagate_inbounds_meta
        similar_type(a, $newsize)(tuple($(exprs...)))
    end
end

# setindex!

@propagate_inbounds function setindex!(a::StaticArray, value, inds::Union{Int, StaticArray{Int}, Colon}...)
    _setindex!(a, value, index_sizes(Size(a), inds...), inds)
end

# Hard to describe "Union{Int, StaticArray{Int}} with at least one StaticArray{Int}"
# Here we require the first StaticArray{Int} to be within the first four dimensions
@propagate_inbounds function setindex!(a::AbstractArray, value, i1::StaticArray{Int}, inds::Union{Int, StaticArray{Int}}...)
    _setindex!(a, value, index_sizes(i1, inds...), (i1, inds...))
end

@propagate_inbounds function setindex!(a::AbstractArray, value, i1::Int, i2::StaticArray{Int}, inds::Union{Int, StaticArray{Int}}...)
    _setindex!(a, value, index_sizes(i1, i2, inds...), (i1, i2, inds...))
end

@propagate_inbounds function setindex!(a::AbstractArray, value, i1::Int, i2::Int, i3::StaticArray{Int}, inds::Union{Int, StaticArray{Int}}...)
    _setindex!(a, value, index_sizes(i1, i2, i3, inds...), (i1, i2, i3, inds...))
end

@propagate_inbounds function setindex!(a::AbstractArray, value, i1::Int, i2::Int, i3::Int, i4::StaticArray{Int}, inds::Union{Int, StaticArray{Int}}...)
    _setindex!(a, value, index_sizes(i1, i2, i3, i4, inds...), (i1, i2, i3, i4, inds...))
end

# Disambiguity methods for the above
@propagate_inbounds function setindex!(a::StaticArray, value, i1::StaticArray{Int}, inds::Union{Int, StaticArray{Int}}...)
    _setindex!(a, value, index_sizes(i1, inds...), (i1, inds...))
end

@propagate_inbounds function setindex!(a::StaticArray, value, i1::Int, i2::StaticArray{Int}, inds::Union{Int, StaticArray{Int}}...)
    _setindex!(a, value, index_sizes(i1, i2, inds...), (i1, i2, inds...))
end

@propagate_inbounds function setindex!(a::StaticArray, value, i1::Int, i2::Int, i3::StaticArray{Int}, inds::Union{Int, StaticArray{Int}}...)
    _setindex!(a, value, index_sizes(i1, i2, i3, inds...), (i1, i2, i3, inds...))
end

@propagate_inbounds function setindex!(a::StaticArray, value, i1::Int, i2::Int, i3::Int, i4::StaticArray{Int}, inds::Union{Int, StaticArray{Int}}...)
    _setindex!(a, value, index_sizes(i1, i2, i3, i4, inds...), (i1, i2, i3, i4, inds...))
end

# disambiguities from Base
@propagate_inbounds function setindex!(a::Array, value, i1::StaticVector{Int})
    _setindex!(a, value, index_sizes(i1), (i1,))
end

@propagate_inbounds function setindex!(a::Array, value::AbstractArray, i1::StaticVector{Int})
    _setindex!(a, value, index_sizes(i1), (i1,))
end

@propagate_inbounds function setindex!(a::Array, value::AbstractArray, i1::StaticArray{Int}, inds::Union{Int, StaticArray{Int}}...)
    _setindex!(a, value, index_sizes(i1, inds...), (i1, inds...))
end

@propagate_inbounds function setindex!(a::Array, value::AbstractArray, i1::Int, i2::StaticArray{Int}, inds::Union{Int, StaticArray{Int}}...)
    _setindex!(a, value, index_sizes(i1, i2, inds...), (i1, i2, inds...))
end

@propagate_inbounds function setindex!(a::Array, value::AbstractArray, i1::Int, i2::Int, i3::StaticArray{Int}, inds::Union{Int, StaticArray{Int}}...)
    _setindex!(a, value, index_sizes(i1, i2, i3, inds...), (i1, i2, i3, inds...))
end

@propagate_inbounds function setindex!(a::Array, value::AbstractArray, i1::Int, i2::Int, i3::Int, i4::StaticArray{Int}, inds::Union{Int, StaticArray{Int}}...)
    _setindex!(a, value, index_sizes(i1, i2, i3, i4, inds...), (i1, i2, i3, i4, inds...))
end

# setindex! from a scalar
@generated function _setindex!(a::AbstractArray, value, ind_sizes::Tuple{Vararg{Size}}, inds)
    linearsizes = linear_index_size(ind_sizes.parameters...)
    exprs = Array{Expr}(linearsizes)

    # Iterate over input indices
    ind_types = inds.parameters
    current_ind = ones(Int,length(ind_types))
    more = linearsizes[1] != 0
    while more
        exprs_tmp = [_ind(i, current_ind[i], ind_types[i]) for i = 1:length(ind_types)]
        exprs[current_ind...] = :(setindex!(a, value, $(exprs_tmp...)))

        # increment current_ind
        current_ind[1] += 1
        for i ∈ 1:length(linearsizes)
            if current_ind[i] > linearsizes[i]
                if i == length(linearsizes)
                    more = false
                    break
                else
                    current_ind[i] = 1
                    current_ind[i+1] += 1
                end
            else
                break
            end
        end
    end

    quote
        @_propagate_inbounds_meta
        $(exprs...)
        return value
    end
end

# setindex! from an array
@generated function _setindex!(a::AbstractArray, v::AbstractArray, ind_sizes::Tuple{Vararg{Size}}, inds)
    linearsizes = linear_index_size(ind_sizes.parameters...)
    exprs = Array{Expr}(linearsizes)

    # Iterate over input indices
    ind_types = inds.parameters
    current_ind = ones(Int,length(ind_types))
    more = true
    j = 1
    while more
        exprs_tmp = [_ind(i, current_ind[i], ind_types[i]) for i = 1:length(ind_types)]
        exprs[current_ind...] = :(setindex!(a, v[$j], $(exprs_tmp...)))

        # increment current_ind
        current_ind[1] += 1
        for i ∈ 1:length(linearsizes)
            if current_ind[i] > linearsizes[i]
                if i == length(linearsizes)
                    more = false
                    break
                else
                    current_ind[i] = 1
                    current_ind[i+1] += 1
                end
            else
                break
            end
        end
        j += 1
    end

    if v <: StaticArray
        if Length(v) != prod(linearsizes)
            return DimensionMismatch("tried to assign $(length(v))-element array to $newsize destination")
        end
        quote
            @_propagate_inbounds_meta
            $(exprs...)
            return v
        end
    else
        quote
            @_propagate_inbounds_meta
            if length(v) != $(prod(linearsizes))
                newsize = $linearsizes
                throw(DimensionMismatch("tried to assign $(length(v))-element array to $newsize destination"))
            end
            $(exprs...)
            return v
        end
    end
end
