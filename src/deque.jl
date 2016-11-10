export push, pop, shift, unshift, insert, deleteat, setindex

@generated function push(vec::StaticVector, x)
    newtype = similar_type(vec, (length(vec) + 1 ,))
    exprs = vcat([:(vec[$i]) for i = 1:length(vec)], :x)
    return quote
        $(Expr(:meta, :inline))
        @inbounds return $(Expr(:call, newtype, Expr(:tuple, exprs...)))
    end
end

@generated function unshift(vec::StaticVector, x)
    newtype = similar_type(vec, (length(vec) + 1 ,))
    exprs = vcat(:x, [:(vec[$i]) for i = 1:length(vec)])
    return quote
        $(Expr(:meta, :inline))
        @inbounds return $(Expr(:call, newtype, Expr(:tuple, exprs...)))
    end
end

@generated function insert(vec::StaticVector, index, x)
    newtype = similar_type(vec, (length(vec) + 1 ,))
    exprs = [(i == 1 ? :(ifelse($i < index, vec[$i], x)) :
              i == length(vec)+1 ? :(ifelse($i == index, x, vec[$i-1])) :
              :(ifelse($i < index, vec[$i], ifelse($i == index, x, vec[$i-1])))) for i = 1:length(vec) + 1]
    return quote
        $(Expr(:meta, :inline))
        @boundscheck if (index < 1 || index > $(length(vec)+1))
            throw(BoundsError(vec, index))
        end
        @inbounds return $(Expr(:call, newtype, Expr(:tuple, exprs...)))
    end
end

@generated function pop(vec::StaticVector)
    newtype = similar_type(vec, (length(vec) - 1 ,))
    exprs = [:(vec[$i]) for i = 1:length(vec)-1]
    return quote
        $(Expr(:meta, :inline))
        @inbounds return $(Expr(:call, newtype, Expr(:tuple, exprs...)))
    end
end

@generated function shift(vec::StaticVector)
    newtype = similar_type(vec, (length(vec) - 1 ,))
    exprs = [:(vec[$i]) for i = 2:length(vec)]
    return quote
        $(Expr(:meta, :inline))
        @inbounds return $(Expr(:call, newtype, Expr(:tuple, exprs...)))
    end
end

@generated function deleteat(vec::StaticVector, index)
    newtype = similar_type(vec, (length(vec) - 1 ,))
    exprs = [:(ifelse($i < index, vec[$i], vec[$i+1])) for i = 1:length(vec) - 1]
    return quote
        $(Expr(:meta, :inline))
        @boundscheck if (index < 1 || index > $(length(vec)+1))
            throw(BoundsError(vec, index))
        end
        @inbounds return $(Expr(:call, newtype, Expr(:tuple, exprs...)))
    end
end

# TODO consider prepend, append (can use vcat, but eltype might change), and
# maybe splice (a bit hard to get statically sized without a "static" range)


# Immutable version of setindex!(). Seems similar in nature to the above, but
# could also be justified to live in src/indexing.jl
@generated function setindex{T}(a::StaticArray{T}, x::T, index::Int)
    newtype = a
    exprs = [:(ifelse($i == index, x, a[$i])) for i = 1:length(a)]
    return quote
        $(Expr(:meta, :inline))
        @boundscheck if (index < 1 || index > $(length(a)))
            throw(BoundsError(a, index))
        end
        @inbounds return $(Expr(:call, newtype, Expr(:tuple, exprs...)))
    end
end

@propagate_inbounds setindex(a::StaticArray, x, index::Int) = setindex(a, convert(eltype(typeof(a)), x), index)

# TODO proper multidimension boundscheck
@propagate_inbounds setindex(a::StaticArray, x, inds::Int...) = setindex(a, x, sub2ind(size(typeof(a)), inds...))
