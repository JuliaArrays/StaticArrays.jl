export push, pop, shift, unshift, insert, deleteat

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

# unfortunately index can't be used directly by a @generated function... this code is pretty slow
@generated function insert(vec::StaticVector, index, x)
    newtype = similar_type(vec, (length(vec) + 1 ,))
    expr = :((Tuple(vec[1:index-1])..., x, Tuple(vec[index:end])...))
    return quote
        $(Expr(:meta, :inline))
        @inbounds return $(Expr(:call, newtype, expr))
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

# unfortunately index can't be used directly by a @generated function... this code is pretty slow
@generated function deleteat(vec::StaticVector, index)
    newtype = similar_type(vec, (length(vec) - 1 ,))
    expr = :((Tuple(vec[1:index-1])..., Tuple(vec[index+1:end])...))
    return quote
        $(Expr(:meta, :inline))
        @inbounds return $(Expr(:call, newtype, expr))
    end
end

# TODO consider prepend, append (can use vcat, but eltype might change), and
# maybe splice (a bit hard to get statically sized)


# Immutable version of setindex!(). Do these belong here?
# TODO make this faster... currently generated code is bad because size of the
# various things is fast. One idea is to populate all the data as variables,
# overwrite one, and put them back into a static array. Same for deleteat() and insert()
@generated function setindex{T}(a::StaticArray{T}, x::T, index)
    newtype = a
    expr = :((Tuple(a[1:index-1])..., x, Tuple(a[index+1:end])...))
    return quote
        $(Expr(:meta, :inline))
        @inbounds return $(Expr(:call, newtype, expr))
    end
end


@generated function setindex(a::StaticArray, x, index)
    newtype = a
    expr = :((Tuple(a[1:index-1])..., convert($(eltype(a)), x), Tuple(a[index+1:end])...))
    return quote
        $(Expr(:meta, :inline))
        @inbounds return $(Expr(:call, newtype, expr))
    end
end

@inline setindex(a::StaticArray, x, inds...) = setindex(a, x, sub2ind(size(a), inds...))
