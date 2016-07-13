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

# unfortunately index can't be used by a @generated function
@inline function insert(vec::StaticVector, index, x)
    similar_type(vec, (length(vec) + 1 ,))((Tuple(vec[1:index-1])..., x, Tuple(vec[index:end])...))
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

# unfortunately index can't be used by a @generated function
@inline function deleteat(vec::StaticVector, index)
    similar_type(vec, (length(vec) - 1 ,))((Tuple(vec[1:index-1])..., Tuple(vec[index+1:end])...))
end

# TODO consider prepend, append (can use vcat), and maybe splice (a bit hard to get statically sized)
