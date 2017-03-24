# For convenience
TupleN{T,N} = NTuple{N,T}

# Cast any Tuple to an TupleN{T}
@inline convert_ntuple{T}(::Type{T},d::T) = T # For zero-dimensional arrays
@inline convert_ntuple{N,T}(::Type{T},d::NTuple{N,T}) = d
@generated function convert_ntuple{N,T}(::Type{T}, d::NTuple{N,Any})
    exprs = ntuple(i -> :(convert(T, d[$i])), Val{N})
    return quote
        $(Expr(:meta, :inline))
        $(Expr(:tuple, exprs...))
    end
end

# Base gives up on tuples for promote_eltype... (TODO can we improve Base?)
@generated function promote_tuple_eltype{T <: Tuple}(::Union{T,Type{T}})
    t = Union{}
    for i = 1:length(T.parameters)
        tmp = T.parameters[i]
        if tmp <: Vararg
            tmp = tmp.parameters[1]
        end
        t = promote_type(t, tmp)
    end
    return quote
        $(Expr(:meta,:pure))
        $t
    end
end
