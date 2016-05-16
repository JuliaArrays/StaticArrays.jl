# For convenience
typealias TupleN{T,N} NTuple{N,T}

# Cast any Tuple to an TupleN{T}
convert_ntuple{T}(::Type{T},d::T) = T # For zero-dimensional arrays
convert_ntuple{N,T}(::Type{T},d::NTuple{N,T}) = d
@generated function convert_ntuple{N,T}(::Type{T},d::NTuple{N})
    exprs = ntuple(i -> :(convert(T, d[$i])), Val{N})
    return Expr(:tuple, exprs...)
end

# Base gives up on tuples... (TODO can we improve Base?)
@generated function promote_tuple_eltype{T <: Tuple}(::Union{T,Type{T}})
    t = Union{}
    for i = 1:length(T.parameters)
        t = promote_type(t, T.parameters[i])
    end
    return t
end
