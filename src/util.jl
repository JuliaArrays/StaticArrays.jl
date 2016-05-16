# For convenience
typealias TupleN{T,N} NTuple{N,T}


convert_ntuple{T}(::Type{T},d::T) = T # For zero-dimensional arrays
convert_ntuple{N,T}(::Type{T},d::NTuple{N,T}) = d
@generated function convert_ntuple{N,T}(::Type{T},d::NTuple{N})
    exprs = ntuple(i -> :(convert(T, d[$i])), Val{N})
    return Expr(:tuple, exprs...)
end

promote_eltype(a) = eltype(a) # For uniform data, e.g. Array.
@generated function promote_eltype{T <: Tuple}(::Union{T,Type{T}}) # Tuples are non-uniform data
    t = T.parameters[1]
    for i = 2:length(T.parameters)
        t = promote_type(t, T.parameters[i])
    end
    return t
end
