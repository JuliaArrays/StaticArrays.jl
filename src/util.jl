# For convenience
typealias TupleN{T,N} NTuple{N,T}

# Cast any Tuple to an TupleN{T}
convert_ntuple{T}(::Type{T},d::T) = T # For zero-dimensional arrays
convert_ntuple{N,T}(::Type{T},d::NTuple{N,T}) = d
@generated function convert_ntuple{N,T}(::Type{T},d::NTuple{N})
    exprs = ntuple(i -> :(convert(T, d[$i])), Val{N})
    return Expr(:tuple, exprs...)
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

# some convenience functions for non-static arrays, generators, etc...
@inline convert{T}(::Type{Tuple}, a::AbstractArray{T}) = (a...)::Tuple{Vararg{T}}
@inline function convert{N,T}(::Type{NTuple{N}}, a::AbstractArray{T})
    @boundscheck if length(a) != N
        error("Array of length $(length(a)) cannot be converted to a $N-tuple")
    end

    @inbounds return ntuple(i -> a[i], Val{N})
end

@inline function convert{N,T1,T2}(::Type{NTuple{N,T1}}, a::AbstractArray{T2})
    @boundscheck if length(a) != N
        error("Array of length $(length(a)) cannot be converted to a $N-tuple")
    end

    @inbounds return ntuple(i -> convert(T1,a[i]), Val{N})
end

# TODO try and make this generate fast code
@inline convert(::Type{Tuple}, g::Base.Generator) = (g...)
@inline function convert{N}(::Type{NTuple{N}}, g::Base.Generator)
    @boundscheck if length(g.iter) != N
        error("Array of length $(length(a)) cannot be converted to a $N-tuple")
    end

    @inbounds return ntuple(i -> g.f(g.iter[i]), Val{N})
end

#=
@generated function convert{N}(::Type{NTuple{N}}, g::Base.Generator)
    exprs = [:(g.f(g.iter[$j])) for j=1:N]
    return quote
        @boundscheck if length(g.iter) != N
            error("Array of length $(length(a)) cannot be converted to a $N-tuple")
        end

        @inbounds return $(Expr(:tuple, exprs...))
    end
end=#
