
@generated function (::Type{MArray{S,T,N}})(::UndefInitializer) where {S,T,N}
    return quote
        $(Expr(:meta, :inline))
        MArray{S, T, N, $(tuple_prod(S))}(undef)
    end
end

@generated function (::Type{MArray{S,T}})(::UndefInitializer) where {S,T}
    return quote
        $(Expr(:meta, :inline))
        MArray{S, T, $(tuple_length(S)), $(tuple_prod(S))}(undef)
    end
end

####################
## MArray methods ##
####################

@propagate_inbounds function getindex(v::MArray, i::Int)
    @boundscheck checkbounds(v,i)
    T = eltype(v)

    if isbitstype(T)
        return GC.@preserve v unsafe_load(Base.unsafe_convert(Ptr{T}, pointer_from_objref(v)), i)
    end
    getfield(v,:data)[i]
end

@propagate_inbounds function setindex!(v::MArray, val, i::Int)
    @boundscheck checkbounds(v,i)
    T = eltype(v)

    if isbitstype(T)
        GC.@preserve v unsafe_store!(Base.unsafe_convert(Ptr{T}, pointer_from_objref(v)), convert(T, val), i)
    else
        # This one is unsafe (#27)
        # unsafe_store!(Base.unsafe_convert(Ptr{Ptr{Nothing}}, pointer_from_objref(v.data)), pointer_from_objref(val), i)
        error("setindex!() with non-isbitstype eltype is not supported by StaticArrays. Consider using SizedArray.")
    end

    return v
end

@inline Base.Tuple(v::MArray) = getfield(v,:data)

Base.dataids(ma::MArray) = (UInt(pointer(ma)),)

@inline function Base.unsafe_convert(::Type{Ptr{T}}, a::MArray{S,T}) where {S,T}
    Base.unsafe_convert(Ptr{T}, pointer_from_objref(a))
end

"""
    @MArray [a b; c d]
    @MArray [[a, b];[c, d]]
    @MArray [i+j for i in 1:2, j in 1:2]
    @MArray ones(2, 2, 2)

A convenience macro to construct `MArray` with arbitrary dimension.
See [`@SArray`](@ref) for detailed features.
"""
macro MArray(ex)
    static_array_gen(MArray, ex, __module__)
end

function promote_rule(::Type{<:MArray{S,T,N,L}}, ::Type{<:MArray{S,U,N,L}}) where {S,T,U,N,L}
    MArray{S,promote_type(T,U),N,L}
end

@generated function _indices_have_bools(indices::Tuple)
    return any(index -> index <: StaticVector{<:Any,Bool}, indices.parameters)
end

function Base.view(
    a::MArray{S},
    indices::Union{Integer, Colon, StaticVector, Base.Slice, SOneTo}...,
) where {S}
    view_from_invoke = invoke(view, Tuple{AbstractArray, typeof(indices).parameters...}, a, indices...)
    if _indices_have_bools(indices)
        return view_from_invoke
    else
        new_size = new_out_size(S, indices...)
        return SizedArray{new_size}(view_from_invoke)
    end
end

Base.elsize(::Type{<:MArray{<:Any, T}}) where T = Base.elsize(Vector{T})
