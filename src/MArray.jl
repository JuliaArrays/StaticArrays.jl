"""
    MArray{S, T, N, L}(undef)
    MArray{S, T, N, L}(x::NTuple{L})
    MArray{S, T, N, L}(x1, x2, x3, ...)


Construct a statically-sized, mutable array `MArray`. The data may optionally be
provided upon construction and can be mutated later. The `S` parameter is a Tuple-type
specifying the dimensions, or size, of the array - such as `Tuple{3,4,5}` for a 3×4×5-sized
array. The `N` parameter is the dimension of the array; the `L` parameter is the `length`
of the array and is always equal to `prod(S)`. Constructors may drop the `L`, `N` and `T`
parameters if they are inferrable from the input (e.g. `L` is always inferrable from `S`).

    MArray{S}(a::Array)

Construct a statically-sized, mutable array of dimensions `S` (expressed as a `Tuple{...}`)
using the data from `a`. The `S` parameter is mandatory since the size of `a` is unknown to
the compiler (the element type may optionally also be specified).
"""
mutable struct MArray{S <: Tuple, T, N, L} <: StaticArray{S, T, N}
    data::NTuple{L,T}

    function MArray{S,T,N,L}(x::NTuple{L,T}) where {S<:Tuple,T,N,L}
        check_array_parameters(S, T, Val{N}, Val{L})
        new{S,T,N,L}(x)
    end

    function MArray{S,T,N,L}(x::NTuple{L,Any}) where {S<:Tuple,T,N,L}
        check_array_parameters(S, T, Val{N}, Val{L})
        new{S,T,N,L}(convert_ntuple(T, x))
    end

    function MArray{S,T,N,L}(::UndefInitializer) where {S<:Tuple,T,N,L}
        check_array_parameters(S, T, Val{N}, Val{L})
        new{S,T,N,L}()
    end
end

@inline MArray{S,T,N}(x::Tuple) where {S<:Tuple,T,N} = MArray{S,T,N,tuple_prod(S)}(x)

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

@inline Tuple(v::MArray) = getfield(v,:data)

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
    esc(static_array_gen(MArray, ex, __module__))
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
