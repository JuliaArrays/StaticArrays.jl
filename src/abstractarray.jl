length(a::SA) where {SA <: StaticArrayLike} = length(SA)
length(a::Type{SA}) where {SA <: StaticArrayLike} = prod(Size(SA))

@pure size(::Type{SA}) where {SA <: StaticArrayLike} = get(Size(SA))
@inline function size(t::Type{<:StaticArrayLike}, d::Int)
    S = size(t)
    d > length(S) ? 1 : S[d]
end
@inline size(a::StaticArrayLike) = size(typeof(a))
@inline size(a::StaticArrayLike, d::Int) = size(typeof(a), d)

Base.axes(s::StaticArray) = _axes(Size(s))
@pure function _axes(::Size{sizes}) where {sizes}
    map(SOneTo, sizes)
end
Base.axes(rv::Adjoint{<:Any,<:StaticVector})   = (SOneTo(1), axes(rv.parent)...)
Base.axes(rv::Transpose{<:Any,<:StaticVector}) = (SOneTo(1), axes(rv.parent)...)

function Base.summary(io::IO, a, inds::Tuple{SOneTo, Vararg{SOneTo}})
    print(io, Base.dims2string(length.(inds)), " ")
    Base.showarg(io, a, true)
end

# This seems to confuse Julia a bit in certain circumstances (specifically for trailing 1's)
@inline function Base.isassigned(a::StaticArray, i::Int...)
    ii = LinearIndices(size(a))[i...]
    1 <= ii <= length(a) ? true : false
end

Base.IndexStyle(::Type{T}) where {T<:StaticArray} = IndexLinear()

# Default type search for similar_type
"""
    similar_type(static_array)
    similar_type(static_array, T)
    similar_type(array, ::Size)
    similar_type(array, T, ::Size)

Returns a constructor for a statically-sized array similar to the input array
(or type) `static_array`/`array`, optionally with different element type `T` or size
`Size`. If the input `array` is not a `StaticArray` then the `Size` is mandatory.

This differs from `similar()` in that the resulting array type may not be
mutable (or define `setindex!()`), and therefore the returned type may need to
be *constructed* with its data.

Note that the (optional) size *must* be specified as a static `Size` object (so the compiler
can infer the result statically).

New types should define the signature `similar_type(::Type{A},::Type{T},::Size{S}) where {A<:MyType,T,S}`
if they wish to overload the default behavior.
"""
function similar_type end

similar_type(::SA) where {SA<:StaticArray} = similar_type(SA,eltype(SA))
similar_type(::Type{SA}) where {SA<:StaticArray} = similar_type(SA,eltype(SA))

similar_type(::SA,::Type{T}) where {SA<:StaticArray,T} = similar_type(SA,T,Size(SA))
similar_type(::Type{SA},::Type{T}) where {SA<:StaticArray,T} = similar_type(SA,T,Size(SA))

similar_type(::A,s::Size{S}) where {A<:AbstractArray,S} = similar_type(A,eltype(A),s)
similar_type(::Type{A},s::Size{S}) where {A<:AbstractArray,S} = similar_type(A,eltype(A),s)

similar_type(::A,::Type{T},s::Size{S}) where {A<:AbstractArray,T,S} = similar_type(A,T,s)

# We should be able to deal with SOneTo axes
similar_type(s::SOneTo) = similar_type(typeof(s))
similar_type(::Type{SOneTo{n}}) where {n} = similar_type(SOneTo{n}, Int, Size(n))

similar_type(::A, shape::Tuple{SOneTo, Vararg{SOneTo}}) where {A<:AbstractArray} = similar_type(A, eltype(A), shape)
similar_type(::Type{A}, shape::Tuple{SOneTo, Vararg{SOneTo}}) where {A<:AbstractArray} = similar_type(A, eltype(A), shape)

similar_type(::A,::Type{T}, shape::Tuple{SOneTo, Vararg{SOneTo}}) where {A<:AbstractArray,T} = similar_type(A, T, Size(last.(shape)))
similar_type(::Type{A},::Type{T}, shape::Tuple{SOneTo, Vararg{SOneTo}}) where {A<:AbstractArray,T} = similar_type(A, T, Size(last.(shape)))


# Default types
# Generally, use SArray
similar_type(::Type{A},::Type{T},s::Size{S}) where {A<:AbstractArray,T,S} = default_similar_type(T,s,length_val(s))
default_similar_type(::Type{T}, s::Size{S}, ::Type{Val{D}}) where {T,S,D} = SArray{Tuple{S...},T,D,prod(s)}

similar_type(::Type{SA},::Type{T},s::Size{S}) where {SA<:Union{MVector,MMatrix,MArray},T,S} = mutable_similar_type(T,s,length_val(s))

mutable_similar_type(::Type{T}, s::Size{S}, ::Type{Val{D}}) where {T,S,D} = MArray{Tuple{S...},T,D,prod(s)}

# Should `SizedArray` stay the same, and also take over an `Array`?
#similar_type{SA<:SizedArray,T,S}(::Type{SA},::Type{T},s::Size{S}) = sizedarray_similar_type(T,s,length_val(s))
#similar_type{A<:Array,T,S}(::Type{A},::Type{T},s::Size{S}) = sizedarray_similar_type(T,s,length_val(s))

sizedarray_similar_type(::Type{T},s::Size{S},::Type{Val{D}}) where {T,S,D} = SizedArray{Tuple{S...},T,D,length(s)}

# Field vectors are user controlled, and currently default to SVector, etc

"""
    similar(static_array)
    similar(static_array, T)
    similar(array, ::Size)
    similar(array, T, ::Size)

Constructs and returns a mutable but statically-sized array (i.e. a `StaticArray`). If the
input `array` is not a `StaticArray`, then the `Size` is required to determine the output
size (or else a dynamically sized array will be returned).
"""
similar(::SA) where {SA<:StaticArray} = similar(SA,eltype(SA))
similar(::Type{SA}) where {SA<:StaticArray} = similar(SA,eltype(SA))

similar(::SA,::Type{T}) where {SA<:StaticArray,T} = similar(SA,T,Size(SA))
similar(::Type{SA},::Type{T}) where {SA<:StaticArray,T} = similar(SA,T,Size(SA))

# Cases where a Size is given as the dimensions
similar(::A,s::Size{S}) where {A<:AbstractArray,S} = similar(A,eltype(A),s)
similar(::Type{A},s::Size{S}) where {A<:AbstractArray,S} = similar(A,eltype(A),s)

similar(::A,::Type{T},s::Size{S}) where {A<:AbstractArray,T,S} = similar(A,T,s)

# defaults to built-in mutable types
similar(::Type{A},::Type{T},s::Size{S}) where {A<:AbstractArray,T,S} = mutable_similar_type(T,s,length_val(s))(undef)

# both SizedArray and Array return SizedArray
similar(::Type{SA},::Type{T},s::Size{S}) where {SA<:SizedArray,T,S} = sizedarray_similar_type(T,s,length_val(s))(undef)
similar(::Type{A},::Type{T},s::Size{S}) where {A<:Array,T,S} = sizedarray_similar_type(T,s,length_val(s))(undef)

# Support tuples of mixtures of `SOneTo`s alongside the normal `Integer` and `OneTo` options
# by simply converting them to either a tuple of Ints or a Size, re-dispatching to either one
# of the above methods (in the case of Size) or a base fallback (in the case of Ints).
const HeterogeneousShape = Union{Integer, Base.OneTo, SOneTo}

similar(A::AbstractArray, ::Type{T}, shape::Tuple{HeterogeneousShape, Vararg{HeterogeneousShape}}) where {T} = similar(A, T, homogenize_shape(shape))
similar(::Type{A}, shape::Tuple{HeterogeneousShape, Vararg{HeterogeneousShape}}) where {A<:AbstractArray} = similar(A, homogenize_shape(shape))
# Use an Array for StaticArrays if we don't have a statically-known size
similar(::Type{A}, shape::Tuple{Int, Vararg{Int}}) where {A<:StaticArray} = Array{eltype(A)}(undef, shape)

homogenize_shape(::Tuple{}) = ()
homogenize_shape(shape::Tuple{Vararg{SOneTo}}) = Size(map(last, shape))
homogenize_shape(shape::Tuple{Vararg{HeterogeneousShape}}) = map(last, shape)


@inline reshape(a::StaticArray, s::Size) = similar_type(a, s)(Tuple(a))
@inline reshape(a::AbstractArray, s::Size) = _reshape(a, IndexStyle(a), s)
@inline reshape(a::StaticArray, s::Tuple{SOneTo,Vararg{SOneTo}}) = reshape(a, homogenize_shape(s))
@generated function _reshape(a::AbstractArray, indexstyle, s::Size{S}) where {S}
    if indexstyle == IndexLinear
        exprs = [:(a[$i]) for i = 1:prod(S)]
    else
        exprs = [:(a[$(inds)]) for inds ∈ CartesianIndices(S)]
    end

    return quote
        @_inline_meta
        if length(a) != prod(s)
            throw(DimensionMismatch("Tried to resize dynamic object of size $(size(a)) to $s"))
        end
        return similar_type(a, s)(tuple($(exprs...)))
    end
end

reshape(a::Array, s::Size{S}) where {S} = s(a)

@inline vec(a::StaticArray) = reshape(a, Size(prod(Size(typeof(a)))))

@inline copy(a::StaticArray) = typeof(a)(Tuple(a))
@inline copy(a::SizedArray) = typeof(a)(copy(a.data))

@inline reverse(v::StaticVector) = typeof(v)(reverse(Tuple(v)))

# TODO permutedims?

# full deprecated in Base
if isdefined(Base, :full)
    import Base: full
    @deprecate full(sym::Symmetric{T,SM}) where {T,SM <: StaticMatrix} SMatrix(sym)
    @deprecate full(herm::Hermitian{T,SM}) where {T,SM <: StaticMatrix} SMatrix(sym)
end
