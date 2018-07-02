length(a::SA) where {SA <: StaticArray} = prod(Size(SA))
length(a::Type{SA}) where {SA <: StaticArray} = prod(Size(SA))

@pure size(::Type{<:StaticArray{S}}) where S = tuple(S.parameters...)
@inline function size(t::Type{<:StaticArray}, d::Int)
    S = size(t)
    d > length(S) ? 1 : S[d]
end
@inline size(a::StaticArray) = size(typeof(a))
@inline size(a::StaticArray, d::Int) = size(typeof(a), d)

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

New types should define the signature `similar_type{A<:MyType,T,S}(::Type{A},::Type{T},::Size{S})`
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

# Default types
# Generally, use SArray
similar_type(::Type{A},::Type{T},s::Size{S}) where {A<:AbstractArray,T,S} = default_similar_type(T,s,length_val(s))
default_similar_type(::Type{T}, s::Size{S}, ::Type{Val{D}}) where {T,S,D} = SArray{Tuple{S...},T,D,prod(s)}


# should mutable things stay mutable?
#similar_type{SA<:Union{MVector,MMatrix,MArray},T,S}(::Type{SA},::Type{T},s::Size{S}) = mutable_similar_type(T,s,length_val(s))

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

similar(::A,s::Size{S}) where {A<:AbstractArray,S} = similar(A,eltype(A),s)
similar(::Type{A},s::Size{S}) where {A<:AbstractArray,S} = similar(A,eltype(A),s)

similar(::A,::Type{T},s::Size{S}) where {A<:AbstractArray,T,S} = similar(A,T,s)

# defaults to built-in mutable types
similar(::Type{A},::Type{T},s::Size{S}) where {A<:AbstractArray,T,S} = mutable_similar_type(T,s,length_val(s))(undef)

# both SizedArray and Array return SizedArray
similar(::Type{SA},::Type{T},s::Size{S}) where {SA<:SizedArray,T,S} = sizedarray_similar_type(T,s,length_val(s))()
similar(::Type{A},::Type{T},s::Size{S}) where {A<:Array,T,S} = sizedarray_similar_type(T,s,length_val(s))()


@inline reshape(a::StaticArray, s::Size) = similar_type(a, s)(Tuple(a))
@inline reshape(a::AbstractArray, s::Size) = _reshape(a, IndexStyle(a), s)
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

# TODO permutedims?

# full deprecated in Base
if isdefined(Base, :full)
    import Base: full
    @deprecate full(sym::Symmetric{T,SM}) where {T,SM <: StaticMatrix} SMatrix(sym)
    @deprecate full(herm::Hermitian{T,SM}) where {T,SM <: StaticMatrix} SMatrix(sym)
end
