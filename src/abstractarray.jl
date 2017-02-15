typealias StaticScalar{T} StaticArray{T,0}

length{T<:StaticArray}(a::T) = prod(Size(T))
length{T<:StaticArray}(a::Type{T}) = prod(Size(T))

size{T<:StaticArray}(::T) = get(Size(T))
size{T<:StaticArray}(::Type{T}) = get(Size(T))

size{T<:StaticArray}(::T, d::Integer) = Size(T)[d]
size{T<:StaticArray}(::Type{T}, d::Integer) = Size(T)[d]

# This has to be defined after length and size because it is generated
@generated function convert{SA<:StaticArray}(::Type{SA}, a::AbstractArray)
    L = length(SA)
    exprs = [:(a[$i]) for i = 1:L]

    return quote
        $(Expr(:meta, :inline))
        if length(a) != $L
            L = $L
            error("Dimension mismatch. Expected input array of length $L, got length $(length(a))")
        end
        @inbounds return SA($(Expr(:tuple, exprs...)))
    end
end

# This seems to confuse Julia a bit in certain circumstances (specifically for trailing 1's)
@inline function Base.isassigned(a::StaticArray, i::Int...)
    ii = sub2ind(size(a), i...)
    1 <= ii <= length(a) ? true : false
end

Base.linearindexing(::StaticArray) = Base.LinearFast()
Base.linearindexing{T<:StaticArray}(::Type{T}) = Base.LinearFast()

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

similar_type{SA<:StaticArray}(::SA) = similar_type(SA,eltype(SA))
similar_type{SA<:StaticArray}(::Type{SA}) = similar_type(SA,eltype(SA))

similar_type{SA<:StaticArray,T}(::SA,::Type{T}) = similar_type(SA,T,Size(SA))
similar_type{SA<:StaticArray,T}(::Type{SA},::Type{T}) = similar_type(SA,T,Size(SA))

similar_type{A<:AbstractArray,S}(::A,s::Size{S}) = similar_type(A,eltype(A),s)
similar_type{A<:AbstractArray,S}(::Type{A},s::Size{S}) = similar_type(A,eltype(A),s)

similar_type{A<:AbstractArray,T,S}(::A,::Type{T},s::Size{S}) = similar_type(A,T,s)

# Default types
# Generally, use SVector, etc
similar_type{A<:AbstractArray,T,S}(::Type{A},::Type{T},s::Size{S}) = default_similar_type(T,s,length_val(s))

default_similar_type{T,S}(::Type{T}, s::Size{S}, ::Type{Val{0}}) = Scalar{T}
@generated default_similar_type{T,S}(::Type{T}, s::Size{S}, ::Type{Val{1}}) = SVector{S[1],T}
@generated default_similar_type{T,S}(::Type{T}, s::Size{S}, ::Type{Val{2}}) = SMatrix{S[1],S[2],T,prod(S)}
@generated default_similar_type{T,S,D}(::Type{T}, s::Size{S}, ::Type{Val{D}}) = SArray{S,T,D,prod(S)}

# mutable things stay mutable
similar_type{SA<:Union{MVector,MMatrix,MArray},T,S}(::Type{SA},::Type{T},s::Size{S}) = mutable_similar_type(T,s,length_val(s))

mutable_similar_type{T,S}(::Type{T}, s::Size{S}, ::Type{Val{0}}) = SizedArray{(),T,0,0}
@generated mutable_similar_type{T,S}(::Type{T}, s::Size{S}, ::Type{Val{1}}) = MVector{S[1],T}
@generated mutable_similar_type{T,S}(::Type{T}, s::Size{S}, ::Type{Val{2}}) = MMatrix{S[1],S[2],T,prod(S)}
@generated mutable_similar_type{T,S,D}(::Type{T}, s::Size{S}, ::Type{Val{D}}) = MArray{S,T,D,prod(S)}

# `SizedArray` stays the same, and also takes over an `Array`.
similar_type{SA<:SizedArray,T,S}(::Type{SA},::Type{T},s::Size{S}) = sizedarray_similar_type(T,s,length_val(s))
similar_type{A<:Array,T,S}(::Type{A},::Type{T},s::Size{S}) = sizedarray_similar_type(T,s,length_val(s))

sizedarray_similar_type{T,S,D}(::Type{T},s::Size{S},::Type{Val{D}}) = SizedArray{S,T,D,D}

# Field vectors are user controlled, and default to SVector, etc

"""
    similar(static_array)
    similar(static_array, T)
    similar(array, ::Size)
    similar(array, T, ::Size)

Constructs and returns a mutable but statically-sized array (i.e. a `StaticArray`). If the
input `array` is not a `StaticArray`, then the `Size` is required to determine the output
size (or else a dynamically sized array will be returned).
"""
similar{SA<:StaticArray}(::SA) = similar(SA,eltype(SA))
similar{SA<:StaticArray}(::Type{SA}) = similar(SA,eltype(SA))

similar{SA<:StaticArray,T}(::SA,::Type{T}) = similar(SA,T,Size(SA))
similar{SA<:StaticArray,T}(::Type{SA},::Type{T}) = similar(SA,T,Size(SA))

similar{A<:AbstractArray,S}(::A,s::Size{S}) = similar(A,eltype(A),s)
similar{A<:AbstractArray,S}(::Type{A},s::Size{S}) = similar(A,eltype(A),s)

similar{A<:AbstractArray,T,S}(::A,::Type{T},s::Size{S}) = similar(A,T,s)

# defaults to built-in mutable types
similar{A<:AbstractArray,T,S}(::Type{A},::Type{T},s::Size{S}) = mutable_similar_type(T,s,length_val(s))()

# both SizedArray and Array return SizedArray
similar{SA<:SizedArray,T,S}(::Type{SA},::Type{T},s::Size{S}) = sizedarray_similar_type(T,s,length_val(s))()
similar{A<:Array,T,S}(::Type{A},::Type{T},s::Size{S}) = sizedarray_similar_type(T,s,length_val(s))()


# This is used in Base.LinAlg quite a lot, and it impacts type stability
# since some functions like expm() branch on a check for Hermitian or Symmetric
# TODO much more work on type stability. Base functions are using similar() with
# size, which poses difficulties!!
@generated function Base.full{T,SM<:StaticMatrix}(sym::Symmetric{T,SM})
    exprs_up = [i <= j ? :(m[$(sub2ind(size(SM), i, j))]) : :(m[$(sub2ind(size(SM), j, i))]) for i = 1:size(SM,1), j=1:size(SM,2)]
    exprs_lo = [i >= j ? :(m[$(sub2ind(size(SM), i, j))]) : :(m[$(sub2ind(size(SM), j, i))]) for i = 1:size(SM,1), j=1:size(SM,2)]

    return quote
        $(Expr(:meta, :inline))
        m = sym.data
        if sym.uplo == 'U'
            @inbounds return SM($(Expr(:call, SM, Expr(:tuple, exprs_up...))))
        else
            @inbounds return SM($(Expr(:call, SM, Expr(:tuple, exprs_lo...))))
        end
    end
end

@generated function Base.full{T,SM<:StaticMatrix}(sym::Hermitian{T,SM})
    exprs_up = [i <= j ? :(m[$(sub2ind(size(SM), i, j))]) : :(conj(m[$(sub2ind(size(SM), j, i))])) for i = 1:size(SM,1), j=1:size(SM,2)]
    exprs_lo = [i >= j ? :(m[$(sub2ind(size(SM), i, j))]) : :(conj(m[$(sub2ind(size(SM), j, i))])) for i = 1:size(SM,1), j=1:size(SM,2)]

    return quote
        $(Expr(:meta, :inline))
        m = sym.data
        if sym.uplo == 'U'
            @inbounds return SM($(Expr(:call, SM, Expr(:tuple, exprs_up...))))
        else
            @inbounds return SM($(Expr(:call, SM, Expr(:tuple, exprs_lo...))))
        end
    end
end

# Reshape used types to specify size (and also conveniently, the output type)
@generated function reshape{SA<:StaticArray}(a::StaticArray, ::Type{SA})
    if !(SA <: StaticVector) && length(a) != length(SA)
        error("Static array of size $(size(a)) cannot be reshaped to size $(size(SA))")
    end

    Base.depwarn("Use reshape(array, Size(dims...)) rather than reshape(array, StaticArrayType)", :reshape)

    return quote
        $(Expr(:meta, :inline))
        return SA(Tuple(a))
    end
end

function reshape{SA<:StaticArray}(a::AbstractArray, ::Type{SA})
    if !(SA <: StaticVector) && length(a) != length(SA)
        error("Static array of size $(size(a)) cannot be reshaped to size $(size(SA))")
    end

    Base.depwarn("Use reshape(array, Size(dims...)) rather than reshape(array, StaticArrayType)", :reshape)

    return SA((a...))
end


# Versions using Size{}
@generated function reshape{S}(a::StaticArray, ::Size{S})
    if length(a) != prod(S)
        error("Static array of size $(size(a)) cannot be reshaped to size $S")
    end

    newtype = similar_type(a, Size(S))

    return quote
        $(Expr(:meta, :inline))
        return $newtype(a)
    end
end

@generated function reshape{S}(a::Array, ::Size{S})
    newtype = SizedArray{S, eltype(a), length(S)}

    return quote
        $(Expr(:meta, :inline))
        return $newtype(a)
    end
end

# Clever ruse to determine if a type is "mutable"
# Definitely *not* a deep copy.
@generated function copy(a::StaticArray)
    try
        out = a()
        return quote
            $(Expr(:meta, :inline))
            out = $(a)()
            out .= a
            return out
        end
    catch
        return quote
            $(Expr(:meta, :inline))
            a
        end
    end
end
