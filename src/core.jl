"""
    abstract StaticArray{T, N} <: DenseArray{T, N}

`StaticArray`s are Julia arrays with fixed, known size.

## Dev docs

They must define the following methods.
 - `size()` on the *type*, as well as the instances, returning a tuple of `Int`s
 - `getindex()` with an integer (linear indexing)
 - `Tuple()`, returning the data in a flat Tuple.

It is strongly recommended to implement

- `similar_type()` returns a type (or type constructor) that accepts a flat
  tuple of data. Otherwise, the built-in `SArray`, `SVector` and `SMatrix` types will
  be used.

Otherwise, it may also be useful to define some of the following:

 - `similar()` returns a *mutable* container of similar type, defaults to using
   a `Ref{}`. You will benefit from overloading this if your container is
   already mutable!
 - `getindex` on Ref{MyStaticArray} or whatever type is returned by `similar` (\*)
 - `setindex!` on Ref{MyStaticArray} or whatever type is returned by `similar` (\*)

(\*) It is assumed by default that your memory is laid out in a dense format,
like `Array`. If your `StaticArray` subtype contains multiple fields, make sure
the data appears first, or if not define `getindex`/`setindex` yourself.
"""
abstract StaticArray{T, N} <: DenseArray{T, N}

typealias StaticVector{T} StaticArray{T, 1}
typealias StaticMatrix{T} StaticArray{T, 2}

# People might not want to use Tuple for everything (TODO: check this with FieldVector...)
# Generic case, with least 2 inputs
@inline (::Type{SA}){SA<:StaticArray}(x1,x2,xs...) = SA((x1,x2,xs...))

@inline (::Type{SA}){SA<:StaticArray}(x::Tuple) = error("No precise constructor found. Length of input was $(length(x)) while length of $SA is $(length(SA)).")

# Avoiding splatting penalties. Being here, implementations of StaticArray will not have to deal with these. TODO check these are necessary or not
#@inline (::Type{SA}){SA<:StaticArray}(x1) = SA((x1,)) # see convert below (lesser precedence than other constructors?)
@inline (::Type{SA}){SA<:StaticArray}(x1,x2) = SA((x1,x2))
@inline (::Type{SA}){SA<:StaticArray}(x1,x2,x3) = SA((x1,x2,x3))
@inline (::Type{SA}){SA<:StaticArray}(x1,x2,x3,x4) = SA((x1,x2,x3,x4))
@inline (::Type{SA}){SA<:StaticArray}(x1,x2,x3,x4,x5) = SA((x1,x2,x3,x4,x5))
@inline (::Type{SA}){SA<:StaticArray}(x1,x2,x3,x4,x5,x6) = SA((x1,x2,x3,x4,x5,x6))
@inline (::Type{SA}){SA<:StaticArray}(x1,x2,x3,x4,x5,x6,x7) = SA((x1,x2,x3,x4,x5,x6,x7))
@inline (::Type{SA}){SA<:StaticArray}(x1,x2,x3,x4,x5,x6,x7,x8) = SA((x1,x2,x3,x4,x5,x6,x7,x8))
@inline (::Type{SA}){SA<:StaticArray}(x1,x2,x3,x4,x5,x6,x7,x8,x9) = SA((x1,x2,x3,x4,x5,x6,x7,x8,x9))
@inline (::Type{SA}){SA<:StaticArray}(x1,x2,x3,x4,x5,x6,x7,x8,x9,x10) = SA((x1,x2,x3,x4,x5,x6,x7,x8,x9,x10))
@inline (::Type{SA}){SA<:StaticArray}(x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11) = SA((x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11))
@inline (::Type{SA}){SA<:StaticArray}(x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12) = SA((x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12))
@inline (::Type{SA}){SA<:StaticArray}(x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13) = SA((x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13))
@inline (::Type{SA}){SA<:StaticArray}(x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14) = SA((x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14))
@inline (::Type{SA}){SA<:StaticArray}(x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14,x15) = SA((x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14,x15))
@inline (::Type{SA}){SA<:StaticArray}(x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14,x15,x16) = SA((x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14,x15,x16))

@inline convert{SA<:StaticArray}(::Type{SA}, x1) = SA((x1,))

# this covers most conversions and "statically-sized reshapes"
@inline convert{SA<:StaticArray}(::Type{SA}, sa::StaticArray) = SA(Tuple(sa))
@inline function convert{SA<:StaticArray}(::Type{SA}, a::AbstractArray)
    SA(NTuple{(length(SA))}(a))
end

function convert{T,N}(::Type{Array}, sa::StaticArray{T,N})
    out = Array{T,N}(size(sa))
    @inbounds for i = 1:length(sa)
        out[i] = sa[i]
    end
    return out
end

function convert{T,N}(::Type{Array{T}}, sa::StaticArray{T,N})
    out = Array{T,N}(size(sa))
    @inbounds for i = 1:length(sa)
        out[i] = sa[i]
    end
    return out
end

function convert{T,N}(::Type{Array{T,N}}, sa::StaticArray{T,N})
    out = Array{T,N}(size(sa))
    @inbounds for i = 1:length(sa)
        out[i] = sa[i]
    end
    return out
end

function convert{T}(::Type{Matrix}, sa::StaticMatrix{T})
    out = Matrix{T}(size(sa))
    @inbounds for i = 1:length(sa)
        out[i] = sa[i]
    end
    return out
end

function convert{T}(::Type{Vector}, sa::StaticVector{T})
    out = Vector{T}(size(sa))
    @inbounds for i = 1:length(sa)
        out[i] = sa[i]
    end
    return out
end

#function convert{A<:AbstractArray}(::Type{A}, sa::StaticArray)
#    out = A(size(sa))
#    @inbounds for i = 1:length(sa)
#        out[i] = sa[i]
#    end
#    return out
#end


# A general way of going back to a tuple, etc
@generated function convert(::Type{Tuple}, a::StaticArray)
    n = length(a)
    exprs = [:(a[$j]) for j = 1:n]
    quote
        $(Expr(:meta, :inline))
        $(Expr(:tuple, exprs...))
    end
end

# We may want a pointer... usefull for LAPACK etc. However, some static arrays
# might not store a dense representation of their matrix (they may be sparse or
# some computed representation) so the fallback needs to be safe.
# Custom-defined, dense, mutable StaticArrays might want to overload this default
@inline function Base.unsafe_convert{T}(::Type{Ptr{T}}, a::StaticArray{T})
    Base.unsafe_convert(Ptr{T}, Base.data_pointer_from_objref(Tuple(a)))
end

@inline size(x::StaticArray) = size(typeof(x))

function size{SA <: StaticArray}(::Type{SA})
    error("""
        The size of type `$SA` is not known.

        If you were trying to call the construct (or `convert` to) a `StaticArray` you
        may need to add the size explicitly as a type parameter so it's size is
        inferrable to the Julia compiler (or performance would be terrible). For
        example, you might try

            m = zeros(3,3)
            SMatrix(m)      # this error
            SMatrix{3,3}(m) # correct - size is inferrable
        """)
end
