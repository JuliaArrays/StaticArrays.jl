"""
    abstract StaticArray{T, N} <: AbstractArray{T, N}
    typealias StaticVector{T} StaticArray{T, 1}
    typealias StaticMatrix{T} StaticArray{T, 2}

`StaticArray`s are Julia arrays with fixed, known size.

## Dev docs

They must define the following methods:
 - Constructors that accept a flat tuple of data.
 - `Size()` on the *type*, returning an *instance* of `Size{(dim1, dim2, ...)}` (preferably `@pure`).
 - `getindex()` with an integer (linear indexing) (preferably `@inline` with `@boundscheck`).
 - `Tuple()`, returning the data in a flat Tuple.

It may be useful to implement:

- `similar_type(::Type{MyStaticArray}, ::Type{NewElType}, ::Size{NewSize})`, returning a
  type (or type constructor) that accepts a flat tuple of data.

For mutable containers you may also need to define the following:

 - `setindex!` for a single elmenent (linear indexing).
 - `similar(::Type{MyStaticArray}, ::Type{NewElType}, ::Size{NewSize})`.
 - In some cases, a zero-parameter constructor, `MyStaticArray{...}()` for unintialized data
   is assumed to exist.

(see also `SVector`, `SMatrix`, `SArray`, `MVector`, `MMatrix`, `MArray`, `SizedArray` and `FieldVector`)
"""
abstract StaticArray{T, N} <: AbstractArray{T, N}

typealias StaticVector{T} StaticArray{T, 1}
typealias StaticMatrix{T} StaticArray{T, 2}

# People might not want to use Tuple for everything (TODO: check this with FieldVector...)
# Generic case, with least 2 inputs
@inline (::Type{SA}){SA<:StaticArray}(x1,x2,xs...) = SA((x1,x2,xs...))

@inline convert{SA<:StaticArray}(::Type{SA}, x::Tuple) = error("No precise constructor found. Length of input was $(length(x)) while length of $SA is $(length(SA)).")

# Avoiding splatting penalties. Being here, implementations of StaticArray will not have to deal with these. TODO check these are necessary or not
#@inline (::Type{SA}){SA<:StaticArray}(x1) = SA((x1,)) # see convert below (lesser precedence than other constructors?)
@inline convert{SA<:StaticArray}(::Type{SA},x1,x2) = SA((x1,x2))
@inline convert{SA<:StaticArray}(::Type{SA},x1,x2,x3) = SA((x1,x2,x3))
@inline convert{SA<:StaticArray}(::Type{SA},x1,x2,x3,x4) = SA((x1,x2,x3,x4))
@inline convert{SA<:StaticArray}(::Type{SA},x1,x2,x3,x4,x5) = SA((x1,x2,x3,x4,x5))
@inline convert{SA<:StaticArray}(::Type{SA},x1,x2,x3,x4,x5,x6) = SA((x1,x2,x3,x4,x5,x6))
@inline convert{SA<:StaticArray}(::Type{SA},x1,x2,x3,x4,x5,x6,x7) = SA((x1,x2,x3,x4,x5,x6,x7))
@inline convert{SA<:StaticArray}(::Type{SA},x1,x2,x3,x4,x5,x6,x7,x8) = SA((x1,x2,x3,x4,x5,x6,x7,x8))
@inline convert{SA<:StaticArray}(::Type{SA},x1,x2,x3,x4,x5,x6,x7,x8,x9) = SA((x1,x2,x3,x4,x5,x6,x7,x8,x9))
@inline convert{SA<:StaticArray}(::Type{SA},x1,x2,x3,x4,x5,x6,x7,x8,x9,x10) = SA((x1,x2,x3,x4,x5,x6,x7,x8,x9,x10))
@inline convert{SA<:StaticArray}(::Type{SA},x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11) = SA((x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11))
@inline convert{SA<:StaticArray}(::Type{SA},x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12) = SA((x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12))
@inline convert{SA<:StaticArray}(::Type{SA},x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13) = SA((x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13))
@inline convert{SA<:StaticArray}(::Type{SA},x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14) = SA((x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14))
@inline convert{SA<:StaticArray}(::Type{SA},x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14,x15) = SA((x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14,x15))
@inline convert{SA<:StaticArray}(::Type{SA},x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14,x15,x16) = SA((x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14,x15,x16))

@inline convert{SA<:StaticArray}(::Type{SA}, x1) = SA((x1,))

# this covers most conversions and "statically-sized reshapes"
@inline convert{SA<:StaticArray}(::Type{SA}, sa::StaticArray) = SA(Tuple(sa))

@inline convert{SA<:StaticArray}(::Type{SA}, sa::SA) = sa

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

# A general way of going back to a tuple, etc
@generated function convert(::Type{Tuple}, a::StaticArray)
    n = length(a)
    exprs = [:(a[$j]) for j = 1:n]
    quote
        $(Expr(:meta, :inline))
        @inbounds return $(Expr(:tuple, exprs...))
    end
end
