"""
    abstract type StaticArray{T, N} <: AbstractArray{T, N} end
    StaticScalar{T} = StaticArray{T, 0}
    StaticVector{T} = StaticArray{T, 1}
    StaticMatrix{T} = StaticArray{T, 2}

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

 - `setindex!` for a single element (linear indexing).
 - `similar(::Type{MyStaticArray}, ::Type{NewElType}, ::Size{NewSize})`.
 - In some cases, a zero-parameter constructor, `MyStaticArray{...}()` for unintialized data
   is assumed to exist.

(see also `SVector`, `SMatrix`, `SArray`, `MVector`, `MMatrix`, `MArray`, `SizedArray` and `FieldVector`)
"""
abstract type StaticArray{T, N} <: AbstractArray{T, N} end

StaticScalar{T} = StaticArray{T, 1}
StaticVector{T} = StaticArray{T, 1}
StaticMatrix{T} = StaticArray{T, 2}

(::Type{SA})(x::Tuple) where {SA <: StaticArray} = error("No precise constructor for $SA found. Length of input was $(length(x)).")

@inline convert(::Type{SA}, x...) where {SA <: StaticArray} = SA(x)

# this covers most conversions and "statically-sized reshapes"
@inline convert(::Type{SA}, sa::StaticArray) where {SA<:StaticArray} = SA(Tuple(sa))
@inline convert(::Type{SA}, sa::SA) where {SA<:StaticArray} = sa

# A general way of going back to a tuple
@inline function convert(::Type{Tuple}, a::StaticArray)
    unroll_tuple((i -> @inbounds return a[i]), length_val(a))
end

@inline function convert(::Type{SA}, a::AbstractArray) where {SA <: StaticArray}
    if length(a) != length(SA)
        error("Dimension mismatch. Expected input array of length $(length(SA)), got length $(length(a))")
    end

    return SA(unroll_tuple((i -> @inbounds return a[i]), length_val(SA)))
end


#=
@generated function convert(::Type{Tuple}, a::StaticArray)
    n = length(a)
    exprs = [:(a[$j]) for j = 1:n]
    quote
        $(Expr(:meta, :inline))
        @inbounds return $(Expr(:tuple, exprs...))
    end
end
=#



# People might not want to use Tuple for everything (TODO: check this with FieldVector...)
# Generic case, with least 2 inputs
#@inline (::Type{SA}){SA<:StaticArray}(x1,x2,xs...) = SA((x1,x2,xs...))


#=
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
=#
