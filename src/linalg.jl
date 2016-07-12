import Base: +, -, *, /

# TODO: more operators, like AbstractArray

# Unary ops
@inline -(a::StaticArray) = map(-, a)

# Binary ops
# Between arrays
@inline +(a::StaticArray, b::StaticArray) = map(+, a, b)
@inline +(a::AbstractArray, b::StaticArray) = map(+, a, b)
@inline +(a::StaticArray, b::AbstractArray) = map(+, a, b)

@inline -(a::StaticArray, b::StaticArray) = map(-, a, b)
@inline -(a::AbstractArray, b::StaticArray) = map(-, a, b)
@inline -(a::StaticArray, b::AbstractArray) = map(-, a, b)

# Scalar-array
@inline +(a::Number, b::StaticArray) = broadcast(+, a, b)
@inline +(a::StaticArray, b::Number) = broadcast(+, a, b)

@inline -(a::Number, b::StaticArray) = broadcast(-, a, b)
@inline -(a::StaticArray, b::Number) = broadcast(-, a, b)

@inline *(a::Number, b::StaticArray) = broadcast(*, a, b)
@inline *(a::StaticArray, b::Number) = broadcast(*, a, b)

@inline /(a::StaticArray, b::Number) = broadcast(/, a, b)


# Transpose, conjugate, etc
@inline conj(a::StaticArray) = map(conj, a)

@generated function transpose(v::StaticVector)
    n = length(v)
    newtype = similar_type(v, (1,n))
    exprs = [:(v[$j]) for j = 1:n]

    return quote
        $(Expr(:meta, :inline))
        @inbounds return $(Expr(:call, newtype, Expr(:tuple, exprs...)))
    end
end

@generated function ctranspose(v::StaticVector)
    n = length(v)
    newtype = similar_type(v, (1,n))
    exprs = [:(conj(v[$j])) for j = 1:n]

    return quote
        $(Expr(:meta, :inline))
        @inbounds return $(Expr(:call, newtype, Expr(:tuple, exprs...)))
    end
end

@generated function transpose(m::StaticMatrix)
    (s1,s2) = size(m)
    if s1 == s2
        newtype = m
    else
        newtype = similar_type(v, (s2,s1))
    end

    exprs = [:(m[$j2, $j1]) for j1 = 1:s1, j2 = 1:s2]

    return quote
        $(Expr(:meta, :inline))
        @inbounds return $(Expr(:call, newtype, Expr(:tuple, exprs...)))
    end
end

@generated function ctranspose(m::StaticMatrix)
    (s1,s2) = size(m)
    if s1 == s2
        newtype = m
    else
        newtype = similar_type(v, (s2,s1))
    end

    exprs = [:(conj(m[$j2, $j1])) for j1 = 1:s1, j2 = 1:s2]

    return quote
        $(Expr(:meta, :inline))
        @inbounds return $(Expr(:call, newtype, Expr(:tuple, exprs...)))
    end
end


#=

abstract SVector{Size,T,D} <: StaticArray{T,D}
@pure size{Size}(::SVector{Size}) = (Size,)
@pure size{T<:SVector}(::Type{T}) = (T.parameters[1],)

#typealias SVector{Size,T,D} SArray{Size,T,1,D}
typealias MVector{Size,T,D} MArray{Size,T,1,D}
typealias StaticVector{Size,T,D} Union{SVector{Size,T,D},MVector{Size,T,D}}

typealias SMatrix{Sizes,T,D} SArray{Sizes,T,2,D}
typealias MMatrix{Sizes,T,D} MArray{Sizes,T,2,D}
typealias StaticMatrix{Size,T,D} Union{SMatrix{Size,T,D},MMatrix{Size,T,D}}

# Constructors not covered by SArray
@generated Base.call{N,T}(::Type{SVector}, d::NTuple{N,T}) = :($(SVector{(N,),T,d})(d))
@generated function Base.call{N}(::Type{SVector}, d::NTuple{N})
    T = promote_tuple_eltype(d)
    return :($(SVector{(N,),T})(convert_ntuple($T, d)))
end
@generated Base.call{Size,N,T}(::Type{SVector{Size}}, d::NTuple{N,T}) = :($(SVector{Size,T,d})(d))
@generated function Base.call{Size,N}(::Type{SVector{Size}}, d::NTuple{N})
    T = promote_tuple_eltype(d)
    return :($(SVector{Size,T})(convert_ntuple($T, d)))
end

@generated Base.call{Sizes,N,T}(::Type{SMatrix{Sizes}}, d::NTuple{N,T}) = :($(SMatrix{Sizes,T,d})(d))
@generated function Base.call{Sizes,N}(::Type{SMatrix{Sizes}}, d::NTuple{N})
    T = promote_tuple_eltype(d)
    return :($(SMatrix{Sizes,T})(convert_ntuple($T, d)))
end

# Conversions to SVector from AbstractVector
@generated function Base.convert{Sizes,T}(::Type{SVector{Sizes}}, a::AbstractVector{T})
    M = prod(Sizes)
    NewType = SArray{Sizes,T,1,NTuple{M,T}}
    quote
        size(a) == Sizes || error("Input array must have size $Sizes, got $(size(a))")
        $(NewType)((a...))
    end
end

# Conversions to SMatrix from AbstractMatrix
@generated function Base.convert{Sizes,T}(::Type{SMatrix{Sizes}}, a::AbstractMatrix{T})
    M = prod(Sizes)
    NewType = SArray{Sizes,T,2,NTuple{M,T}}
    quote
        size(a) == Sizes || error("Input array must have size $Sizes, got $(size(a))")
        $(NewType)((a...))
    end
end

# Constructors not covered by MArray
@generated Base.call{N,T}(::Type{MVector}, d::NTuple{N,T}) = :($(MVector{(N,),T,d})(d))
@generated function Base.call{N}(::Type{MVector}, d::NTuple{N})
    T = promote_tuple_eltype(d)
    return :($(MVector{(N,),T})(convert_ntuple($T, d)))
end
@generated Base.call{Size,N,T}(::Type{MVector{Size}}, d::NTuple{N,T}) = :($(MVector{Size,T,d})(d))
@generated function Base.call{Size,N}(::Type{MVector{Size}}, d::NTuple{N})
    T = promote_tuple_eltype(d)
    return :($(MVector{Size,T})(convert_ntuple($T, d)))
end

@generated Base.call{Sizes,N,T}(::Type{MMatrix{Sizes}}, d::NTuple{N,T}) = :($(MMatrix{Sizes,T,d})(d))
@generated function Base.call{Sizes,N}(::Type{MMatrix{Sizes}}, d::NTuple{N})
    T = promote_tuple_eltype(d)
    return :($(MMatrix{Sizes,T})(convert_ntuple($T, d)))
end
# Conversions to MVector from AbstractVector
@generated function Base.convert{Sizes,T}(::Type{MVector{Sizes}}, a::AbstractVector{T})
    M = prod(Sizes)
    NewType = MArray{Sizes,T,1,NTuple{M,T}}
    quote
        size(a) == Sizes || error("Input array must have size $Sizes, got $(size(a))")
        $(NewType)((a...))
    end
end

# Conversions to MMatrix from AbstractMatrix
@generated function Base.convert{Sizes,T}(::Type{MMatrix{Sizes}}, a::AbstractMatrix{T})
    M = prod(Sizes)
    NewType = MArray{Sizes,T,2,NTuple{M,T}}
    quote
        size(a) == Sizes || error("Input array must have size $Sizes, got $(size(a))")
        $(NewType)((a...))
    end
end

# Conversions to Vector from StaticVector
function Base.convert{Sizes,T}(::Type{Vector},a::StaticVector{Sizes,T})
    out = Vector{T}(Sizes...)
    out[:] = a[:]
    return out
end

# Conversions to Matrix from StaticMatrix
function Base.convert{Sizes,T}(::Type{Matrix},a::StaticMatrix{Sizes,T})
    out = Matrix{T}(Sizes...)
    out[:] = a[:]
    return out
end

# Level-0 ops: Reductions *should* already work

# Level-1 ops: Mostly covered by map(), etc. Vector transpose... need operators for Julia 0.4.

# Level-2 ops: matrix-vector multiplication and matrix transpose

# Level-3 ops: matrix-matrix multiplication
=#
