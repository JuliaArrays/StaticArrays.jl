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
        newtype = similar_type(m, (s2,s1))
    end

    exprs = [:(m[$j1, $j2]) for j2 = 1:s2, j1 = 1:s1]

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
        newtype = similar_type(m, (s2,s1))
    end

    exprs = [:(conj(m[$j1, $j2])) for j2 = 1:s2, j1 = 1:s1]

    return quote
        $(Expr(:meta, :inline))
        @inbounds return $(Expr(:call, newtype, Expr(:tuple, exprs...)))
    end
end


@inline vcat(a::Union{StaticVector,StaticMatrix}) = a
@generated function vcat(a::Union{StaticVector, StaticMatrix}, b::Union{StaticVector,StaticMatrix})
    if size(a,2) != size(b,2)
        error("Dimension mismatch")
    end

    if a <: StaticVector && b <: StaticVector
        newtype = similar_type(a, (length(a) + length(b),))
        exprs = vcat([:(a[$i]) for i = 1:length(a)],
                     [:(b[$i]) for i = 1:length(b)])
    else
        newtype = similar_type(a, (size(a,1) + size(b,1), size(a,2)))
        exprs = [((i <= size(a,1)) ? ((a <: StaticVector) ? :(a[$i]) : :(a[$i,$j]))
                                   : ((b <: StaticVector) ? :(b[$(i-size(a,1))]) : :(b[$(i-size(a,1)),$j])))
                                   for i = 1:(size(a,1)+size(b,1)), j = 1:size(a,2)]
    end

    return quote
        $(Expr(:meta, :inline))
        @inbounds return $(Expr(:call, newtype, Expr(:tuple, exprs...)))
    end
end
# TODO make these more efficient
@inline vcat(a::Union{StaticVector,StaticMatrix}, b::Union{StaticVector,StaticMatrix}, c::Union{StaticVector,StaticMatrix}) =
    vcat(vcat(a,b), c)
@inline vcat(a::Union{StaticVector,StaticMatrix}, b::Union{StaticVector,StaticMatrix}, c::Union{StaticVector,StaticMatrix}...) =
    vcat(vcat(a,b), c...)

@generated function hcat(a::StaticVector)
    newtype = similar_type(a, (length(a),1))
    exprs = [:(a[$i]) for i = 1:length(a)]
    return quote
        $(Expr(:meta, :inline))
        @inbounds return $(Expr(:call, newtype, Expr(:tuple, exprs...)))
    end
end
@inline hcat(a::StaticMatrix) = a
@generated function hcat(a::Union{StaticVector,StaticMatrix}, b::Union{StaticVector,StaticMatrix})
    if size(a,1) != size(b,1)
        error("Dimension mismatch")
    end

    exprs1 = [:(a[$i]) for i = 1:length(a)]
    exprs2 = [:(b[$i]) for i = 1:length(b)]

    newtype = similar_type(a, (size(a,1), size(a,2) + size(b,2)))

    return quote
        $(Expr(:meta, :inline))
        @inbounds return $(Expr(:call, newtype, Expr(:tuple, exprs1..., exprs2...)))
    end
end
# TODO make these more efficient
@inline hcat(a::Union{StaticVector,StaticMatrix}, b::Union{StaticVector,StaticMatrix}, c::Union{StaticVector,StaticMatrix}) =
    hcat(hcat(a,b), c)
@inline hcat(a::Union{StaticVector,StaticMatrix}, b::Union{StaticVector,StaticMatrix}, c::Union{StaticVector,StaticMatrix}...) =
    hcat(hcat(a,b), c...)

@generated function eye{SA <: StaticArray}(::Type{SA})
    s = size(SA)
    T = eltype(SA)
    if T == Any
        T = Float64
    end
    if length(s) != 2
        error("Must call `eye` with a two-dimensional array. Got size $s")
    end
    e = eye(T, s...)
    return quote
        $(Expr(:meta, :inline))
        $(Expr(:call, SA, Expr(:tuple, e...)))
    end
end

@generated function eye{SA <: StaticMatrix}(::SA)
    s = size(SA)
    T = eltype(SA)
    e = eye(T, s...)
    return quote
        $(Expr(:meta, :inline))
        $(Expr(:call, SA, Expr(:tuple, e...)))
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
