# Generic Cholesky decomposition for fixed-size matrices, mostly unrolled
@inline function Base.chol(A::StaticMatrix)
    ishermitian(A) || Base.LinAlg.non_hermitian_error("chol")
    _chol(Size(A), A)
end

@inline function Base.chol{T<:Real, SM <: StaticMatrix}(A::Base.LinAlg.RealHermSymComplexHerm{T,SM})
    ishermitian(A) || Base.LinAlg.non_hermitian_error("chol")
    _chol(Size(A), A)
end

@inline function Base.chol{SM<:StaticMatrix}(A::Symmetric{SM})
    eltype(A) <: Real && (ishermitian(A) || Base.LinAlg.non_hermitian_error("chol"))
    _chol(Size(A), A)
end

@generated function _chol(::Size{(1,1)}, A::StaticMatrix)
    @assert size(A) == (1,1)
    T = promote_type(typeof(sqrt(one(eltype(A)))), Float32)
    newtype = similar_type(A,T)

    quote
        $(Expr(:meta, :inline))
        ($newtype)((sqrt(A[1]), ))
    end
end

@generated function _chol(::Size{(2,2)}, A::StaticMatrix)
    @assert size(A) == (2,2)
    T = promote_type(typeof(sqrt(one(eltype(A)))), Float32)
    newtype = similar_type(A,T)

    quote
        $(Expr(:meta, :inline))
        @inbounds a = sqrt(A[1])
        @inbounds b = A[3] / a
        @inbounds c = sqrt(A[4] - b'*b)
        ($newtype)((a, $(zero(T)), b, c))
    end
end

@generated function _chol(::Size{(3,3)}, A::StaticMatrix)
    @assert size(A) == (3,3)
    T = promote_type(typeof(sqrt(one(eltype(A)))), Float32)
    newtype = similar_type(A,T)

    quote
        $(Expr(:meta, :inline))
        @inbounds a11 = sqrt(A[1])
        @inbounds a12 = A[4] / a11
        @inbounds a22 = sqrt(A[5] - a12'*a12)
        @inbounds a13 = A[7] / a11
        @inbounds a23 = (A[8] - a12'*a13) / a22
        @inbounds a33 = sqrt(A[9] - a13'*a13 - a23'*a23)
        ($newtype)((a11, $(zero(T)), $(zero(T)), a12, a22, $(zero(T)), a13, a23, a33))
    end
end

# Otherwise default algorithm returning wrapped SizedArray
@inline _chol(s::Size, A::StaticArray) = s(full(chol(Hermitian(Array(A)))))
