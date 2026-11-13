# Generic Cholesky decomposition for fixed-size matrices, mostly unrolled
non_hermitian_error() = throw(LinearAlgebra.PosDefException(-1))
@inline function LinearAlgebra.cholesky(A::StaticMatrix; check::Bool = true)
    ishermitian(A) || non_hermitian_error()
    _cholesky(Size(A), A, check)
end

@inline function LinearAlgebra.cholesky(A::LinearAlgebra.RealHermSymComplexHerm{<:Real, <:StaticMatrix}; check::Bool = true)
    C = _cholesky(Size(A), (A.uplo == 'U') ? A.data : A.data', check)
    (A.uplo == 'U') ? C : Cholesky(C.factors', A.uplo, C.info)
end
@inline LinearAlgebra._chol!(A::StaticMatrix, ::Type{UpperTriangular}) = (cholesky(A).U, 0)
@inline LinearAlgebra._chol!(A::StaticMatrix, ::Type{LowerTriangular}) = (cholesky(A).L, 0)

@inline function _chol_failure(A, info, check)
    if check
        throw(LinearAlgebra.PosDefException(info))
    else
        Cholesky(A, 'U', info)
    end
end

# x - abs2(z)
@inline _subabs2(x, z) = muladd(-imag(z), imag(z), muladd(-real(z), real(z), x))

# x < zero(x) is check used in `sqrt`, letting LLVM eliminate that check and remove error code.
@inline _nonpdcheck(x::Real) = x > zero(x)

@generated function _cholesky(::Size{S}, A::StaticMatrix{M,M}, check::Bool) where {S,M}
    @assert (M,M) == S
    M > 24 && return :(_cholesky_large(Size{$S}(), A, check))
    q = Expr(:block)
    for n ∈ 1:M
        L_n_n = Symbol(:L_,n,:_,n)
        # Only the real part of the diagonal contributes to Hermitian(A)
        push!(q.args, :($L_n_n = real(@inbounds A[$n, $n])))
        for m ∈ n+1:M
            L_m_n = Symbol(:L_,m,:_,n)
            push!(q.args, :($L_m_n = @inbounds A[$n, $m]))
        end
        for k ∈ 1:n-1
            L_n_k = Symbol(:L_,n,:_,k)
            push!(q.args, :($L_n_n = _subabs2($L_n_n, $L_n_k)))
        end
        for k ∈ 1:n-1, m ∈ n+1:M
            L_m_n = Symbol(:L_,m,:_,n)
            L_m_k = Symbol(:L_,m,:_,k)
            L_n_k = Symbol(:L_,n,:_,k)
            push!(q.args, :($L_m_n = muladd(-$L_m_k, $L_n_k', $L_m_n)))
        end
        push!(q.args, :(_nonpdcheck($L_n_n) || return _chol_failure(A, $n, check)))
        push!(q.args, :($L_n_n = Base.FastMath.sqrt_fast($L_n_n)))
        Linv_n_n = Symbol(:Linv_,n,:_,n)
        push!(q.args, :($Linv_n_n = inv($L_n_n)))
        for m ∈ n+1:M
            L_m_n = Symbol(:L_,m,:_,n)
            push!(q.args, :($L_m_n *= $Linv_n_n))
        end
    end
    push!(q.args, :(T = promote_type(typeof(sqrt(one(eltype(A)))), Float32)))
    ret = Expr(:tuple)
    for n ∈ 1:M
        for m ∈ 1:n
            push!(ret.args, Symbol(:L_,n,:_,m))
        end
        for m ∈ n+1:M
            push!(ret.args, :(zero(T)))
        end
    end
    push!(q.args, :(Cholesky(similar_type(A, T)($ret), 'U', 0)))
    return Expr(:block, Expr(:meta, :inline), q)
end

# Otherwise default algorithm returning wrapped SizedArray
@inline function _cholesky_large(::Size{S}, A::StaticArray, check::Bool) where {S}
    C = cholesky(Hermitian(Matrix(A)); check=check)
    Cholesky(similar_type(A)(C.U), 'U', C.info)
end

LinearAlgebra.hermitian_type(::Type{SA}) where {T, S, SA<:SArray{S,T}} = Hermitian{T,SA}

function inv(A::Cholesky{T,<:StaticMatrix{N,N,T}}) where {N,T}
    # We can't use M = inv(A.U) here since it returns an `UpperTriangular` of a `Matrix`
    M = UpperTriangular(A.U \ SDiagonal{N}(I))
    return M * M'
end

function Base.:\(A::Cholesky{T,<:StaticMatrix{N,N,T}}, B::StaticVecOrMatLike) where {N,T}
    return A.U \ (A.U' \ B)
end

function Base.:/(B::StaticMatrixLike, A::Cholesky{T,<:StaticMatrix{N,N,T}}) where {N,T}
    return (B / A.U) / A.U'
end

