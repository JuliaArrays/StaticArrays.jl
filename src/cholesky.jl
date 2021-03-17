# Generic Cholesky decomposition for fixed-size matrices, mostly unrolled
non_hermitian_error() = throw(LinearAlgebra.PosDefException(-1))
@inline function LinearAlgebra.cholesky(A::StaticMatrix; check::Bool = true)
    ishermitian(A) || non_hermitian_error()
    _cholesky(Size(A), A, check)
    # (check && (info ≠ 0)) && throw(LinearAlgebra.PosDefException(info))
    # return Cholesky(C, 'U', info)
end

@inline function LinearAlgebra.cholesky(A::LinearAlgebra.RealHermSymComplexHerm{<:Real, <:StaticMatrix}; check::Bool = true)
    C = _cholesky(Size(A), A.data, check)
    # (check && (info ≠ 0)) && throw(LinearAlgebra.PosDefException(info))
    # return Cholesky(C, 'U', 0)
end
@inline LinearAlgebra._chol!(A::StaticMatrix, ::Type{UpperTriangular}) = (cholesky(A).U, 0)

@inline function _check_chol(A, info, check)
    if check
        throw(LinearAlgebra.PosDefException(info))
    else
        return Cholesky(A, 'U', info)
    end
end
@inline _nonpdcheck(x::Real) = x < zero(x)
@inline _nonpdcheck(x) = false

@generated function _cholesky(::Size{S}, A::StaticMatrix{M,M}, check::Bool) where {S,M}
    @assert (M,M) == S
    M > 24 && return :(_cholesky_large(Size{$S}(), A))
    q = Expr(:block, :(info = 0), :(failure = false))
    for n ∈ 1:M
        for m ∈ n:M
            L_m_n = Symbol(:L_,m,:_,n)
            push!(q.args, :($L_m_n = @inbounds A[$n, $m]))
        end
        for k ∈ 1:n-1, m ∈ n:M
            L_m_n = Symbol(:L_,m,:_,n)
            L_m_k = Symbol(:L_,m,:_,k)
            L_n_k = Symbol(:L_,n,:_,k)
            push!(q.args, :($L_m_n = muladd(-$L_m_k, $L_n_k', $L_m_n)))
        end
        L_n_n = Symbol(:L_,n,:_,n)
        L_n_n_ltz = Symbol(:L_,n,:_,n,:_,:ltz)
        # x < 0.0 is check used in `sqrt`, letting LLVM eliminate that check and remove error code.
        # push!(q.args, :($L_n_n_ltz = )
        push!(q.args, :($L_n_n = _nonpdcheck($L_n_n) ? (return _check_chol(A, $n, check)) : sqrt($L_n_n)))
        # push!(q.args, :(info = ($L_n_n_ltz & (!failure)) ? $n : info))
        # push!(q.args, :(failure |= $L_n_n_ltz))
        # push!(q.args, :($L_n_n = $L_n_n_ltz ? float(typeof($L_n_n))(NaN) : sqrt($L_n_n)))
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
@inline _cholesky_large(::Size{S}, A::StaticArray) where {S} =
    Cholesky(similar_type(A)(cholesky(Hermitian(Matrix(A))).U), 'U', 0)

LinearAlgebra.hermitian_type(::Type{SA}) where {T, S, SA<:SArray{S,T}} = Hermitian{T,SA}

function inv(A::Cholesky{T,<:StaticMatrix{N,N,T}}) where {N,T}
    return A.U \ (A.U' \ SDiagonal{N}(I))
end

function Base.:\(A::Cholesky{T,<:StaticMatrix{N,N,T}}, B::StaticVecOrMatLike) where {N,T}
    return A.U \ (A.U' \ B)
end

function Base.:/(B::StaticMatrixLike, A::Cholesky{T,<:StaticMatrix{N,N,T}}) where {N,T}
    return (B / A.U) / A.U'
end

