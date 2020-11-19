# Generic Cholesky decomposition for fixed-size matrices, mostly unrolled
non_hermitian_error() = throw(LinearAlgebra.PosDefException(-1))
@inline function LinearAlgebra.cholesky(A::StaticMatrix)
    ishermitian(A) || non_hermitian_error()
    C = _cholesky(Size(A), A)
    return Cholesky(C, 'U', 0)
end

@inline function LinearAlgebra.cholesky(A::LinearAlgebra.RealHermSymComplexHerm{<:Real, <:StaticMatrix})
    C = _cholesky(Size(A), A.data)
    return Cholesky(C, 'U', 0)
end
@inline LinearAlgebra._chol!(A::StaticMatrix, ::Type{UpperTriangular}) = (cholesky(A).U, 0)

@generated function _cholesky(::Size{S}, A::StaticMatrix{M,M}) where {S,M}
    @assert (M,M) == S
    M > 24 && return :(_cholesky_large(Size{$S}(), A))
    q = Expr(:block)
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
        push!(q.args, :($L_n_n = sqrt($L_n_n)))
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
    push!(q.args, :(similar_type(A, T)($ret)))
    return Expr(:block, Expr(:meta, :inline), q)
end

# Otherwise default algorithm returning wrapped SizedArray
@inline _cholesky_large(::Size{S}, A::StaticArray) where {S} =
    similar_type(A)(cholesky(Hermitian(Matrix(A))).U)

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

