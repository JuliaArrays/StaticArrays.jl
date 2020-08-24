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
    M > 24 && return :(_cholesky_large(Size{$S}(), :A))
    q = Expr(:block)
    for n in 1:M
        for m ∈ n:M
	    r = Expr(:ref, :A, n, m)
	    ln = LineNumberNode(@__LINE__,Symbol(@__FILE__))
	    r = Expr(:macrocall, Symbol("@inbounds"), ln, r)
            push!(q.args, Expr(:(=), Symbol(:L_,m,:_,n), r))
        end
        for k ∈ 1:n-1, m in n:M
            L_m_n = Symbol(:L_,m,:_,n)
            push!(q.args, Expr(:(=), L_m_n, Expr(:call, :muladd, Expr(:call, :(-), Symbol(:L_,m,:_,k)), Symbol(:L_,n,:_,k), L_m_n)))
        end
        L_n_n = Symbol(:L_,n,:_,n)
        push!(q.args, Expr(:(=), L_n_n, Expr(:call, :sqrt, L_n_n)))
        Linv_n_n = Symbol(:Linv_,n,:_,n)
        push!(q.args, Expr(:(=), Linv_n_n, Expr(:call, :inv, L_n_n)))
        for m ∈ n+1:M
            push!(q.args, Expr(:(*=), Symbol(:L_,m,:_,n), Linv_n_n))
        end
    end
    push!(q.args, :(T = promote_type(typeof(sqrt(one(eltype(A)))), Float32)))
    ret = Expr(:tuple)
    for n ∈ 1:M
        for m ∈ 1:n
            push!(ret.args, Symbol(:L_,n,:_,m))
        end
        for m ∈ n+1:M
            push!(ret.args, Expr(:call, :zero, :T))
        end
    end
    push!(q.args, Expr(:call, Expr(:call, :similar_type, :A, :T), ret))
    Expr(:block, Expr(:meta, :inline), q)
end


# Otherwise default algorithm returning wrapped SizedArray
@inline _cholesky_large(::Size{S}, A::StaticArray) where {S} =
    SizedArray{Tuple{S...}}(Matrix(cholesky(Hermitian(Matrix(A))).U))

LinearAlgebra.hermitian_type(::Type{SA}) where {T, S, SA<:SArray{S,T}} = Hermitian{T,SA}
