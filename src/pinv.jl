# Moore-Penrose pseudoinverse

@inline function pinv(A::StaticMatrix{m,n,T} where m where n; atol::Real = 0.0, rtol::Real = (eps(real(float(one(T))))*min(size(A)...))*iszero(atol)) where T
    # This function is a StaticMatrix version of `LinearAlgebra.pinv`.
    S = typeof(zero(T)/sqrt(one(T) + one(T)))
    A_S = convert(similar_type(A,S),A)
    return _pinv(A_S, atol, rtol)
end

@inline function _pinv(A::StaticMatrix{m,n,T}, atol::Real, rtol::Real) where T where m where n
    if m == 0 || n == 0
        return similar_type(A, Size(n,m))()
    end
    if isdiag(A)
        maxabsA = maximum(abs.(diag(A)))
        tol = max(rtol*maxabsA, atol)
        return _pinv_diag(A, tol)
    end
    ssvd = svd(A, full = false)
    tol = max(rtol*maximum(ssvd.S), atol)
    sinv = _pinv_vector(ssvd.S, tol)
    return ssvd.Vt'*SDiagonal(sinv)*ssvd.U'
end

@inline function pinv(D::Diagonal{T,<:StaticVector}) where T
    V = D.diag
    S = typeof(zero(T)/sqrt(one(T) + one(T)))
    V_S = convert(similar_type(V,S),V)
    return Diagonal(_pinv_vector(V_S))
end

@generated function _pinv_diag(A::StaticMatrix{m,n,T}, tol) where m where n where T
    minlen = min(m,n)
    exprs = [:(zero($T)) for i in 1:n, j in 1:m]
    for i in 1:minlen
        exprs[i,i] = :(ifelse(abs(A[$i,$i]) > tol, inv(A[$i,$i]), zero($T)))
    end
    return quote
        Base.@_inline_meta
        @inbounds return similar_type(A, Size($n, $m))(tuple($(exprs...)))
    end
end

@generated function _pinv_vector(v::StaticVector{n,T}, tol) where n where T
    exprs = [
        :(ifelse(v[$i] > tol, inv(v[$i]), zero(T)))
        for i in 1:n
    ]
    return quote
        Base.@_inline_meta
        @inbounds return similar_type(v, Size($n))(tuple($(exprs...)))
    end
end

@generated function _pinv_vector(v::StaticVector{n,T}) where n where T
    exprs = [
        :(pinv(v[$i]))
        for i in 1:n
    ]
    return quote
        Base.@_inline_meta
        @inbounds return similar_type(v, Size($n))(tuple($(exprs...)))
    end
end
