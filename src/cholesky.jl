import Base: chol, cholfact, \

@inline _chol(A::StaticMatrix{<:Any,<:Any,T}, ::Type{UpperTriangular}) where {T} = _chol(Size(A), A, UpperTriangular)
@inline _chol(A::StaticMatrix{<:Any,<:Any,T}, ::Type{LowerTriangular}) where {T} = _chol(Size(A), A, LowerTriangular)

@generated function _chol(::Size{s}, A::StaticMatrix{<:Any,<:Any,T}, ::Type{UpperTriangular}) where {s, T}
    if s[1] != s[2]
        error("matrix must be square")
    end
    n = s[1]
    TX = promote_type(typeof(chol(one(T), UpperTriangular)), Float32)

    X = [Symbol("X_$(i)_$(j)") for i = 1:n, j = 1:n]
    init = [:($(X[i,j]) = A[$(sub2ind(s,i,j))]) for i = 1:n, j = 1:n]

    code = quote end
    for k = 1:n
        ex = :($(X[k,k]))
        for i = 1:k-1
            ex = :($ex - $(X[i,k])'*$(X[i,k]))
        end
        push!(code.args, quote $(X[k,k]), info = _chol($ex, UpperTriangular) end)
        push!(code.args, :(info == 0 || return UpperTriangular(similar_type(A, $TX)(tuple($(X...)))), info))
        if k < n
            push!(code.args, :(XkkInv = inv($(X[k,k])')))
        end
        for j = k + 1:n
            ex = :($(X[k,j]))
            for i = 1:k-1
                ex = :($ex - $(X[i,k])'*$(X[i,j]))
            end
            push!(code.args, :($(X[k,j]) = XkkInv*$ex))
        end
    end

    quote
        @_inline_meta
        @inbounds $(Expr(:block, init...))
        @inbounds $code
        @inbounds return UpperTriangular(similar_type(A, $TX)(tuple($(X...)))), convert(Base.LinAlg.BlasInt, 0)
    end
end

@generated function _chol(::Size{s}, A::StaticMatrix{<:Any, <:Any, T}, ::Type{LowerTriangular}) where {s, T}
    if s[1] != s[2]
        error("matrix must be square")
    end

    n = s[1]
    TX = promote_type(typeof(chol(one(T), LowerTriangular)), Float32)

    X = [Symbol("X_$(i)_$(j)") for i = 1:n, j = 1:n]
    init = [:($(X[i,j]) = A[$(sub2ind(s,i,j))]) for i = 1:n, j = 1:n]

    code = quote end
    for k = 1:n
        ex = :($(X[k,k]))
        for i = 1:k-1
            ex = :($ex - $(X[k,i])*$(X[k,i])')
        end
        push!(code.args, quote $(X[k,k]), info = _chol($ex, LowerTriangular) end)
        push!(code.args, :(info == 0 || return LowerTriangular(similar_type(A, $TX)(tuple($(X...)))), info))
        if k < n
            push!(code.args, :(XkkInv = inv($(X[k,k])')))
        end
        for j = 1:k-1
            for i = k+1:n
                push!(code.args, :($(X[i,k]) -= $(X[i,j])*$(X[k,j])'))
            end
        end
        for i = k+1:n
            push!(code.args, :($(X[i,k]) *= XkkInv))
        end
    end

    quote
        @_inline_meta
        @inbounds $(Expr(:block, init...))
        @inbounds $code
        @inbounds return LowerTriangular(similar_type(A, $TX)(tuple($(X...)))), convert(Base.LinAlg.BlasInt, 0)
    end
end

## Numbers
_chol(x::Number, uplo) = Base.LinAlg._chol!(x, uplo)

@inline function chol(A::Base.LinAlg.RealHermSymComplexHerm{<:Real,<:StaticMatrix})
    C, info = _chol(Size(A), A.uplo == 'U' ? A.data : ctranspose(A.data), UpperTriangular)
    Base.LinAlg.@assertposdef C info
end

@inline function chol(A::StaticMatrix)
    ishermitian(A) || Base.LinAlg.non_hermitian_error("chol")
    return chol(Hermitian(A))
end

function cholfact(A::Base.LinAlg.RealHermSymComplexHerm{<:Real,<:StaticMatrix}, ::Type{Val{false}}=Val{false})
    if A.uplo == 'U'
        CU, info = _chol(Size(A), A.data, UpperTriangular)
        Base.LinAlg.Cholesky(CU.data, 'U', info)
    else
        CL, info = _chol(Size(A), A.data, LowerTriangular)
        Base.LinAlg.Cholesky(CL.data, 'L', info)
    end
end

@inline function cholfact(A::StaticMatrix, ::Type{Val{false}}=Val{false})
    ishermitian(A) || Base.LinAlg.non_hermitian_error("cholfact")
    cholfact(Hermitian(A), Val{false})
end

function \(C::Base.LinAlg.Cholesky{<:Any,<:StaticMatrix}, B::StaticVecOrMat)
    if C.uplo == 'L'
        return LowerTriangular(C.factors)' \ (LowerTriangular(C.factors) \ B)
    else
        return UpperTriangular(C.factors) \ (UpperTriangular(C.factors)' \ B)
    end
end
