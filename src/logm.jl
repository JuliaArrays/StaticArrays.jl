@inline log(A::StaticMatrix) = _log(Size(A), A)

@inline function _log(::Size{(0,0)}, A::StaticMatrix)
    T = typeof(log(one(eltype(A))))
    SMT = similar_type(A,T)
    return SMT()
end

@inline function _log(::Size{(1,1)}, A::StaticMatrix)
    T = typeof(log(one(eltype(A))))
    SMT = similar_type(A,T)
    return SMT((log(A[1]), ))
end

function _log(::Size, A::StaticMatrix)
    eigA = eigen(A)
    T = typeof(log(one(eltype(A)) * one(eltype(eigA.values))))
    VT = similar_type(typeof(eigA.values),T)
    log_values = log.(VT(eigA.values))
    log_eigA = Eigen(log_values, eigA.vectors)
    logA = AbstractMatrix(log_eigA)
    return logA
end
