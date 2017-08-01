@inline function det(A::StaticMatrix)
    T = eltype(A)
    S = typeof((one(T)*zero(T) + zero(T))/one(T))
    _det(Size(A),A,S)
end

@inline logdet(A::StaticMatrix) = log(det(A))

@inline _det(::Size{(1,1)}, A::StaticMatrix,::Type) = @inbounds return A[1]

@inline function _det(::Size{(2,2)}, A::StaticMatrix, S::Type)
    A = similar_type(A,S)(A)
    @inbounds return A[1]*A[4] - A[3]*A[2]
end

@inline function _det(::Size{(3,3)}, A::StaticMatrix, S::Type)
    A = similar_type(A,S)(A)
    @inbounds x0 = SVector(A[1], A[2], A[3])
    @inbounds x1 = SVector(A[4], A[5], A[6])
    @inbounds x2 = SVector(A[7], A[8], A[9])
    return vecdot(x0, cross(x1, x2))
end

@inline function _det(::Size, A::StaticMatrix,::Type)
    return det(Matrix(A))
end
