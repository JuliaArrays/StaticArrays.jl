@inline det(A::StaticMatrix) = _det(Size(A), A)

"""
    det(Size(m,m), mat)

Calculate the matrix determinate using an algorithm specialized on the size of
the `m`×`m` matrix `mat`, which is much faster for small matrices.
"""
@inline _det(::Size{(1,1)}, A::AbstractMatrix) = @inbounds return A[1]

@inline function _det(::Size{(2,2)}, A::AbstractMatrix)
    @inbounds return A[1]*A[4] - A[3]*A[2]
end

@inline function _det(::Size{(3,3)}, A::AbstractMatrix)
    @inbounds x0 = SVector(A[1], A[2], A[3])
    @inbounds x1 = SVector(A[4], A[5], A[6])
    @inbounds x2 = SVector(A[7], A[8], A[9])
    return vecdot(x0, cross(x1, x2))
end

@generated function _det{S,T}(::Size{S}, A::AbstractMatrix{T})
    if S[1] != S[2]
        throw(DimensionMismatch("matrix is not square"))
    end
    T2 = typeof((one(T)*zero(T) + zero(T))/one(T))
    return quote # Implementation from Base
        if istriu(A) || istril(A)
            return convert($T2, det(UpperTriangular(A)))::$T2 # Is this a Julia bug that a convert is not type stable??
        end
        AA = convert(Array{$T2}, A)
        return det(lufact(AA))
    end
end
