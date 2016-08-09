
@generated function det{T}(A::StaticMatrix{T})
    if size(A) == (1,1)
        return quote
            $(Expr(:meta, :inline))
            @inbounds return A[1]
        end
    elseif size(A) == (2,2)
        return quote
            $(Expr(:meta, :inline))
            @inbounds return A[1]*A[4] - A[3]*A[2]
        end
    elseif size(A) == (3,3)
        return quote
            $(Expr(:meta, :inline))
            #@inbounds a = A[5]*A[9] - A[8]*A[6]
            #@inbounds b = A[8]*A[3] - A[2]*A[9]
            #@inbounds c = A[2]*A[6] - A[5]*A[3]
            #@inbounds return A[1]*a + A[4]*b + A[7]*c

            @inbounds x0 = SVector(A[1], A[2], A[3])
            @inbounds x1 = SVector(A[4], A[5], A[6])
            @inbounds x2 = SVector(A[7], A[8], A[9])
            return vecdot(x0, cross(x1, x2))
        end
    else
        S = typeof((one(T)*zero(T) + zero(T))/one(T))
        return quote # Implementation from Base
            if istriu(A) || istril(A)
                return convert($S, det(UpperTriangular(A)))::$S # Is this a Julia bug that a convert is not type stable??
            end
            AA = convert(Array{$S}, A)
            return det(lufact(AA))
        end
    end
end
