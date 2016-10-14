@generated function (\){T}(A::StaticMatrix{T}, b::StaticVector{T})
    S = typeof((one(T)*zero(T) + zero(T))/one(T))
    newtype = similar_type(b, S)

    if size(A) == (1,1) && length(b) == 1
        return quote
            $(Expr(:meta, :inline))
            @inbounds return $newtype( b[1] / A[1,1] )
        end
    elseif size(A) == (2,2) && length(b) == 2
        return quote
            $(Expr(:meta, :inline))
            d = det(A)
            @inbounds return $newtype(
                A[2,2]*b[1] - A[1,2]*b[2],
                A[1,1]*b[2] - A[2,1]*b[1] ) / d
        end
    elseif size(A) == (3,3) && length(b) == 3
        return quote
            $(Expr(:meta, :inline))
            d = det(A)
            @inbounds return $newtype(
                (A[2,2]*A[3,3] - A[2,3]*A[3,2])*b[1] +
                    (A[1,3]*A[3,2] - A[1,2]*A[3,3])*b[2] +
                        (A[1,2]*A[2,3] - A[1,3]*A[2,2])*b[3],
                (A[2,3]*A[3,1] - A[2,1]*A[3,3])*b[1] +
                    (A[1,1]*A[3,3] - A[1,3]*A[3,1])*b[2] +
                        (A[1,3]*A[2,1] - A[1,1]*A[2,3])*b[3],
                (A[2,1]*A[3,2] - A[2,2]*A[3,1])*b[1] +
                    (A[1,2]*A[3,1] - A[1,1]*A[3,2])*b[2] +
                        (A[1,1]*A[2,2] - A[1,2]*A[2,1])*b[3] ) / d
        end
    else
        # FixMe! Unsatisfactory ineffective but requires some infrastructure
        # to make efficient so we fall back on inv for now
        quote
            inv(A)*b
        end
    end
end