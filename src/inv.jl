
@generated function inv{T}(A::StaticMatrix{T})
    S = typeof((one(T)*zero(T) + zero(T))/one(T))
    newtype = similar_type(A, S)

    if size(A) == (1,1)
        return quote
            $(Expr(:meta, :inline))
            @inbounds return $newtype((inv(A[1]),))
        end
    elseif size(A) == (2,2)
        return quote
            $(Expr(:meta, :inline))
            d = A[1]*A[4] - A[3]*A[2]
            @inbounds return $newtype((A[4]/d, -A[2]/d, -A[3]/d, A[1]/d))
        end
    elseif size(A) == (3,3)
        return quote
            $(Expr(:meta, :inline))
            @inbounds x0 = SVector(A[1], A[2], A[3])
            @inbounds x1 = SVector(A[4], A[5], A[6])
            @inbounds x2 = SVector(A[7], A[8], A[9])

            y0 = cross(x1,x2)
            d  = vecdot(x0, y0)
            x0 = x0 / d
            y0 = y0 / d
            y1 = cross(x2,x0)
            y2 = cross(x0,x1)

            @inbounds return $newtype((y0[1], y1[1], y2[1], y0[2], y1[2], y2[2], y0[3], y1[3], y2[3]))
        end
    else
        return quote # Implementation from Base
            AA = convert(Array{$S}, A) # lufact() doesn't work with StaticArrays at the moment... and the branches below must be type-stable
            if istriu(A)
                Ai = inv(UpperTriangular(AA))
            elseif istril(A)
                Ai = inv(LowerTriangular(AA))
            else
                Ai = inv(lufact(AA))
            end
            # Return a `SizedArray`
            return $(Size(A))(convert(typeof(parent(Ai)), Ai))
        end
    end
end
