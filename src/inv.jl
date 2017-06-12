@inline inv(A::StaticMatrix) = _inv(Size(A), A)

@inline _inv(::Size{(1,1)}, A) = similar_type(typeof(A), typeof(inv(one(eltype(A)))))(inv(A[1]))

@inline function _inv(::Size{(2,2)}, A)
    T = eltype(A)
    S = typeof((one(T)*zero(T) + zero(T))/one(T))
    newtype = similar_type(A, S)

    d = det(A)
    @inbounds return newtype((A[4]/d, -(A[2]/d), -(A[3]/d), A[1]/d))
end

@inline function _inv(::Size{(3,3)}, A)
    T = eltype(A)
    S = typeof((one(T)*zero(T) + zero(T))/one(T))
    newtype = similar_type(A, S)

    @inbounds x0 = SVector(A[1], A[2], A[3])
    @inbounds x1 = SVector(A[4], A[5], A[6])
    @inbounds x2 = SVector(A[7], A[8], A[9])

    y0 = cross(x1,x2)
    d  = vecdot(x0, y0)
    x0 = x0 / d
    y0 = y0 / d
    y1 = cross(x2,x0)
    y2 = cross(x0,x1)

    @inbounds return newtype((y0[1], y1[1], y2[1], y0[2], y1[2], y2[2], y0[3], y1[3], y2[3]))
end

Base.@pure function splitrange(r::SUnitRange)
    mid = (first(r) + last(r)) ÷ 2
    (SUnitRange(first(r), mid), SUnitRange(mid+1, last(r)))
end

@noinline function _inv(::Size{(4,4)}, A)
    # Partition matrix into four 2x2 blocks.  For 4x4 matrices this seems to be
    # more stable than directly using the adjugate expansion.
    # See http://www.freevec.org/function/inverse_matrix_4x4_using_partitioning
    #
    # TODO: This decomposition works in higher dimensions, but numerical
    # stability doesn't seem good for badly conditioned matrices.  Can be
    # fixed?
    (i1,i2) = splitrange(SUnitRange(1,4))
    @inbounds P = A[i1,i1]
    @inbounds Q = A[i1,i2]
    @inbounds R = A[i2,i1]
    @inbounds S = A[i2,i2]
    invP = inv(P)
    invP_Q = invP*Q
    S2 = inv(S - R*invP_Q)
    R2 = -S2*(R*invP)
    Q2 = -invP_Q*S2
    P2 = invP - invP_Q*R2
    [[P2 Q2];
     [R2 S2]]
end

@inline function _inv(::Size, A)
    T = eltype(A)
    S = typeof((one(T)*zero(T) + zero(T))/one(T))
    AA = convert(Array{S}, A) # lufact() doesn't work with StaticArrays at the moment... and the branches below must be type-stable
    if istriu(A)
        Ai_ut = inv(UpperTriangular(AA))
        # TODO double check these routines leave the parent in a clean (upper triangular) state
        return Size(A)(parent(Ai_ut)) # Return a `SizedArray`
    elseif istril(A)
        Ai_lt = inv(LowerTriangular(AA))
        # TODO double check these routines leave the parent in a clean (lower triangular) state
        return Size(A)(parent(Ai_lt)) # Return a `SizedArray`
    else
        Ai_lu = inv(lufact(AA))
        return Size(A)(Ai_lu) # Return a `SizedArray`
    end
end
