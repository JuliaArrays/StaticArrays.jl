@inline function eig(A::StaticMatrix; permute::Bool=true, scale::Bool=true)
    _eig(Size(A), A, permute, scale)
end

@inline function eig{T, SM <: StaticMatrix}(A::Base.LinAlg.HermOrSym{T,SM}; permute::Bool=true, scale::Bool=true)
    _eig(Size(SM), A, permute, scale)
end

@inline function _eig(s::Size, A::StaticMatrix, permute, scale)
    # Only cover the hermitian branch, for now ast least
    # This also solves some type-stability issues such as arise in Base
    if ishermitian(A)
       return _eig(s, Hermitian(A), permute, scale)
    else
       error("Only hermitian matrices are diagonalizable by *StaticArrays*. Non-Hermitian matrices should be converted to `Array` first.")
    end
end

@inline function _eig{T<:Real}(s::Size, A::Base.LinAlg.RealHermSymComplexHerm{T}, permute, scale)
    eigen = eigfact(Hermitian(Array(parent(A))); permute=permute, scale=scale)
    return (s(eigen.values), s(eigen.vectors)) # Return a SizedArray
end


@inline function _eig{T<:Real}(::Size{(1,1)}, A::Base.LinAlg.RealHermSymComplexHerm{T}, permute, scale)
    @inbounds return (SVector{1,T}((A[1],)), eye(SMatrix{1,1,T}))
end

# TODO adapt the below to be complex-safe?
@inline function _eig{T<:Real}(::Size{(2,2)}, A::Base.LinAlg.RealHermSymComplexHerm{T}, permute, scale)
    a = A.data

    if A.uplo == 'U'
        @inbounds t_half = real(a[1] + a[4])/2
        @inbounds d = real(a[1]*a[4] - a[3]'*a[3]) # Should be real

        tmp2 = t_half*t_half - d
        tmp2 < 0 ? tmp = zero(tmp2) : tmp = sqrt(tmp2) # Numerically stable for identity matrices, etc.
        vals = SVector(t_half - tmp, t_half + tmp)

        @inbounds if a[3] == 0
            vecs = eye(SMatrix{2,2,T})
        else
            @inbounds v11 = vals[1]-a[4]
            @inbounds n1 = sqrt(v11'*v11 + a[3]'*a[3])
            v11 = v11 / n1
            @inbounds v12 = a[3]' / n1

            @inbounds v21 = vals[2]-a[4]
            @inbounds n2 = sqrt(v21'*v21 + a[3]'*a[3])
            v21 = v21 / n2
            @inbounds v22 = a[3]' / n2

            vecs = @SMatrix [ v11  v21 ;
                              v12  v22 ]
        end
        return (vals,vecs)
    else
        @inbounds t_half = real(a[1] + a[4])/2
        @inbounds d = real(a[1]*a[4] - a[2]'*a[2]) # Should be real

        tmp2 = t_half*t_half - d
        tmp2 < 0 ? tmp = zero(tmp2) : tmp = sqrt(tmp2) # Numerically stable for identity matrices, etc.
        vals = SVector(t_half - tmp, t_half + tmp)

        @inbounds if a[2] == 0
            vecs = eye(SMatrix{2,2,T})
        else
            @inbounds v11 = vals[1]-a[4]
            @inbounds n1 = sqrt(v11'*v11 + a[2]'*a[2])
            v11 = v11 / n1
            @inbounds v12 = a[2] / n1

            @inbounds v21 = vals[2]-a[4]
            @inbounds n2 = sqrt(v21'*v21 + a[2]'*a[2])
            v21 = v21 / n2
            @inbounds v22 = a[2] / n2

            vecs = @SMatrix [ v11  v21 ;
                              v12  v22 ]
        end
        return (vals,vecs)
    end
end

# TODO fix for complex case
@generated function _eig{T<:Real}(::Size{(3,3)}, A::Base.LinAlg.RealHermSymComplexHerm{T}, permute, scale)
    S = typeof((one(T)*zero(T) + zero(T))/one(T))

    return quote
        $(Expr(:meta, :inline))

        uplo = A.uplo
        data = A.data
        if uplo == 'U'
            @inbounds Afull = SMatrix{3,3}(data[1], data[4], data[7], data[4], data[5], data[8], data[7], data[8], data[9])
        else
            @inbounds Afull = SMatrix{3,3}(data[1], data[2], data[3], data[2], data[5], data[6], data[3], data[6], data[9])
        end

        # Adapted from Wikipedia
        @inbounds p1 = Afull[4]*Afull[4] + Afull[7]*Afull[7] + Afull[8]*Afull[8]
        if (p1 == 0)
            # Afull is diagonal.
            @inbounds eig1 = Afull[1]
            @inbounds eig2 = Afull[5]
            @inbounds eig3 = Afull[9]

            return (SVector{3,$S}(eig1, eig2, eig3), eye(SMatrix{3,3,$S}))
        else
            q = trace(Afull)/3
            @inbounds p2 = (Afull[1] - q)^2 + (Afull[5] - q)^2 + (Afull[9] - q)^2 + 2 * p1
            p = sqrt(p2 / 6)
            B = (1 / p) * (Afull - UniformScaling(q)) # q*I
            r = det(B) / 2

            # In exact arithmetic for a symmetric matrix  -1 <= r <= 1
            # but computation error can leave it slightly outside this range.
            if (r <= -1) # TODO what type should phi be?
                phi = pi / 3
            elseif (r >= 1)
                phi = 0.0
            else
                phi = acos(r) / 3
            end

            # the eigenvalues satisfy eig1 <= eig2 <= eig3
            eig3 = q + 2 * p * cos(phi)
            eig1 = q + 2 * p * cos(phi + (2*pi/3))
            eig2 = 3 * q - eig1 - eig3     # since trace(Afull) = eig1 + eig2 + eig3

            # Now get the eigenvectors
            # TODO branch for when eig1 == eig2?
            @inbounds tmp1 = SVector(Afull[1] - eig1, Afull[2], Afull[3])
            @inbounds tmp2 = SVector(Afull[4], Afull[5] - eig1, Afull[6])
            v1 = cross(tmp1, tmp2)
            v1 = v1 / vecnorm(v1)

            @inbounds tmp1 = SVector(Afull[1] - eig2, Afull[2], Afull[3])
            @inbounds tmp2 = SVector(Afull[4], Afull[5] - eig2, Afull[6])
            v2 = cross(tmp1, tmp2)
            v2 = v2 / vecnorm(v2)

            v3 = cross(v1, v2) # should be normalized already

            @inbounds return (SVector((eig1, eig2, eig3)), SMatrix{3,3}((v1[1], v1[2], v1[3], v2[1], v2[2], v2[3], v3[1], v3[2], v3[3])))
        end
    end
end
