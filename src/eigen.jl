
@inline function eigvals(a::LinearAlgebra.RealHermSymComplexHerm{T,SA},; permute::Bool=true, scale::Bool=true) where {T <: Real, SA <: StaticArray}
    _eigvals(Size(SA), a, permute, scale)
end

@inline function eigvals(a::StaticArray; permute::Bool=true, scale::Bool=true)
    if ishermitian(a)
        _eigvals(Size(a), Hermitian(a), permute, scale)
    elseif Size(a) == Size{(1,1)}()
        _eigvals(Size(a), a, permute, scale)
    else
        error("Only hermitian matrices are diagonalizable by *StaticArrays*. Non-Hermitian matrices should be converted to `Array` first.")
    end
end

@inline _eigvals(::Size{(1,1)}, a, permute, scale) = @inbounds return SVector(Tuple(a))
@inline _eigvals(::Size{(1, 1)}, a::LinearAlgebra.RealHermSymComplexHerm{T}, permute, scale) where {T <: Real} = @inbounds return SVector(real(parent(a).data[1]))

@inline function _eigvals(::Size{(2,2)}, A::LinearAlgebra.RealHermSymComplexHerm{T}, permute, scale) where {T <: Real}
    a = A.data

    if A.uplo == 'U'
        @inbounds t_half = real(a[1] + a[4])/2
        @inbounds d = real(a[1]*a[4] - a[3]'*a[3]) # Should be real

        tmp2 = t_half*t_half - d
        tmp2 < 0 ? tmp = zero(tmp2) : tmp = sqrt(tmp2) # Numerically stable for identity matrices, etc.
        return SVector(t_half - tmp, t_half + tmp)
    else
        @inbounds t_half = real(a[1] + a[4])/2
        @inbounds d = real(a[1]*a[4] - a[2]'*a[2]) # Should be real

        tmp2 = t_half*t_half - d
        tmp2 < 0 ? tmp = zero(tmp2) : tmp = sqrt(tmp2) # Numerically stable for identity matrices, etc.
        return SVector(t_half - tmp, t_half + tmp)
    end
end

@inline function _eigvals(::Size{(3,3)}, A::LinearAlgebra.RealHermSymComplexHerm{T}, permute, scale) where {T <: Real}
    S = arithmetic_closure(T)
    Sreal = real(S)

    @inbounds a11 = convert(Sreal, A.data[1])
    @inbounds a22 = convert(Sreal, A.data[5])
    @inbounds a33 = convert(Sreal, A.data[9])
    if A.uplo == 'U'
        @inbounds a12 = convert(S, A.data[4])
        @inbounds a13 = convert(S, A.data[7])
        @inbounds a23 = convert(S, A.data[8])
    else
        @inbounds a12 = conj(convert(S, A.data[2]))
        @inbounds a13 = conj(convert(S, A.data[3]))
        @inbounds a23 = conj(convert(S, A.data[6]))
    end

    p1 = abs2(a12) + abs2(a13) + abs2(a23)
    if (p1 == 0)
        # Matrix is diagonal
        if a11 < a22
            if a22 < a33
                return SVector(a11, a22, a33)
            elseif a33 < a11
                return SVector(a33, a11, a22)
            else
                return SVector(a11, a33, a22)
            end
        else #a22 < a11
            if a11 < a33
                return SVector(a22, a11, a33)
            elseif a33 < a22
                return SVector(a33, a22, a11)
            else
                return SVector(a22, a33, a11)
            end
        end
    end

    q = (a11 + a22 + a33) / 3
    p2 = abs2(a11 - q) + abs2(a22 - q) + abs2(a33 - q) + 2 * p1
    p = sqrt(p2 / 6)
    invp = inv(p)
    b11 = (a11 - q) * invp
    b22 = (a22 - q) * invp
    b33 = (a33 - q) * invp
    b12 = a12 * invp
    b13 = a13 * invp
    b23 = a23 * invp
    B = SMatrix{3,3,S}((b11, conj(b12), conj(b13), b12, b22, conj(b23), b13, b23, b33))
    r = real(det(B)) / 2

    # In exact arithmetic for a symmetric matrix  -1 <= r <= 1
    # but computation error can leave it slightly outside this range.
    if (r <= -1)
        phi = Sreal(pi) / 3
    elseif (r >= 1)
        phi = zero(Sreal)
    else
        phi = acos(r) / 3
    end

    eig3 = q + 2 * p * cos(phi)
    eig1 = q + 2 * p * cos(phi + (2*Sreal(pi)/3))
    eig2 = 3 * q - eig1 - eig3     # since tr(A) = eig1 + eig2 + eig3

    return SVector(eig1, eig2, eig3)
end

@inline function _eigvals(s::Size, A::LinearAlgebra.RealHermSymComplexHerm{T}, permute, scale) where {T <: Real}
    vals = eigvals(Hermitian(Array(parent(A))))
    return SVector{s[1], T}(vals)
end

# Utility to rewrap `Eigen` of normal `Array` into an Eigen containing `SArray`.
@inline function _make_static(s::Size, E::Eigen{T,V}) where {T,V}
    Eigen(similar_type(SVector, V, Size(s[1]))(E.values),
          similar_type(SMatrix, T, s)(E.vectors))
end

@inline function _eig(s::Size, A::T, permute, scale) where {T <: StaticMatrix}
    if ishermitian(A)
        return _eig(s, Hermitian(A), permute, scale)
    else
        # For the non-hermitian branch fall back to LinearAlgebra eigen().
        # Eigenvalues could be real or complex so a Union of concrete types is
        # inferred. Having _make_static a separate function allows inference to
        # preserve the union of concrete types:
        #   Union{E{A,B},E{C,D}} -> Union{E{SA,SB},E{SC,SD}}
        _make_static(s, eigen(Array(A); permute = permute, scale = scale))
    end
end

@inline function _eig(s::Size, A::LinearAlgebra.RealHermSymComplexHerm{T}, permute, scale) where {T <: Real}
    E = eigen(Hermitian(Array(parent(A))))
    return Eigen(SVector{s[1], T}(E.values), SMatrix{s[1], s[2], eltype(A)}(E.vectors))
end


@inline function _eig(::Size{(1,1)}, A::LinearAlgebra.RealHermSymComplexHerm{T}, permute, scale) where {T <: Real}
    @inbounds return Eigen(SVector{1,T}((real(A[1]),)), SMatrix{1,1,eltype(A)}(I))
end

@inline function _eig(::Size{(2,2)}, A::LinearAlgebra.RealHermSymComplexHerm{T}, permute, scale) where {T <: Real}
    a = A.data
    TA = eltype(A)

    @inbounds if A.uplo == 'U'
        if !iszero(a[3]) # A is not diagonal
            t_half = real(a[1] + a[4]) / 2
            d = real(a[1] * a[4] - a[3]' * a[3]) # Should be real

            tmp2 = t_half * t_half - d
            tmp = tmp2 < 0 ? zero(tmp2) : sqrt(tmp2) # Numerically stable for identity matrices, etc.
            vals = SVector(t_half - tmp, t_half + tmp)

            v11 = vals[1] - a[4]
            n1 = sqrt(v11' * v11 + a[3]' * a[3])
            v11 = v11 / n1
            v12 = a[3]' / n1

            v21 = vals[2] - a[4]
            n2 = sqrt(v21' * v21 + a[3]' * a[3])
            v21 = v21 / n2
            v22 = a[3]' / n2

            vecs = @SMatrix [ v11  v21 ;
                              v12  v22 ]

            return Eigen(vals, vecs)
        end
    else # A.uplo == 'L'
        if !iszero(a[2]) # A is not diagonal
            t_half = real(a[1] + a[4]) / 2
            d = real(a[1] * a[4] - a[2]' * a[2]) # Should be real

            tmp2 = t_half * t_half - d
            tmp = tmp2 < 0 ? zero(tmp2) : sqrt(tmp2) # Numerically stable for identity matrices, etc.
            vals = SVector(t_half - tmp, t_half + tmp)

            v11 = vals[1] - a[4]
            n1 = sqrt(v11' * v11 + a[2]' * a[2])
            v11 = v11 / n1
            v12 = a[2] / n1

            v21 = vals[2] - a[4]
            n2 = sqrt(v21' * v21 + a[2]' * a[2])
            v21 = v21 / n2
            v22 = a[2] / n2

            vecs = @SMatrix [ v11  v21 ;
                              v12  v22 ]

            return Eigen(vals,vecs)
        end
    end

    # A must be diagonal if we reached this point; treatment of uplo 'L' and 'U' is then identical
    A11 = real(a[1])
    A22 = real(a[4])
    if A11 < A22
        vals = SVector(A11, A22)
        vecs = @SMatrix [convert(TA, 1) convert(TA, 0);
                         convert(TA, 0) convert(TA, 1)]
    else # A22 <= A11
        vals = SVector(A22, A11)
        vecs = @SMatrix [convert(TA, 0) convert(TA, 1);
                         convert(TA, 1) convert(TA, 0)]
    end
    return Eigen(vals,vecs)
end

# A small part of the code in the following method was inspired by works of David
# Eberly, Geometric Tools LLC, in code released under the Boost Software
# License (included at the end of this file).
# TODO extend the method to complex hermitian
@inline function _eig(::Size{(3,3)}, A::LinearAlgebra.HermOrSym{T}, permute, scale) where {T <: Real}
    S = arithmetic_closure(T)
    Sreal = real(S)

    @inbounds a11 = convert(Sreal, A.data[1])
    @inbounds a22 = convert(Sreal, A.data[5])
    @inbounds a33 = convert(Sreal, A.data[9])
    if A.uplo == 'U'
        @inbounds a12 = convert(S, A.data[4])
        @inbounds a13 = convert(S, A.data[7])
        @inbounds a23 = convert(S, A.data[8])
    else
        @inbounds a12 = conj(convert(S, A.data[2]))
        @inbounds a13 = conj(convert(S, A.data[3]))
        @inbounds a23 = conj(convert(S, A.data[6]))
    end

    p1 = abs2(a12) + abs2(a13) + abs2(a23)
    if (p1 == 0)
        # Matrix is diagonal
        v1 = SVector(one(S),  zero(S), zero(S))
        v2 = SVector(zero(S), one(S),  zero(S))
        v3 = SVector(zero(S), zero(S), one(S) )

        if a11 < a22
            if a22 < a33
                return Eigen(SVector((a11, a22, a33)), hcat(v1,v2,v3))
            elseif a33 < a11
                return Eigen(SVector((a33, a11, a22)), hcat(v3,v1,v2))
            else
                return Eigen(SVector((a11, a33, a22)), hcat(v1,v3,v2))
            end
        else #a22 < a11
            if a11 < a33
                return Eigen(SVector((a22, a11, a33)), hcat(v2,v1,v3))
            elseif a33 < a22
                return Eigen(SVector((a33, a22, a11)), hcat(v3,v2,v1))
            else
                return Eigen(SVector((a22, a33, a11)), hcat(v2,v3,v1))
            end
        end
    end

    q = (a11 + a22 + a33) / 3
    p2 = abs2(a11 - q) + abs2(a22 - q) + abs2(a33 - q) + 2 * p1
    p = sqrt(p2 / 6)
    invp = inv(p)
    b11 = (a11 - q) * invp
    b22 = (a22 - q) * invp
    b33 = (a33 - q) * invp
    b12 = a12 * invp
    b13 = a13 * invp
    b23 = a23 * invp
    B = SMatrix{3,3,S}((b11, conj(b12), conj(b13), b12, b22, conj(b23), b13, b23, b33))
    r = real(det(B)) / 2

    # In exact arithmetic for a symmetric matrix  -1 <= r <= 1
    # but computation error can leave it slightly outside this range.
    if (r <= -1)
        phi = Sreal(pi) / 3
    elseif (r >= 1)
        phi = zero(Sreal)
    else
        phi = acos(r) / 3
    end

    eig3 = q + 2 * p * cos(phi)
    eig1 = q + 2 * p * cos(phi + (2*Sreal(pi)/3))
    eig2 = 3 * q - eig1 - eig3     # since tr(A) = eig1 + eig2 + eig3

    if r > 0 # Helps with conditioning the eigenvector calculation
        (eig1, eig3) = (eig3, eig1)
    end

    # Calculate the first eigenvector
    # This should be orthogonal to these three rows of A - eig1*I
    # Use all combinations of cross products and choose the "best" one
    r₁ = SVector(a11 - eig1, a12, a13)
    r₂ = SVector(conj(a12), a22 - eig1, a23)
    r₃ = SVector(conj(a13), conj(a23), a33 - eig1)
    n₁ = sum(abs2, r₁)
    n₂ = sum(abs2, r₂)
    n₃ = sum(abs2, r₃)

    r₁₂ = r₁ × r₂
    r₂₃ = r₂ × r₃
    r₃₁ = r₃ × r₁
    n₁₂ = sum(abs2, r₁₂)
    n₂₃ = sum(abs2, r₂₃)
    n₃₁ = sum(abs2, r₃₁)

    # we want best angle so we put all norms on same footing
    # (cheaper to multiply by third nᵢ rather than divide by the two involved)
    if n₁₂ * n₃ > n₂₃ * n₁
        if n₁₂ * n₃ > n₃₁ * n₂
            eigvec1 = r₁₂ / sqrt(n₁₂)
        else
            eigvec1 = r₃₁ / sqrt(n₃₁)
        end
    else
        if n₂₃ * n₁ > n₃₁ * n₂
            eigvec1 = r₂₃ / sqrt(n₂₃)
        else
            eigvec1 = r₃₁ / sqrt(n₃₁)
        end
    end

    # Calculate the second eigenvector
    # This should be orthogonal to the previous eigenvector and the three
    # rows of A - eig2*I. However, we need to "solve" the remaining 2x2 subspace
    # problem in case the cross products are identically or nearly zero

    # The remaing 2x2 subspace is:
    @inbounds if abs(eigvec1[1]) < abs(eigvec1[2]) # safe to set one component to zero, depending on this
        orthogonal1 = SVector(-eigvec1[3], zero(S), eigvec1[1]) / sqrt(abs2(eigvec1[1]) + abs2(eigvec1[3]))
    else
        orthogonal1 = SVector(zero(S), eigvec1[3], -eigvec1[2]) / sqrt(abs2(eigvec1[2]) + abs2(eigvec1[3]))
    end
    orthogonal2 = eigvec1 × orthogonal1

    # The projected 2x2 eigenvalue problem is C x = 0 where C is the projection
    # of (A - eig2*I) onto the subspace {orthogonal1, orthogonal2}
    @inbounds a_orth1_1 = a11 * orthogonal1[1] + a12 * orthogonal1[2] + a13 * orthogonal1[3]
    @inbounds a_orth1_2 = conj(a12) * orthogonal1[1] + a22 * orthogonal1[2] + a23 * orthogonal1[3]
    @inbounds a_orth1_3 = conj(a13) * orthogonal1[1] + conj(a23) * orthogonal1[2] + a33 * orthogonal1[3]

    @inbounds a_orth2_1 = a11 * orthogonal2[1] + a12 * orthogonal2[2] + a13 * orthogonal2[3]
    @inbounds a_orth2_2 = conj(a12) * orthogonal2[1] + a22 * orthogonal2[2] + a23 * orthogonal2[3]
    @inbounds a_orth2_3 = conj(a13) * orthogonal2[1] + conj(a23) * orthogonal2[2] + a33 * orthogonal2[3]

    @inbounds c11 = conj(orthogonal1[1])*a_orth1_1 + conj(orthogonal1[2])*a_orth1_2 + conj(orthogonal1[3])*a_orth1_3 - eig2
    @inbounds c12 = conj(orthogonal1[1])*a_orth2_1 + conj(orthogonal1[2])*a_orth2_2 + conj(orthogonal1[3])*a_orth2_3
    @inbounds c22 = conj(orthogonal2[1])*a_orth2_1 + conj(orthogonal2[2])*a_orth2_2 + conj(orthogonal2[3])*a_orth2_3 - eig2

    # Solve this robustly (some values might be small or zero)
    c11² = abs2(c11)
    c12² = abs2(c12)
    c22² = abs2(c22)
    if c11² >= c22²
        if c11² > 0 || c12² > 0
            if c11² >= c12²
                tmp = c12 / c11 # TODO check for compex input
                p2 = inv(sqrt(1 + abs2(tmp)))
                p1 = tmp * p2
            else
                tmp = c11 / c12 # TODO check for compex input
                p1 = inv(sqrt(1 + abs2(tmp)))
                p2 = tmp * p1
            end
            eigvec2 = p1*orthogonal1 - p2*orthogonal2
        else # c11 == 0 && c12 == 0 && c22 == 0 (smaller than c11)
            eigvec2 = orthogonal1
        end
    else
        if c22² >= c12²
            tmp = c12 / c22 # TODO check for compex input
            p1 = inv(sqrt(1 + abs2(tmp)))
            p2 = tmp * p1
        else
            tmp = c22 / c12 # TODO check for compex input
            p2 = inv(sqrt(1 + abs2(tmp)))
            p1 = tmp * p2
        end
        eigvec2 = p1*orthogonal1 - p2*orthogonal2
    end

    # The third eigenvector is a simple cross product of the other two
    eigvec3 = eigvec1 × eigvec2 # should be normalized already

    # Sort them back to the original ordering, if necessary
    if r > 0
        (eig1, eig3) = (eig3, eig1)
        (eigvec1, eigvec3) = (eigvec3, eigvec1)
    end

    return Eigen(SVector(eig1, eig2, eig3), hcat(eigvec1, eigvec2, eigvec3))
end

@inline function eigen(A::StaticMatrix; permute::Bool=true, scale::Bool=true)
    _eig(Size(A), A, permute, scale)
end

# to avoid method ambiguity with LinearAlgebra
@inline eigen(A::Hermitian{<:Real,<:StaticMatrix}; kwargs...)    = _eigen(A; kwargs...)
@inline eigen(A::Hermitian{<:Complex,<:StaticMatrix}; kwargs...) = _eigen(A; kwargs...)
@inline eigen(A::Symmetric{<:Real,<:StaticMatrix}; kwargs...)    = _eigen(A; kwargs...)
@inline eigen(A::Symmetric{<:Complex,<:StaticMatrix}; kwargs...) = _eigen(A; kwargs...)

@inline function _eigen(A::LinearAlgebra.HermOrSym; permute::Bool=true, scale::Bool=true)
    _eig(Size(A), A, permute, scale)
end

# NOTE: The following Boost Software License applies to parts of the method:
#     _eig{T<:Real}(::Size{(3,3)}, A::LinearAlgebra.RealHermSymComplexHerm{T}, permute, scale)

#=
Boost Software License - Version 1.0 - August 17th, 2003



Permission is hereby granted, free of charge, to any person or organization
obtaining a copy of the software and accompanying documentation covered by
this license (the "Software") to use, reproduce, display, distribute,
execute, and transmit the Software, and to prepare derivative works of the
Software, and to permit third-parties to whom the Software is furnished to
do so, all subject to the following:

The copyright notices in the Software and this entire statement, including
the above license grant, this restriction and the following disclaimer,
must be included in all copies of the Software, in whole or in part, and
all derivative works of the Software, unless such copies or derivative
works are solely in the form of machine-executable object code generated by
a source language processor.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE, TITLE AND NON-INFRINGEMENT. IN NO EVENT
SHALL THE COPYRIGHT HOLDERS OR ANYONE DISTRIBUTING THE SOFTWARE BE LIABLE
FOR ANY DAMAGES OR OTHER LIABILITY, WHETHER IN CONTRACT, TORT OR OTHERWISE,
ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.
=#
