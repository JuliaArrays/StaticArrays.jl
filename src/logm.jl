@inline log(A::StaticMatrix) = _log(Size(A), A)

@inline function _log(::Size{(1,1)}, A::StaticMatrix)
    T = typeof(log(zero(eltype(A))))
    newtype = similar_type(A,T)

    (newtype)((log(A[1]), ))
end

# Adapted from implementation in LinearAlgebra;
# if `A` is symmetric or Hermitian, its eigendecomposition is
# used, if `A` is triangular an improved version of the inverse scaling and squaring method is
# employed (see [^AH12] and [^AHR13]). For general matrices, the complex Schur form
# is computed and the triangular algorithm is used on the
# triangular factor.
# [^AH12]: Awad H. Al-Mohy and Nicholas J. Higham, "Improved inverse  scaling and squaring algorithms for the matrix logarithm", SIAM Journal on Scientific Computing, 34(4), 2012, C153-C169. [doi:10.1137/110852553](https://doi.org/10.1137/110852553)
# [^AHR13]: Awad H. Al-Mohy, Nicholas J. Higham and Samuel D. Relton, "Computing the Fréchet derivative of the matrix logarithm and estimating the condition number", SIAM Journal on Scientific Computing, 35(4), 2013, C394-C410. [doi:10.1137/120885991](https://doi.org/10.1137/120885991)
function _log(s::Size, _A::StaticMatrix{<:Any,<:Any,T}) where T

    S = typeof((zero(T)*zero(T) + zero(T)*zero(T))/one(T))
    A = S.(_A)
    # If possible, use diagonalization
    if ishermitian(A)
        # TODO: add proper static hermitian matrices
        logHermA = s(log(Array(Hermitian(A))))
    end

    # Use Schur decomposition
    if istriu(A)
        return triu!(parent(_log_UT(complex(A))))
    else
        if isreal(A)
            SchurF = schur(real(A))
        else
            SchurF = schur(A)
        end
        if !istriu(SchurF.T)
            SchurS = schur(complex(SchurF.T))
            logT = SchurS.Z * _log_UT(SchurS.T) * SchurS.Z'
            return SchurF.Z * logT * SchurF.Z'
        else
            R = _log_UT(complex(SchurF.T))
            return SchurF.Z * R * SchurF.Z'
        end
    end
end

# logarithm of an upper triangular matrix
# adapted from Julia base
# Complex matrix logarithm for the upper triangular factor, see:
#   Al-Mohy and Higham, "Improved inverse  scaling and squaring algorithms for
#     the matrix logarithm", SIAM J. Sci. Comput., 34(4), (2012), pp. C153–C169.
#   Al-Mohy, Higham and Relton, "Computing the Frechet derivative of the matrix
#     logarithm and estimating the condition number", SIAM J. Sci. Comput.,
#     35(4), (2013), C394–C410.
#
# Based on the code available at http://eprints.ma.man.ac.uk/1851/02/logm.zip,
# Copyright (c) 2011, Awad H. Al-Mohy and Nicholas J. Higham
# Julia version relicensed with permission from original authors
function _log_UT(A0::MMatrix{N,N,T}) where {N,T}
    maxsqrt = 100
    theta = @SVector [1.586970738772063e-005,
         2.313807884242979e-003,
         1.938179313533253e-002,
         6.209171588994762e-002,
         1.276404810806775e-001,
         2.060962623452836e-001,
         2.879093714241194e-001]
    tmax = size(theta, 1)
    A = complex(A0) #was Array()
    p = 0
    m = 0

    # Compute repeated roots
    d = complex(MVector{N}(diag(A)))
    dm1 = MVector{N}(d .- 1)
    s = 0
    while norm(dm1, Inf) > theta[tmax] && s < maxsqrt
        d .= sqrt.(d)
        dm1 .= d .- 1
        s = s + 1
    end
    s0 = s
    for k = 1:min(s, maxsqrt)
        A = MMatrix{N,N}(sqrt(A))
    end

    AmI = A - I
    d2 = sqrt(opnorm(AmI^2, 1))
    d3 = cbrt(opnorm(AmI^3, 1))
    alpha2 = max(d2, d3)
    foundm = false
    if alpha2 <= theta[2]
        m = alpha2 <= theta[1] ? 1 : 2
        foundm = true
    end

    while !foundm
        more = false
        if s > s0
            d3 = cbrt(opnorm(AmI^3, 1))
        end
        d4 = opnorm(AmI^4, 1)^(1/4)
        alpha3 = max(d3, d4)
        if alpha3 <= theta[tmax]
            local j
            for outer j = 3:tmax
                if alpha3 <= theta[j]
                    break
                end
            end
            if j <= 6
                m = j
                break
            elseif alpha3 / 2 <= theta[5] && p < 2
                more = true
                p = p + 1
           end
        end

        if !more
            d5 = opnorm(AmI^5, 1)^(1/5)
            alpha4 = max(d4, d5)
            eta = min(alpha3, alpha4)
            if eta <= theta[tmax]
                j = 0
                for outer j = 6:tmax
                    if eta <= theta[j]
                        m = j
                        break
                    end
                end
                break
            end
        end

        if s == maxsqrt
            m = tmax
            break
        end
        A = MMatrix{N,N}(sqrt(A))
        AmI = A - I
        s = s + 1
    end

    # Compute accurate superdiagonal of T
    p = 1 / 2^s
    for k = 1:N-1
        Ak = A0[k,k]
        Akp1 = A0[k+1,k+1]
        Akp = Ak^p
        Akp1p = Akp1^p
        A[k,k] = Akp
        A[k+1,k+1] = Akp1p
        if Ak == Akp1
            A[k,k+1] = p * A0[k,k+1] * Ak^(p-1)
        elseif 2 * abs(Ak) < abs(Akp1) || 2 * abs(Akp1) < abs(Ak)
            A[k,k+1] = A0[k,k+1] * (Akp1p - Akp) / (Akp1 - Ak)
        else
            logAk = log(Ak)
            logAkp1 = log(Akp1)
            w = atanh((Akp1 - Ak)/(Akp1 + Ak)) + im*pi*ceil((imag(logAkp1-logAk)-pi)/(2*pi))
            dd = 2 * exp(p*(logAk+logAkp1)/2) * sinh(p*w) / (Akp1 - Ak)
            A[k,k+1] = A0[k,k+1] * dd
        end
    end

    # Compute accurate diagonal of T
    for i = 1:N
        a = A0[i,i]
        if s == 0
            r = a - 1
        end
        s0 = s
        if angle(a) >= pi / 2
            a = sqrt(a)
            s0 = s - 1
        end
        z0 = a - 1
        a = sqrt(a)
        r = 1 + a
        for j = 1:s0-1
            a = sqrt(a)
            r = r * (1 + a)
        end
        A[i,i] = z0 / r
    end

    # Get the Gauss-Legendre quadrature points and weights
    R = zeros(T, m, m)
    for i = 1:m - 1
        R[i,i+1] = T(i / sqrt((2 * i)^2 - 1))
        R[i+1,i] = R[i,i+1]
    end
    x,V = eigen(R)
    w = Vector{Float64}(undef, m)
    for i = 1:m
        x[i] = (x[i] + 1) / 2
        w[i] = V[1,i]^2
    end

    # Compute the Padé approximation
    Y = zeros(MMatrix{N,N,T})
    for k = 1:m
        Y .= Y .+ (w[k] * (A / (x[k] * A + I)))
    end

    # Scale back
    lmul!(2^s, Y)

    # Compute accurate diagonal and superdiagonal of log(T)
    for k = 1:N-1
        Ak = A0[k,k]
        Akp1 = A0[k+1,k+1]
        logAk = log(Ak)
        logAkp1 = log(Akp1)
        Y[k,k] = logAk
        Y[k+1,k+1] = logAkp1
        if Ak == Akp1
            Y[k,k+1] = A0[k,k+1] / Ak
        elseif 2 * abs(Ak) < abs(Akp1) || 2 * abs(Akp1) < abs(Ak)
            Y[k,k+1] = A0[k,k+1] * (logAkp1 - logAk) / (Akp1 - Ak)
        else
            w = atanh((Akp1 - Ak)/(Akp1 + Ak) + im*pi*(ceil((imag(logAkp1-logAk) - pi)/(2*pi))))
            Y[k,k+1] = 2 * A0[k,k+1] * w / (Akp1 - Ak)
        end
    end

    return Y
end
