@inline exp(A::StaticMatrix) = _exp(Size(A), A)

@inline function _exp(::Size{(0,0)}, A::StaticMatrix)
    T = typeof(exp(zero(eltype(A))))
    newtype = similar_type(A,T)

    (newtype)()
end

@inline function _exp(::Size{(1,1)}, A::StaticMatrix)
    T = typeof(exp(zero(eltype(A))))
    newtype = similar_type(A,T)

    (newtype)((exp(A[1]), ))
end

# Bernstein, D. S. & So, W. 1993. "Some Explicit Formulas for the Matrix Exponential"
@inline function _exp(::Size{(2,2)}, A::StaticMatrix{<:Any,<:Any,<:Real})
    T = typeof(exp(zero(eltype(A))))
    newtype = similar_type(A,T)

    @inbounds a = A[1]
    @inbounds c = A[2]
    @inbounds b = A[3]
    @inbounds d = A[4]

    v = (a-d)^2 + 4*b*c

    m = (a + d) / 2

    if v > 0
        # In this case the formulas in the entries of the matrix
        # are a function of cosh and sinh, and could (in theory)
        # follow the same code pattern of the other branches (v ≤ 0).
        # However, cosh and sinh explode with large arguments and
        # we use the following identity to avoid numerical issues:
        #
        # exp(m) * [c₁ * cohs(δ) + c₂ * sinh(δ)] =
        #   c₁ * (e₊ + e₋) / 2 + c₂ * (e₊ - e₋) / 2
        #
        # where e₊ = exp(m + δ) and e₋ = exp(m - δ).
        #
        # See https://github.com/JuliaArrays/StaticArrays.jl/issues/1295
        δ = sqrt(v) / 2
        e₊ = exp(m + δ)
        e₋ = exp(m - δ)
        e₁ = (e₊ + e₋) / 2
        e₂ = (e₊ - e₋) / 2
        c₂ = (a - d) / 2δ
        m11 = (e₁ + c₂ * e₂)
        m12 = (b / δ) * e₂
        m21 = (c / δ) * e₂
        m22 = (e₁ - c₂ * e₂)
    elseif v < 0
        z = sqrt(-v)
        r = exp(m)
        z1 = cos(z / 2)
        z2 = sin(z / 2) / z
        m11 = r * (z1 + (a - d) * z2)
        m12 = r * 2b * z2
        m21 = r * 2c * z2
        m22 = r * (z1 - (a - d) * z2)
    else # if v == 0
        r = exp(m)
        z1 = T(1.0)
        z2 = T(0.5)
        m11 = r * (z1 + (a - d) * z2)
        m12 = r * 2b * z2
        m21 = r * 2c * z2
        m22 = r * (z1 - (a - d) * z2)
    end

    (newtype)((m11, m21, m12, m22))
end

@inline function _exp(::Size{(2,2)}, A::StaticMatrix{<:Any,<:Any,<:Complex})
    T = typeof(exp(zero(eltype(A))))
    newtype = similar_type(A,T)

    @inbounds a = A[1]
    @inbounds c = A[2]
    @inbounds b = A[3]
    @inbounds d = A[4]

    z = sqrt((a - d)*(a - d) + 4*b*c )
    e = expm1((a + d - z) / 2)
    f = expm1((a + d + z) / 2)
    ϵ = eps()
    g = abs2(z) < ϵ^2 ? exp((a + d) / 2) * (1 + z^2 / 24) : (f - e) / z

    m11 = (g * (a - d) + f + e) / 2 + 1
    m12 = g * b
    m21 = g * c
    m22 = (-g * (a - d) + f + e) / 2 + 1

    (newtype)((m11, m21, m12, m22))
end

# Adapted from implementation in Base; algorithm from
# Higham, "Functions of Matrices: Theory and Computation", SIAM, 2008
function _exp(::Size, _A::StaticMatrix{<:Any,<:Any,T}) where T
    S = typeof((zero(T)*zero(T) + zero(T)*zero(T))/one(T))
    A = S.(_A)
    # omitted: matrix balancing, i.e., LAPACK.gebal!
    nA = maximum(sum(abs.(A); dims=Val(1)))    # marginally more performant than norm(A, 1)
    ## For sufficiently small nA, use lower order Padé-Approximations
    if (nA <= 2.1)
        A2 = A*A
        if nA > 0.95
            U = @evalpoly(A2, S(8821612800)*I, S(302702400)*I, S(2162160)*I, S(3960)*I, S(1)*I)
            U = A*U
            V = @evalpoly(A2, S(17643225600)*I, S(2075673600)*I, S(30270240)*I, S(110880)*I, S(90)*I)
        elseif nA > 0.25
            U = @evalpoly(A2, S(8648640)*I, S(277200)*I, S(1512)*I, S(1)*I)
            U = A*U
            V = @evalpoly(A2, S(17297280)*I, S(1995840)*I, S(25200)*I, S(56)*I)
        elseif nA > 0.015
            U = @evalpoly(A2, S(15120)*I, S(420)*I, S(1)*I)
            U = A*U
            V = @evalpoly(A2, S(30240)*I, S(3360)*I, S(30)*I)
        else
            U = @evalpoly(A2, S(60)*I, S(1)*I)
            U = A*U
            V = @evalpoly(A2, S(120)*I, S(12)*I)
        end
        expA = (V - U) \ (V + U)
    else
        s  = log2(nA/5.4)               # power of 2 later reversed by squaring
        if s > 0
            si = ceil(Int,s)
            A = A / S(2^si)
        end

        A2 = A*A
        A4 = A2*A2
        A6 = A2*A4

        U = A6*(S(1)*A6 + S(16380)*A4 + S(40840800)*A2) +
            (S(33522128640)*A6 + S(10559470521600)*A4 + S(1187353796428800)*A2) +
            S(32382376266240000)*I
        U = A*U
        V = A6*(S(182)*A6 + S(960960)*A4 + S(1323241920)*A2) +
            (S(670442572800)*A6 + S(129060195264000)*A4 + S(7771770303897600)*A2) +
            S(64764752532480000)*I
        expA = (V - U) \ (V + U)

        if s > 0            # squaring to reverse dividing by power of 2
            for t=1:si
                expA = expA*expA
            end
        end
    end

    expA
end
