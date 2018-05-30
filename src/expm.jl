if VERSION < v"0.7-"
    @inline expm(A::StaticMatrix) = _exp(Size(A), A)
else
    @inline exp(A::StaticMatrix) = _exp(Size(A), A)
end

@inline function _exp(::Size{(1,1)}, A::StaticMatrix)
    T = typeof(exp(zero(eltype(A))))
    newtype = similar_type(A,T)

    (newtype)((exp(A[1]), ))
end

@inline function _exp(::Size{(2,2)}, A::StaticMatrix{<:Any,<:Any,<:Real})
    T = typeof(exp(zero(eltype(A))))
    newtype = similar_type(A,T)

    @inbounds a = A[1]
    @inbounds c = A[2]
    @inbounds b = A[3]
    @inbounds d = A[4]

    v = (a-d)^2 + 4*b*c

    if v > 0
        z = sqrt(v)
        z1 = cosh(z / 2)
        z2 = sinh(z / 2) / z
    elseif v < 0
        z = sqrt(-v)
        z1 = cos(z / 2)
        z2 = sin(z / 2) / z
    else # if v == 0
        z1 = T(1.0)
        z2 = T(0.5)
    end

    r = exp((a + d) / 2)
    m11 = r * (z1 + (a - d) * z2)
    m12 = r * 2b * z2
    m21 = r * 2c * z2
    m22 = r * (z1 - (a - d) * z2)

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
    e = exp((a + d - z)/2)
    f = exp((a + d + z)/2)
    zr = inv(z)

    m11 = (-e*(a - d - z) + f*(a - d + z)) * zr/2  
    m12 = (f-e) * b * zr
    m21 = (f-e) * c * zr
    m22 = (-e*(-a + d - z) + f*(-a + d + z)) * zr/2

    (newtype)((m11, m21, m12, m22))
end

# Adapted from implementation in Base; algorithm from
# Higham, "Functions of Matrices: Theory and Computation", SIAM, 2008
function _exp(::Size, A::StaticMatrix{<:Any,<:Any,T}) where T
    # omitted: matrix balancing, i.e., LAPACK.gebal!
    nA = maximum(sum(abs.(A), Val{1}))    # marginally more performant than norm(A, 1)
    ## For sufficiently small nA, use lower order Padé-Approximations
    if (nA <= 2.1)
        A2 = A*A
        if nA > 0.95
            U = @evalpoly(A2, T(8821612800)*I, T(302702400)*I, T(2162160)*I, T(3960)*I, T(1)*I)
            U = A*U
            V = @evalpoly(A2, T(17643225600)*I, T(2075673600)*I, T(30270240)*I, T(110880)*I, T(90)*I)
        elseif nA > 0.25
            U = @evalpoly(A2, T(8648640)*I, T(277200)*I, T(1512)*I, T(1)*I)
            U = A*U
            V = @evalpoly(A2, T(17297280)*I, T(1995840)*I, T(25200)*I, T(56)*I)
        elseif nA > 0.015
            U = @evalpoly(A2, T(15120)*I, T(420)*I, T(1)*I)
            U = A*U
            V = @evalpoly(A2, T(30240)*I, T(3360)*I, T(30)*I)
        else
            U = @evalpoly(A2, T(60)*I, T(1)*I)
            U = A*U
            V = @evalpoly(A2, T(120)*I, T(12)*I)
        end
        expA = (V - U) \ (V + U)
    else
        s  = log2(nA/5.4)               # power of 2 later reversed by squaring
        if s > 0
            si = ceil(Int,s)
            A = A / T(2^si)
        end

        A2 = A*A
        A4 = A2*A2
        A6 = A2*A4

        U = A6*(T(1)*A6 + T(16380)*A4 + T(40840800)*A2) +
            (T(33522128640)*A6 + T(10559470521600)*A4 + T(1187353796428800)*A2) +
            T(32382376266240000)*I
        U = A*U
        V = A6*(T(182)*A6 + T(960960)*A4 + T(1323241920)*A2) +
            (T(670442572800)*A6 + T(129060195264000)*A4 + T(7771770303897600)*A2) +
            T(64764752532480000)*I
        expA = (V - U) \ (V + U)

        if s > 0            # squaring to reverse dividing by power of 2
            for t=1:si
                expA = expA*expA
            end
        end
    end

    expA
end
