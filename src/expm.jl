@inline expm(A::StaticMatrix) = _expm(Size(A), A)

@inline function _expm(::Size{(1,1)}, A::StaticMatrix)
    T = typeof(exp(zero(eltype(A))))
    newtype = similar_type(A,T)

    (newtype)((exp(A[1]), ))
end

@inline function _expm(::Size{(2,2)}, A::StaticMatrix{<:Any,<:Any,<:Real})
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

@inline function _expm(::Size{(2,2)}, A::StaticMatrix{<:Any,<:Any,<:Complex})
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

# TODO add special case for 3x3 matrices

@inline _expm(s::Size, A::StaticArray) = s(Base.expm(Array(A)))
