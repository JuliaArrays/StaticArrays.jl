@inline expm(A::StaticMatrix) = _expm(Size(A), A)

@generated function _expm(::Size{(1,1)}, A::StaticMatrix)
    @assert size(A) == (1,1)
    T = promote_type(typeof(sqrt(one(eltype(A)))), Float32)
    newtype = similar_type(A,T)

    quote
        $(Expr(:meta, :inline))
        ($newtype)((exp(A[1]), ))
    end
end

@generated function _expm{S<:Real}(::Size{(2,2)}, A::StaticMatrix{S})
    @assert size(A) == (2,2)
    T = promote_type(typeof(sqrt(one(eltype(A)))), Float32)
    newtype = similar_type(A,T)

    quote
        $(Expr(:meta, :inline))

        @inbounds a = A[1]
        @inbounds c = A[2]
        @inbounds b = A[3]
        @inbounds d = A[4]

        v = (a-d)^2 + 4*b*c

        # if v == 0
        z1::$T = 1.0
        z2::$T = 0.5

        if v > 0
          z = sqrt(v)
          z1 = cosh(z / 2)
          z2 = sinh(z / 2) / z
        elseif v < 0
          z = sqrt(-v)
          z1 = cos(z / 2)
          z2 = sin(z / 2) / z
        end

        r = exp((a + d) / 2)
        m11 = r * (z1 + (a - d) * z2)
        m12 = r * 2b * z2
        m21 = r * 2c * z2
        m22 = r * (z1 - (a - d) * z2)

        ($newtype)((m11, m21, m12, m22))
    end
end

# TODO add complex valued expm
# TODO add special case for 3x3 matrices

@generated function _expm(::Size, A::StaticArray)
  T = promote_type(typeof(sqrt(one(eltype(A)))), Float32)
  newtype = similar_type(A,T)

  quote
      $(Expr(:meta, :inline))
      ($newtype)(Base.expm(Array(A)))
  end
end
