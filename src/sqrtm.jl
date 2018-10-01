
@inline sqrt(A::StaticMatrix) = _sqrt(Size(A),A)

@inline function _sqrt(::Size{(1,1)}, A::SA) where {SA<:StaticArray}
    s = sqrt(A[1,1])
    similar_type(SA,typeof(s))(s)
end

@inline function _sqrt(::Size{(2,2)}, A::SA) where {SA<:StaticArray}
    a,b,c,d = A
    if a==b==c==d==0
        zero(A)
    else
        determinant = a*d-b*c
        if isreal(determinant) && real(determinant) ≥ 0
            s = sqrt(determinant)
            t = inv(sqrt(a+d+2s))
            similar_type(SA,typeof(t))(t*(a+s), t*b, t*c, t*(d+s))
        else
            s = sqrt(complex(determinant))
            t = inv(sqrt(a+d+2s))
            similar_type(SA,typeof(t))(t*(a+s), t*b, t*c, t*(d+s))
        end
    end
end

#@inline _sqrt(s::Size, A::StaticArray) = s(sqrt(Array(A)))

function _sqrt(s::Size, A::StaticArray)
    TEl = typeof(exp(zero(eltype(A))))
    if issymmetric(A)
        # TODO: a better implementation once symmetric matrices are here
        return s(sqrt(Array(A)))
    end
    AT = MMatrix{s[1], s[2], TEl}(A)
    if istriu(A)
        return _sqrt_UT(AT)
    else
        SchurF = schur(complex(AT))
        R = _sqrt_UT(SchurF.T)
        return SchurF.Z * R * SchurF.Z'
    end
end

# square root of an upper triangular matrix `A`
# TODO: make it work with `A` wrapped in `UpperTriangular`
function _sqrt_UT(A::StaticMatrix{N,N,T}) where {N,T}
    realmatrix = false
    if isreal(A)
        realmatrix = true
        for i = 1:N
            @inbounds x = real(A[i,i])
            if x < zero(x)
                realmatrix = false
                break
            end
        end
    end

    # for some reason, this is faster than `_sqrt_UT(A, Val(realmatrix))` on Julia 1.0.0
    if realmatrix
        return _sqrt_UT(A, Val(true))
    else
        return _sqrt_UT(A, Val(false))
    end
end

@generated function _sqrt_UT(A::StaticMatrix{N,N,T},::Val{realmatrix}) where {N,T,realmatrix}
    t = realmatrix ? typeof(sqrt(zero(T))) : typeof(sqrt(complex(zero(T))))
    tt = typeof(zero(t)*zero(t))
    quote
        R = zeros(MMatrix{N,N,$t})
        for j = 1:N
            @inbounds R[j,j] = $(realmatrix ? :(sqrt(A[j,j])) : :(sqrt(complex(A[j,j]))))
            for i = j-1:-1:1
                @inbounds r::$tt = A[i,j]
                @simd for k = i+1:j-1
                    @inbounds r -= R[i,k]*R[k,j]
                end
                @inbounds iszero(r) || (R[i,j] = sylvester(R[i,i],R[j,j],-r))
            end
        end
        return R
    end
end
