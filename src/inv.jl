@inline function inv(A::StaticMatrix)
    T = eltype(A)
    S = arithmetic_closure(T)
    A_S = convert(similar_type(A,S),A)
    _inv(Size(A_S),A_S)
end

@inline _inv(::Size{(0,0)}, A) = similar_type(A,typeof(inv(one(eltype(A)))))()

@inline _inv(::Size{(1,1)}, A) = similar_type(A)(inv(A[1]))

@inline function _inv(::Size{(2,2)}, A)
    newtype = similar_type(A)
    idet = 1/det(A)
    @inbounds return newtype((A[4]*idet, -(A[2]*idet), -(A[3]*idet), A[1]*idet))
end

@inline function _inv(::Size{(3,3)}, A)
    newtype = similar_type(A)

    @inbounds x0 = SVector{3}(A[1], A[2], A[3])
    @inbounds x1 = SVector{3}(A[4], A[5], A[6])
    @inbounds x2 = SVector{3}(A[7], A[8], A[9])

    y0 = cross(x1,x2)
    d  = bilinear_vecdot(x0, y0)
    x0 = x0 / d
    y0 = y0 / d
    y1 = cross(x2,x0)
    y2 = cross(x0,x1)

    @inbounds return newtype((y0[1], y1[1], y2[1], y0[2], y1[2], y2[2], y0[3], y1[3], y2[3]))
end

@inline function _inv(::Size{(4,4)}, A)
    # https://www.geometrictools.com/Documentation/LaplaceExpansionTheorem.pdf

    s0 = A[1] * A[6]  - A[2]  * A[5]
    s1 = A[1] * A[10] - A[2]  * A[9]
    s2 = A[1] * A[14] - A[2]  * A[13]
    s3 = A[5] * A[10] - A[6]  * A[9]
    s4 = A[5] * A[14] - A[6]  * A[13]
    s5 = A[9] * A[14] - A[10] * A[13]

    c5 = A[11] * A[16]  - A[12] * A[15]
    c4 = A[7]  * A[16]  - A[8]  * A[15]
    c3 = A[7]  * A[12]  - A[8]  * A[11]
    c2 = A[3]  * A[16]  - A[4]  * A[15]
    c1 = A[3]  * A[12]  - A[4]  * A[11]
    c0 = A[3]  * A[8]   - A[4]  * A[7]

    invdet = 1 / (s0 * c5 - s1 * c4 + s2 * c3 + s3 * c2 - s4 * c1 + s5 * c0)

    B =  @SMatrix [
        ( A[6] * c5 - A[10] * c4 + A[14] * c3) * invdet
    	(-A[2] * c5 + A[10] * c2 - A[14] * c1) * invdet
        ( A[2] * c4 - A[6]  * c2 + A[14] * c0) * invdet
        (-A[2] * c3 + A[6]  * c1 - A[10] * c0) * invdet
        (-A[5] * c5 + A[9]  * c4 - A[13] * c3) * invdet
    	( A[1] * c5 - A[9]  * c2 + A[13] * c1) * invdet
    	(-A[1] * c4 + A[5]  * c2 - A[13] * c0) * invdet
    	( A[1] * c3 - A[5]  * c1 + A[9]  * c0) * invdet
    	( A[8] * s5 - A[12] * s4 + A[16] * s3) * invdet
    	(-A[4] * s5 + A[12] * s2 - A[16] * s1) * invdet
    	( A[4] * s4 - A[8]  * s2 + A[16] * s0) * invdet
    	(-A[4] * s3 + A[8]  * s1 - A[12] * s0) * invdet
        (-A[7] * s5 + A[11] * s4 - A[15] * s3) * invdet
        ( A[3] * s5 - A[11] * s2 + A[15] * s1) * invdet
        (-A[3] * s4 + A[7]  * s2 - A[15] * s0) * invdet
        ( A[3] * s3 - A[7]  * s1 + A[11] * s0) * invdet]
    return similar_type(A)(B)
end

@generated function _inv(::Size{S}, A) where S
    LinearAlgebra.checksquare(A)
    if prod(S) ≤ 14*14
        quote
            @_inline_meta
            inv(lu(A))
        end
    else
        :(@_inline_meta; similar_type(A)(inv(Matrix(A))))
    end
end

function inv(LUp::LU)
    if !(LUp.L isa LowerTriangular)
        checksquare(LUp.L)
        checksquare(LUp.U)
    end
    LUp.U \ (LUp.L \ typeof(parent(LUp.L))(I)[LUp.p,:])
end
