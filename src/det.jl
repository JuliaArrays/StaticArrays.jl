@inline function det(A::StaticMatrix)
    T = eltype(A)
    S = arithmetic_closure(T)
    A_S = convert(similar_type(A,S),A)
    _det(Size(A_S),A_S)
end

@inline _det(::Size{(1,1)}, A::StaticMatrix) = @inbounds return A[1]

@inline function _det(::Size{(2,2)}, A::StaticMatrix)
    @inbounds return A[1]*A[4] - A[3]*A[2]
end

@inline function _det(::Size{(3,3)}, A::StaticMatrix)
    @inbounds x0 = SVector(A[1], A[2], A[3])
    @inbounds x1 = SVector(A[4], A[5], A[6])
    @inbounds x2 = SVector(A[7], A[8], A[9])
    return bilinear_vecdot(x0, cross(x1, x2))
end

@inline function _det(::Size{(4,4)}, A::StaticMatrix)
    @inbounds return (
        A[13] * A[10]  * A[7]  * A[4]  - A[9] * A[14] * A[7]  * A[4]   -
        A[13] * A[6]   * A[11] * A[4]  + A[5] * A[14] * A[11] * A[4]   +
        A[9]  * A[6]   * A[15] * A[4]  - A[5] * A[10] * A[15] * A[4]   -
        A[13] * A[10]  * A[3]  * A[8]  + A[9] * A[14] * A[3]  * A[8]   +
        A[13] * A[2]   * A[11] * A[8]  - A[1] * A[14] * A[11] * A[8]   -
        A[9]  * A[2]   * A[15] * A[8]  + A[1] * A[10] * A[15] * A[8]   +
        A[13] * A[6]   * A[3]  * A[12] - A[5] * A[14] * A[3]  * A[12]  -
        A[13] * A[2]   * A[7]  * A[12] + A[1] * A[14] * A[7]  * A[12]  +
        A[5]  * A[2]   * A[15] * A[12] - A[1] * A[6]  * A[15] * A[12]  -
        A[9]  * A[6]   * A[3]  * A[16] + A[5] * A[10] * A[3]  * A[16]  +
        A[9]  * A[2]   * A[7]  * A[16] - A[1] * A[10] * A[7]  * A[16]  -
        A[5]  * A[2]   * A[11] * A[16] + A[1] * A[6]  * A[11] * A[16])
end

@inline function _parity(p)    # inefficient compared to computing cycle lengths, but non-allocating
    s = 0
    for i in 1:length(p), j in i+1:length(p)
        s += p[i] > p[j]
    end
    -2*rem(s, 2) + 1
end

det(F::LU) = det(F.U) * _parity(F.p)

function logabsdet(A::Union{LowerTriangular{<:Any,<:StaticMatrix},
                            UpperTriangular{<:Any,<:StaticMatrix}})
    checksquare(A)
    mapreduce(x -> (log(abs(x)), sign(x)), ((l1, s1), (l2, s2)) -> (l1 + l2, s1 * s2),
              diag(A))
end

function logdet(F::LU)
    d, s = logabsdet(F.U)
    d + log(s * _parity(F.p))
end

function logabsdet(F::LU)
    d, s = logabsdet(F.U)
    d, s * _parity(F.p)
end

@generated function _det(::Size{S}, A::StaticMatrix) where S
    checksquare(A)
    if prod(S) ≤ 14*14
        quote
            @_inline_meta
            det(lu(A, Val(true); check = false))
        end
    else
        :(@_inline_meta; det(Matrix(A)))
    end
end

@inline logdet(A::StaticMatrix) = _logdet(Size(A), A)
@inline _logdet(::Union{Size{(1,1)}, Size{(2,2)}, Size{(3,3)}, Size{(4,4)}}, A::StaticMatrix) = log(det(A))
@generated function _logdet(::Size{S}, A::StaticMatrix) where S
    checksquare(A)
    if prod(S) ≤ 14*14
        quote
            @_inline_meta
            logdet(lu(A, Val(true); check = false))
        end
    else
        :(@_inline_meta; logdet(drop_sdims(A)))
    end
end

@inline logabsdet(A::StaticMatrix) = _logabsdet(Size(A), A)
@generated function _logabsdet(::Size{S}, A::StaticMatrix) where S
    checksquare(A)
    if prod(S) ≤ 14*14
        quote
            @_inline_meta
            logabsdet(lu(A, Val(true); check = false))
        end
    else
        :(@_inline_meta; logabsdet(drop_sdims(A)))
    end
end
