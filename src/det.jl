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
    return vecdot(x0, cross(x1, x2))
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

@inline function _det(::Size, A::StaticMatrix)
    return det(Matrix(A))
end

@inline logdet(a::StaticMatrix) = _logdet(Size(a), a)
@inline _logdet(::Union{Size{(1,1)}, Size{(2,2)}, Size{(3,3)}}, a::StaticMatrix) = log(det(a))
@inline _logdet(::Size, a::StaticMatrix) = logdet(drop_sdims(a))
