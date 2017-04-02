import Base: +, -, *, /, \

# TODO: more operators, like AbstractArray

# Unary ops
@inline -(a::StaticArray) = map(-, a)

# Binary ops
# Between arrays
@inline +(a::StaticArray, b::StaticArray) = map(+, a, b)
@inline +(a::AbstractArray, b::StaticArray) = map(+, a, b)
@inline +(a::StaticArray, b::AbstractArray) = map(+, a, b)

@inline -(a::StaticArray, b::StaticArray) = map(-, a, b)
@inline -(a::AbstractArray, b::StaticArray) = map(-, a, b)
@inline -(a::StaticArray, b::AbstractArray) = map(-, a, b)

# Scalar-array
@inline +(a::Number, b::StaticArray) = broadcast(+, a, b)
@inline +(a::StaticArray, b::Number) = broadcast(+, a, b)

@inline -(a::Number, b::StaticArray) = broadcast(-, a, b)
@inline -(a::StaticArray, b::Number) = broadcast(-, a, b)

@inline *(a::Number, b::StaticArray) = broadcast(*, a, b)
@inline *(a::StaticArray, b::Number) = broadcast(*, a, b)

@inline /(a::StaticArray, b::Number) = broadcast(/, a, b)
@inline \(a::Number, b::StaticArray) = broadcast(\, a, b)


# With UniformScaling
@inline +(a::StaticMatrix, b::UniformScaling) = _plus_uniform(Size(a), a, b.λ)
@inline +(a::UniformScaling, b::StaticMatrix) = _plus_uniform(Size(b), b, a.λ)
@inline -(a::StaticMatrix, b::UniformScaling) = _plus_uniform(Size(a), a, -b.λ)
@inline -(a::UniformScaling, b::StaticMatrix) = _plus_uniform(Size(b), -b, a.λ)

@generated function _plus_uniform(::Size{S}, a::StaticMatrix, λ) where {S}
    if S[1] != S[2]
        throw(DimensionMismatch("matrix is not square: dimensions are $S"))
    end
    n = S[1]
    exprs = [i == j ? :(a[$(sub2ind(size(a), i, j))] + λ) : :(a[$(sub2ind(size(a), i, j))]) for i = 1:n, j = 1:n]
    return quote
        $(Expr(:meta, :inline))
        @inbounds return similar_type(a, promote_type(eltype(a), typeof(λ)))(tuple($(exprs...)))
    end
end

@inline *(a::UniformScaling, b::Union{StaticMatrix,StaticVector}) = a.λ * b
@inline *(a::StaticMatrix, b::UniformScaling) = a * b.λ
@inline \(a::UniformScaling, b::Union{StaticMatrix,StaticVector}) = a.λ \ b
@inline /(a::StaticMatrix, b::UniformScaling) = a / b.λ


# Transpose, conjugate, etc
@inline conj(a::StaticArray) = map(conj, a)
@inline transpose(m::StaticMatrix) = _transpose(Size(m), m)
# note: transpose of StaticVector is a RowVector, handled by Base

@generated function _transpose(::Size{S}, m::StaticMatrix) where {S}
    Snew = (S[2], S[1])

    exprs = [:(m[$(sub2ind(S, j1, j2))]) for j2 = 1:S[2], j1 = 1:S[1]]

    return quote
        $(Expr(:meta, :inline))
        @inbounds return similar_type($m, Size($Snew))(tuple($(exprs...)))
    end
end

@inline ctranspose(m::StaticMatrix) = _transpose(Size(m), m)

@generated function _ctranspose(::Size{S}, m::StaticMatrix) where {S}
    Snew = (S[2], S[1])

    exprs = [:(conj(m[$(sub2ind(S, j1, j2))])) for j2 = 1:S[2], j1 = 1:S[1]]

    return quote
        $(Expr(:meta, :inline))
        @inbounds return similar_type($m, Size($Snew))(tuple($(exprs...)))
    end
end

@inline vcat(a::Union{StaticVector,StaticMatrix}) = a
@inline vcat(a::Union{StaticVector, StaticMatrix}, b::Union{StaticVector,StaticMatrix}) = _vcat(Size(a), Size(b), a, b)
@generated function _vcat(::Size{Sa}, ::Size{Sb}, a::Union{StaticVector, StaticMatrix}, b::Union{StaticVector,StaticMatrix}) where {Sa, Sb}
    if Size(Sa)[2] != Size(Sb)[2]
        throw(DimensionMismatch("Tried to vcat arrays of size $Sa and $Sb"))
    end

    # TODO cleanup?
    if a <: StaticVector && b <: StaticVector
        Snew = (Sa[1] + Sb[1],)
        exprs = vcat([:(a[$i]) for i = 1:Sa[1]],
                     [:(b[$i]) for i = 1:Sb[1]])
    else
        Snew = (Sa[1] + Sb[1], Size(Sa)[2])
        exprs = [((i <= size(a,1)) ? ((a <: StaticVector) ? :(a[$i]) : :(a[$i,$j]))
                                   : ((b <: StaticVector) ? :(b[$(i-size(a,1))]) : :(b[$(i-size(a,1)),$j])))
                                   for i = 1:(Sa[1]+Sb[1]), j = 1:Size(Sa)[2]]
    end

    return quote
        @_inline_meta
        @inbounds return similar_type(a, Size($Snew))(tuple($(exprs...)))
    end
end
# TODO make these more efficient
@inline vcat(a::Union{StaticVector,StaticMatrix}, b::Union{StaticVector,StaticMatrix}, c::Union{StaticVector,StaticMatrix}) =
    vcat(vcat(a,b), c)
@inline vcat(a::Union{StaticVector,StaticMatrix}, b::Union{StaticVector,StaticMatrix}, c::Union{StaticVector,StaticMatrix}...) =
    vcat(vcat(a,b), c...)


@inline hcat(a::StaticVector) = similar_type(a, Size(Size(a)[1],1))(a)
@inline hcat(a::StaticMatrix) = a
@inline hcat(a::Union{StaticVector,StaticMatrix}, b::Union{StaticVector,StaticMatrix}) = _hcat(Size(a), Size(b), a, b)

@generated function _hcat(::Size{Sa}, ::Size{Sb}, a::Union{StaticVector,StaticMatrix}, b::Union{StaticVector,StaticMatrix}) where {Sa, Sb}
    if Sa[1] != Sb[1]
        error("Dimension mismatch")
    end

    exprs = vcat([:(a[$i]) for i = 1:prod(Sa)],
                 [:(b[$i]) for i = 1:prod(Sb)])

    Snew = (Sa[1], Size(Sa)[2] + Size(Sb)[2])

    return quote
        @_inline_meta
        @inbounds return similar_type(a, Size($Snew))(tuple($(exprs...)))
    end
end
# TODO make these more efficient
@inline hcat(a::Union{StaticVector,StaticMatrix}, b::Union{StaticVector,StaticMatrix}, c::Union{StaticVector,StaticMatrix}) =
    hcat(hcat(a,b), c)
@inline hcat(a::Union{StaticVector,StaticMatrix}, b::Union{StaticVector,StaticMatrix}, c::Union{StaticVector,StaticMatrix}...) =
    hcat(hcat(a,b), c...)

@inline Base.zero{SA <: StaticArray}(a::SA) = zeros(SA)
@inline Base.zero{SA <: StaticArray}(a::Type{SA}) = zeros(SA)

@inline one(::SM) where {SM <: StaticMatrix} = _one(Size(SM), SM)
@inline one(::Type{SM}) where {SM <: StaticMatrix} = _one(Size(SM), SM)
@generated function _one(::Size{S}, ::Type{SM}) where {S, SM <: StaticArray}
    if (length(S) != 2) || (S[1] != S[2])
        error("multiplicative identity defined only for square matrices")
    end
    T = eltype(SM) # should be "hyperpure"
    if T == Any
        T = Float64
    end
    exprs = [i == j ? :(one($T)) : :(zero($T)) for i ∈ 1:S[1], j ∈ 1:S[2]]
    return quote
        $(Expr(:meta, :inline))
        SM(tuple($(exprs...)))
    end
end

@inline eye(::SM) where {SM <: StaticMatrix} = _eye(Size(SM), SM)
@inline eye(::Type{SM}) where {SM <: StaticMatrix} = _eye(Size(SM), SM)
@generated function _eye(::Size{S}, ::Type{SM}) where {S, SM <: StaticArray}
    T = eltype(SM) # should be "hyperpure"
    if T == Any
        T = Float64
    end
    exprs = [i == j ? :(one($T)) : :(zero($T)) for i ∈ 1:S[1], j ∈ 1:S[2]]
    return quote
        $(Expr(:meta, :inline))
        SM(tuple($(exprs...)))
    end
end

@inline diagm(v::StaticVector) = _diagm(Size(v), v)
@generated function _diagm(::Size{S}, v::StaticVector) where {S}
    Snew = (S[1], S[1])
    T = eltype(v)
    exprs = [i == j ? :(v[$i]) : zero(T) for i = 1:S[1], j = 1:S[1]]
    return quote
        $(Expr(:meta, :inline))
        @inbounds return similar_type($v, Size($Snew))(tuple($(exprs...)))
    end
end

@inline cross(a::StaticVector, b::StaticVector) = _cross(same_size(a, b), a, b)
_cross(::Size{S}, a::StaticVector, b::StaticVector) where {S} = error("Cross product not defined for $(S[1])-vectors")
@inline function _cross(::Size{(2,)}, a::StaticVector, b::StaticVector)
    @inbounds return a[1]*b[2] - a[2]*b[1]
end
@inline function _cross(::Size{(3,)}, a::StaticVector, b::StaticVector)
    @inbounds return similar_type(a, typeof(a[2]*b[3]-a[3]*b[2]))((a[2]*b[3]-a[3]*b[2], a[3]*b[1]-a[1]*b[3], a[1]*b[2]-a[2]*b[1]))
end
@inline function _cross(::Size{(2,)}, a::StaticVector{<:Any, <:Unsigned}, b::StaticArray{<:Any, <:Unsigned})
    @inbounds return Signed(a[1]*b[2]) - Signed(a[2]*b[1])
end
@inline function _cross(::Size{(3,)}, a::StaticArray{<:Any, <:Unsigned}, b::StaticArray{<:Any, <:Unsigned})
    @inbounds return similar_type(a, typeof(Signed(a[2]*b[3])-Signed(a[3]*b[2])))(((Signed(a[2]*b[3])-Signed(a[3]*b[2]), Signed(a[3]*b[1])-Signed(a[1]*b[3]), Signed(a[1]*b[2])-Signed(a[2]*b[1]))))
end

@inline dot(a::StaticVector, b::StaticVector) = _vecdot(same_size(a, b), a, b)
@inline vecdot(a::StaticArray, b::StaticArray) = _vecdot(same_size(a, b), a, b)
@generated function _vecdot(::Size{S}, a::StaticArray, b::StaticArray) where {S}
    if prod(S) == 0
        return :(zero(promote_op(*, eltype(a), eltype(b))))
    end

    expr = :(conj(a[1]) * b[1])
    for j = 2:prod(S)
        expr = :($expr + conj(a[$j]) * b[$j])
    end

    return quote
        @_inline_meta
        @inbounds return $expr
    end
end

@inline norm(v::StaticVector) = vecnorm(v)
@inline norm(v::StaticVector, p::Real) = vecnorm(v, p)

@inline Base.LinAlg.norm_sqr(v::StaticVector) = mapreduce(abs2, +, zero(real(eltype(v))), v)

@inline vecnorm(a::StaticArray) = _vecnorm(Size(a), a)
@generated function _vecnorm(::Size{S}, a::StaticArray) where {S}
    if prod(S) == 0
        return zero(real(eltype(a)))
    end

    expr = :(abs2(a[1]))
    for j = 2:prod(S)
        expr = :($expr + abs2(a[$j]))
    end

    return quote
        $(Expr(:meta, :inline))
        @inbounds return sqrt($expr)
    end
end

_norm_p0(x) = x == 0 ? zero(x) : one(x)

@inline vecnorm(a::StaticArray, p::Real) = _vecnorm(Size(a), a, p)
@generated function _vecnorm(::Size{S}, a::StaticArray, p::Real) where {S}
    if prod(S) == 0
        return zero(real(eltype(a)))
    end

    expr = :(abs(a[1])^p)
    for j = 2:prod(S)
        expr = :($expr + abs(a[$j])^p)
    end

    expr_p1 = :(abs(a[1]))
    for j = 2:prod(S)
        expr_p1 = :($expr_p1 + abs(a[$j]))
    end

    return quote
        $(Expr(:meta, :inline))
        if p == Inf
            return mapreduce(abs, max, $(zero(real(eltype(a)))), a)
        elseif p == 1
            @inbounds return $expr_p1
        elseif p == 2
            return vecnorm(a)
        elseif p == 0
            return mapreduce(_norm_p0, +, $(zero(real(eltype(a)))), a)
        else
            @inbounds return ($expr)^(inv(p))
        end
    end
end

@inline normalize(a::StaticVector) = inv(vecnorm(a))*a
@inline normalize(a::StaticVector, p::Real) = inv(vecnorm(a, p))*a

@inline normalize!(a::StaticVector) = (a .*= inv(vecnorm(a)))
@inline normalize!(a::StaticVector, p::Real) = (a .*= inv(vecnorm(a, p)))

@inline trace(a::StaticMatrix) = _trace(Size(a), a)
@generated function _trace(::Size{S}, a::StaticMatrix) where {S}
    if S[1] != S[2]
        throw(DimensionMismatch("matrix is not square"))
    end

    if S[1] == 0
        return zero(eltype(a))
    end

    exprs = [:(a[$(sub2ind(S, i, i))]) for i = 1:S[1]]
    total = reduce((ex1, ex2) -> :($ex1 + $ex2), exprs)

    return quote
        @_inline_meta
        @inbounds return $total
    end
end

# TODO same for `RowVector`?
@inline Size(::Union{RowVector{T, SA}, Type{RowVector{T, SA}}}) where {T, SA <: StaticArray} = Size(1, Size(SA)[1])
@inline Size(::Union{Symmetric{T,SA}, Type{Symmetric{T,SA}}}) where {T,SA<:StaticArray} = Size(SA)
@inline Size(::Union{Hermitian{T,SA}, Type{Hermitian{T,SA}}}) where {T,SA<:StaticArray} = Size(SA)

# some micro-optimizations (TODO check these make sense for v0.6)
@inline Base.LinAlg.checksquare{SM<:StaticMatrix}(::SM) = _checksquare(Size(SM))
@inline Base.LinAlg.checksquare{SM<:StaticMatrix}(::Type{SM}) = _checksquare(Size(SM))

@pure _checksquare{S}(::Size{S}) = (S[1] == S[2] || error("marix must be square"); S[1])

@inline Base.LinAlg.Symmetric(A::StaticMatrix, uplo::Char='U') = (Base.LinAlg.checksquare(A);Symmetric{eltype(A),typeof(A)}(A, uplo))
@inline Base.LinAlg.Hermitian(A::StaticMatrix, uplo::Char='U') = (Base.LinAlg.checksquare(A);Hermitian{eltype(A),typeof(A)}(A, uplo))
