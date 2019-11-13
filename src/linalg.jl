import Base: +, -, *, /, \

#--------------------------------------------------
# Vector space algebra

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
    n = checksquare(a)
    exprs = [i == j ? :(a[$(LinearIndices(S)[i, j])] + λ) : :(a[$(LinearIndices(S)[i, j])]) for i = 1:n, j = 1:n]
    return quote
        $(Expr(:meta, :inline))
        @inbounds return similar_type(a, promote_type(eltype(a), typeof(λ)))(tuple($(exprs...)))
    end
end

@inline *(a::UniformScaling, b::Union{StaticMatrix,StaticVector}) = a.λ * b
@inline *(a::StaticMatrix, b::UniformScaling) = a * b.λ
@inline \(a::UniformScaling, b::Union{StaticMatrix,StaticVector}) = a.λ \ b
@inline /(a::StaticMatrix, b::UniformScaling) = a / b.λ

#--------------------------------------------------
# Matrix algebra

# Transpose, conjugate, etc
@inline conj(a::StaticArray) = map(conj, a)
@inline transpose(m::StaticMatrix) = _transpose(Size(m), m)
# note: transpose of StaticVector is a Transpose, handled by Base
@inline transpose(a::Transpose{<:Any,<:Union{StaticVector,StaticMatrix}}) = a.parent
@inline transpose(a::Adjoint{<:Any,<:Union{StaticVector,StaticMatrix}}) = conj(a.parent)
@inline transpose(a::Adjoint{<:Real,<:Union{StaticVector,StaticMatrix}}) = a.parent

@generated function _transpose(::Size{S}, m::StaticMatrix) where {S}
    Snew = (S[2], S[1])

    exprs = [:(transpose(m[$(LinearIndices(S)[j1, j2])])) for j2 = 1:S[2], j1 = 1:S[1]]

    return quote
        $(Expr(:meta, :inline))
        @inbounds return similar_type($m, Size($Snew))(tuple($(exprs...)))
    end
end

@inline adjoint(m::StaticMatrix) = _adjoint(Size(m), m)
@inline adjoint(a::Transpose{<:Any,<:Union{StaticVector,StaticMatrix}}) = conj(a.parent)
@inline adjoint(a::Transpose{<:Real,<:Union{StaticVector,StaticMatrix}}) = a.parent
@inline adjoint(a::Adjoint{<:Any,<:Union{StaticVector,StaticMatrix}}) = a.parent

@generated function _adjoint(::Size{S}, m::StaticMatrix) where {S}
    Snew = (S[2], S[1])

    exprs = [:(adjoint(m[$(LinearIndices(S)[j1, j2])])) for j2 = 1:S[2], j1 = 1:S[1]]

    return quote
        $(Expr(:meta, :inline))
        @inbounds return similar_type($m, Size($Snew))(tuple($(exprs...)))
    end
end

@inline Base.zero(a::SA) where {SA <: StaticArray} = zeros(SA)
@inline Base.zero(a::Type{SA}) where {SA <: StaticArray} = zeros(SA)

@inline one(m::StaticMatrixLike) = _one(Size(m), m)
@inline one(::Type{SM}) where {SM<:StaticMatrixLike}= _one(Size(SM), SM)
function _one(s::Size, m_or_SM)
    if (length(s) != 2) || (s[1] != s[2])
        throw(DimensionMismatch("multiplicative identity defined only for square matrices"))
    end
    _scalar_matrix(s, m_or_SM, one(_eltype_or(m_or_SM, Float64)))
end

# StaticMatrix(I::UniformScaling)
(::Type{SM})(I::UniformScaling) where {SM<:StaticMatrix} = _scalar_matrix(Size(SM), SM, I.λ)
# The following oddity is needed if we want `SArray{Tuple{2,3}}(I)` to work
# because we do not have `SArray{Tuple{2,3}} <: StaticMatrix`.
(::Type{SM})(I::UniformScaling) where {SM<:(StaticArray{Tuple{N,M}} where {N,M})} =
    _scalar_matrix(Size(SM), SM, I.λ)

# Construct a matrix with the scalar λ on the diagonal and zeros off the
# diagonal. The matrix can be non-square.
@generated function _scalar_matrix(s::Size{S}, m_or_SM, λ) where {S}
    elements = Symbol[i == j ? :λ : :λzero for i in 1:S[1], j in 1:S[2]]
    return quote
        $(Expr(:meta, :inline))
        λzero = zero(λ)
        _construct_similar(m_or_SM, s, tuple($(elements...)))
    end
end

@generated function diagm(kvs::Pair{<:Val,<:StaticVector}...)
    N = maximum(abs(kv.parameters[1].parameters[1]) + length(kv.parameters[2]) for kv in kvs)
    X = [Symbol("x_$(i)_$(j)") for i in 1:N, j in 1:N]
    T = promote_type((eltype(kv.parameters[2]) for kv in kvs)...)
    exprs = fill(:(zero($T)), N*N)
    for m in eachindex(kvs)
        kv = kvs[m]
        ind = diagind(N, N, kv.parameters[1].parameters[1])
        for n = 1:length(kv.parameters[2])
            exprs[ind[n]] = :(kvs[$m].second[$n])
        end
    end
    return quote
        $(Expr(:meta, :inline))
        @inbounds return SMatrix{$N,$N,$T}(tuple($(exprs...)))
    end
end

@inline diag(m::StaticMatrix, k::Type{Val{D}}=Val{0}) where {D} = _diag(Size(m), m, k)
@generated function _diag(::Size{S}, m::StaticMatrix, ::Type{Val{D}}) where {S,D}
    S1, S2 = S
    rng = D ≤ 0 ? range(1-D, step=S1+1, length=min(S1+D, S2)) :
                  range(D*S1+1, step=S1+1, length=min(S1, S2-D))
    Snew = length(rng)
    T = eltype(m)
    exprs = [:(m[$i]) for i = rng]
    return quote
        $(Expr(:meta, :inline))
        @inbounds return similar_type($m, Size($Snew))(tuple($(exprs...)))
    end
end

#--------------------------------------------------
# Vector products
@inline cross(a::StaticVector, b::StaticVector) = _cross(same_size(a, b), a, b)
_cross(::Size{S}, a::StaticVector, b::StaticVector) where {S} = error("Cross product not defined for $(S[1])-vectors")
@inline function _cross(::Size{(2,)}, a::StaticVector, b::StaticVector)
    @inbounds return a[1]*b[2] - a[2]*b[1]
end
@inline function _cross(::Size{(3,)}, a::StaticVector, b::StaticVector)
    @inbounds return similar_type(a, typeof(a[2]*b[3]-a[3]*b[2]))((a[2]*b[3]-a[3]*b[2], a[3]*b[1]-a[1]*b[3], a[1]*b[2]-a[2]*b[1]))
end
@inline function _cross(::Size{(2,)}, a::StaticVector{<:Any, <:Unsigned}, b::StaticVector{<:Any, <:Unsigned})
    @inbounds return Signed(a[1]*b[2]) - Signed(a[2]*b[1])
end
@inline function _cross(::Size{(3,)}, a::StaticVector{<:Any, <:Unsigned}, b::StaticVector{<:Any, <:Unsigned})
    @inbounds return similar_type(a, typeof(Signed(a[2]*b[3])-Signed(a[3]*b[2])))(((Signed(a[2]*b[3])-Signed(a[3]*b[2]), Signed(a[3]*b[1])-Signed(a[1]*b[3]), Signed(a[1]*b[2])-Signed(a[2]*b[1]))))
end

@inline dot(a::StaticArray, b::StaticArray) = _vecdot(same_size(a, b), a, b, dot)
@inline bilinear_vecdot(a::StaticArray, b::StaticArray) = _vecdot(same_size(a, b), a, b, *)

@inline function _vecdot(::Size{S}, a::StaticArray, b::StaticArray, product) where {S}
    if prod(S) == 0
        za, zb = zero(eltype(a)), zero(eltype(b))
    else
        # Use an actual element if there is one, to support e.g. Vector{<:Number}
        # element types for which runtime size information is required to construct
        # a zero element.
        za, zb = zero(a[1]), zero(b[1])
    end
    ret = product(za, zb) + product(za, zb)
    @inbounds @simd for j = 1 : prod(S)
        ret += product(a[j], b[j])
    end
    return ret
end

#--------------------------------------------------
# Norms
@inline LinearAlgebra.norm_sqr(v::StaticVector) = mapreduce(abs2, +, v; init=zero(real(eltype(v))))

@inline norm(a::StaticArray) = _norm(Size(a), a)
@generated function _norm(::Size{S}, a::StaticArray) where {S}
    if prod(S) == 0
        return :(zero(real(eltype(a))))
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

@inline norm(a::StaticArray, p::Real) = _norm(Size(a), a, p)
@generated function _norm(::Size{S}, a::StaticArray, p::Real) where {S}
    if prod(S) == 0
        return :(zero(real(eltype(a))))
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
            return mapreduce(abs, max, a)
        elseif p == 1
            @inbounds return $expr_p1
        elseif p == 2
            return norm(a)
        elseif p == 0
            return mapreduce(_norm_p0, +, a)
        else
            @inbounds return ($expr)^(inv(p))
        end
    end
end

@inline normalize(a::StaticVector) = inv(norm(a))*a
@inline normalize(a::StaticVector, p::Real) = inv(norm(a, p))*a

@inline normalize!(a::StaticVector) = (a .*= inv(norm(a)); return a)
@inline normalize!(a::StaticVector, p::Real) = (a .*= inv(norm(a, p)); return a)

@inline tr(a::StaticMatrix) = _tr(Size(a), a)
@generated function _tr(::Size{S}, a::StaticMatrix) where {S}
    checksquare(a)

    if S[1] == 0
        return :(zero(eltype(a)))
    end

    exprs = [:(a[$(LinearIndices(S)[i, i])]) for i = 1:S[1]]
    total = reduce((ex1, ex2) -> :($ex1 + $ex2), exprs)

    return quote
        @_inline_meta
        @inbounds return $total
    end
end


#--------------------------------------------------
# Outer products

const _length_limit = Length(200)

@inline kron(a::StaticMatrix, b::StaticMatrix) = _kron(_length_limit, Size(a), Size(b), a, b)
@generated function _kron(::Length{length_limit}, ::Size{SA}, ::Size{SB}, a, b) where {length_limit,SA,SB}
    outsize = SA .* SB
    if prod(outsize) > length_limit
        return :( SizedMatrix{$(outsize[1]),$(outsize[2])}( kron(drop_sdims(a), drop_sdims(b)) ) )
    end

    M = [ :(a[$ia,$ja] * b[$ib,$jb]) for ib in 1:SB[1], ia in 1:SA[1], jb in 1:SB[2], ja in 1:SA[2] ]

    return quote
        @_inline_meta
        @inbounds return  similar_type($a, promote_type(eltype(a),eltype(b)), Size($(outsize)))(tuple($(M...)))
    end
end

@inline kron(a::StaticVector, b::StaticVector) = _kron_vec_x_vec(_length_limit, Size(a), Size(b), a, b)
@generated function _kron_vec_x_vec(::Length{length_limit}, ::Size{SA}, ::Size{SB}, a, b) where {length_limit,SA,SB}
    outsize = SA .* SB
    if prod(outsize) > length_limit
        return :( SizedVector{$(outsize[1])}( kron(drop_sdims(a), drop_sdims(b)) ) )
    end

    m = [ :(a[$ia] * b[$ib]) for ib in 1:SB[1], ia in 1:SA[1]]

    return quote
        @_inline_meta
        @inbounds return similar_type($a, promote_type(eltype(a),eltype(b)), Size($(outsize)))(tuple($(m...)))
    end
end

@inline function kron(
        a::Union{Transpose{<:Number,<:StaticVector}, Adjoint{<:Number,<:StaticVector}},
        b::Union{Transpose{<:Number,<:StaticVector}, Adjoint{<:Number,<:StaticVector}})
    _kron_tvec_x_tvec(_length_limit, Size(a), Size(b), a, b)
end
@generated function _kron_tvec_x_tvec(::Length{length_limit}, ::Size{SA}, ::Size{SB}, a, b) where {length_limit,SA,SB}
    outsize = SA .* SB
    if prod(outsize) > length_limit
        return :( SizedMatrix{$(outsize[1]),$(outsize[2])}( kron(drop_sdims(a), drop_sdims(b)) ) )
    end

    m = [ :(a[$ia] * b[$ib]) for ib in 1:SB[2], ia in 1:SA[2]]

    return quote
        @_inline_meta
        @inbounds return similar_type($a, promote_type(eltype(a),eltype(b)), Size($(outsize)))(tuple($(m...)))
    end
end

@inline function kron(
        a::Union{Transpose{<:Number,<:StaticVector}, Adjoint{<:Number,<:StaticVector}},
        b::StaticVector)
    _kron_tvec_x_vec(_length_limit, Size(a), Size(b), a, b)
end
@generated function _kron_tvec_x_vec(::Length{length_limit}, ::Size{SA}, ::Size{SB}, a, b) where {length_limit,SA,SB}
    outsize = (SA[1] * SB[1], SA[2])
    if prod(outsize) > length_limit
        return :( SizedMatrix{$(outsize[1]),$(outsize[2])}( kron(drop_sdims(a), drop_sdims(b)) ) )
    end

    M = [ :(a[$ia] * b[$ib]) for ib in 1:SB[1], ia in 1:SA[2]]

    return quote
        @_inline_meta
        @inbounds return similar_type($a, promote_type(eltype(a),eltype(b)), Size($(outsize)))(tuple($(M...)))
    end
end

@inline function kron(
        a::StaticVector,
        b::Union{Transpose{<:Number,<:StaticVector}, Adjoint{<:Number,<:StaticVector}})
    _kron_vec_x_tvec(_length_limit, Size(a), Size(b), a, b)
end
@generated function _kron_vec_x_tvec(::Length{length_limit}, ::Size{SA}, ::Size{SB}, a, b) where {length_limit,SA,SB}
    outsize = (SA[1] * SB[1], SB[2])
    if prod(outsize) > length_limit
        return :( SizedMatrix{$(outsize[1]),$(outsize[2])}( kron(drop_sdims(a), drop_sdims(b)) ) )
    end

    M = [ :(a[$ia] * b[$ib]) for  ia in 1:SA[1], ib in 1:SB[2]]

    return quote
        @_inline_meta
        @inbounds return similar_type($a, promote_type(eltype(a),eltype(b)), Size($(outsize)))(tuple($(M...)))
    end
end

@inline kron(a::StaticVector, b::StaticMatrix) = _kron_vec_x_mat(_length_limit, Size(a), Size(b), a, b)
@generated function _kron_vec_x_mat(::Length{length_limit}, ::Size{SA}, ::Size{SB}, a, b) where {length_limit,SA,SB}
    outsize = (SA[1] * SB[1], SB[2])
    if prod(outsize) > length_limit
        return :( SizedMatrix{$(outsize[1]),$(outsize[2])}( kron(drop_sdims(a), drop_sdims(b)) ) )
    end

    M = [ :(a[$ia] * b[$ib,$jb]) for ib in 1:SB[1], ia in 1:SA[1], jb in 1:SB[2]]

    return quote
        @_inline_meta
        @inbounds return similar_type($b, promote_type(eltype(a),eltype(b)), Size($(outsize)))(tuple($(M...)))
    end
end

@inline kron(a::StaticMatrix, b::StaticVector) = _kron_mat_x_vec(_length_limit, Size(a), Size(b), a, b)
@generated function _kron_mat_x_vec(::Length{length_limit}, ::Size{SA}, ::Size{SB}, a, b) where {length_limit,SA,SB}
    outsize = (SA[1] * SB[1], SA[2])
    if prod(outsize) > length_limit
        return :( SizedMatrix{$(outsize[1]),$(outsize[2])}( kron(drop_sdims(a), drop_sdims(b)) ) )
    end

    M = [ :(a[$ia,$ja] * b[$ib]) for ib in 1:SB[1], ia in 1:SA[1], ja in 1:SA[2] ]

    return quote
        @_inline_meta
        @inbounds return similar_type($a, promote_type(eltype(a),eltype(b)), Size($(outsize)))(tuple($(M...)))
    end
end

@inline function kron(
        a::StaticMatrix,
        b::Union{Transpose{<:Number,<:StaticVector}, Adjoint{<:Number,<:StaticVector}})
    _kron_mat_x_tvec(_length_limit, Size(a), Size(b), a, b)
end
@generated function _kron_mat_x_tvec(::Length{length_limit}, ::Size{SA}, ::Size{SB}, a, b) where {length_limit,SA,SB}
    outsize = SA .* SB
    if prod(outsize) > length_limit
        return :( SizedMatrix{$(outsize[1]),$(outsize[2])}( kron(drop_sdims(a), drop_sdims(b)) ) )
    end

    M = [ :(a[$ia,$ja] * b[$ib,$jb]) for ib in 1:SB[1], ia in 1:SA[1], jb in 1:SB[2], ja in 1:SA[2] ]

    return quote
        @_inline_meta
        @inbounds return  similar_type($a, promote_type(eltype(a),eltype(b)), Size($(outsize)))(tuple($(M...)))
    end
end

@inline function kron(
        a::Union{Transpose{<:Number,<:StaticVector}, Adjoint{<:Number,<:StaticVector}},
        b::StaticMatrix)
    _kron_tvec_x_mat(_length_limit, Size(a), Size(b), a, b)
end
@generated function _kron_tvec_x_mat(::Length{length_limit}, ::Size{SA}, ::Size{SB}, a, b) where {length_limit,SA,SB}
    outsize = SA .* SB
    if prod(outsize) > length_limit
        return :( SizedMatrix{$(outsize[1]),$(outsize[2])}( kron(drop_sdims(a), drop_sdims(b)) ) )
    end

    M = [ :(a[$ia,$ja] * b[$ib,$jb]) for ib in 1:SB[1], ia in 1:SA[1], jb in 1:SB[2], ja in 1:SA[2] ]

    return quote
        @_inline_meta
        @inbounds return  similar_type($b, promote_type(eltype(a),eltype(b)), Size($(outsize)))(tuple($(M...)))
    end
end


#--------------------------------------------------
# Some shimming for special linear algebra matrix types
@inline LinearAlgebra.Symmetric(A::StaticMatrix, uplo::Char='U') = (checksquare(A); Symmetric{eltype(A),typeof(A)}(A, uplo))
@inline LinearAlgebra.Hermitian(A::StaticMatrix, uplo::Char='U') = (checksquare(A); Hermitian{eltype(A),typeof(A)}(A, uplo))

