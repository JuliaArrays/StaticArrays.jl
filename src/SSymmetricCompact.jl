import Base: ==, -, +, *, /, \, abs, real, imag, conj, transpose, convert

@generated function check_symmetric_parameters(::Val{N}, ::Val{L}) where {N, L}
    if 2 * L != N * (N + 1)
        return :(throw(ArgumentError("Size mismatch in SSymmetricCompact parameters. Got dimension $N and length $L.")))
    end
    :(nothing)
end

struct SSymmetricCompact{N, T, L} <: StaticMatrix{N, N, T}
    lowertriangle::SVector{L, T}

    @inline function SSymmetricCompact{N, T, L}(lowertriangle::SVector{L}) where {N, T, L}
        check_symmetric_parameters(Val(N), Val(L))
        new{N, T, L}(lowertriangle)
    end
end

lowertriangletype(::Type{SSymmetricCompact{N, T, L}}) where {N, T, L} = SVector{L, T}
lowertriangletype(::Type{<:SSymmetricCompact}) = SVector
Base.@pure triangularnumber(N::Int) = div(N * (N + 1), 2)
Base.@pure triangularroot(L::Int) = div(isqrt(8 * L + 1) - 1, 2) # from quadratic formula

@inline (::Type{SSymmetricCompact{N, T}})(lowertriangle::SVector{L}) where {N, T, L} = SSymmetricCompact{N, T, L}(lowertriangle)
@inline (::Type{SSymmetricCompact{N}})(lowertriangle::SVector{L, T}) where {N, T, L} = SSymmetricCompact{N, T, L}(lowertriangle)

@inline function SSymmetricCompact(lowertriangle::SVector{L, T}) where {T, L}
    N = triangularroot(L)
    SSymmetricCompact{N, T, L}(lowertriangle)
end

@generated function (::Type{SSymmetricCompact{N, T, L}})(a::Tuple) where {N, T, L}
    expr = Vector{Expr}(L)
    i = 0
    for col = 1 : N, row = col : N
        index = N * (col - 1) + row
        expr[i += 1] = :(a[$index])
    end
    quote
        @_inline_meta
        @inbounds return SSymmetricCompact{N, T, L}(SVector{L, T}($(expr...)))
    end
end

@inline function (::Type{SSymmetricCompact{N, T}})(a::Tuple) where {N, T}
    L = triangularnumber(N)
    SSymmetricCompact{N, T, L}(a)
end

@inline (::Type{SSymmetricCompact{N}})(a::NTuple{M, T}) where {N, T, M} = SSymmetricCompact{N, T}(a)
@inline SSymmetricCompact(a::StaticMatrix{N, N, T}) where {N, T} = SSymmetricCompact{N, T}(a)

@inline (::Type{SSC})(a::SSymmetricCompact) where {SSC <: SSymmetricCompact} = SSC(a.lowertriangle)
@inline (::Type{SSC})(a::SSC) where {SSC <: SSymmetricCompact} = SSC(a.lowertriangle)

@inline (::Type{SSC})(a::AbstractVector) where {SSC <: SSymmetricCompact} = SSC(convert(lowertriangletype(SSC), a))
@inline (::Type{SSC})(a::Tuple) where {SSC <: SSymmetricCompact} = SSymmetricCompact(convert(lowertriangletype(SSC), a))

convert(::Type{SSC}, a::SSC) where {SSC <: SSymmetricCompact} = a # TODO: needed?
# TODO: more convert methods?

# TODO: is the following a good idea?
@inline function similar_type(::Type{SSC}, ::Type{T}, ::Size{S}) where {SSC <: SSymmetricCompact, T, S <: Tuple{Int, Int}}
    if S[1] === S[2]
        N = S[1]
        L = triangularnumber(N)
        SSymmetricCompact{N, T, L}
    else
        default_similar_type(T, S, length_val(S))
    end
end

@inline indextuple(::T) where {T <: SSymmetricCompact} = indextuple(T)
@generated function indextuple(::Type{<:SSymmetricCompact{N}}) where N
    indexmat = zeros(Int, N, N)
    i = 0
    for col = 1 : N, row = 1 : N
        indexmat[row, col] = if row >= col
            i += 1
        else
            indexmat[col, row]
        end
    end
    quote
        Base.@_inline_meta
        return $(tuple(indexmat...))
    end
end

@inline function Base.getindex(a::SSymmetricCompact{N}, i::Int) where {N}
    index = indextuple(a)[i]
    @inbounds return a.lowertriangle[index]
end

@generated function setindex(a::SSymmetricCompact{N, T, L}, x, index::Int) where {N, T, L}
    I = indextuple(a)
    exprs = [:(ifelse($i == $(I)[index], T(x), a.lowertriangle[$i])) for i = 1:L]
    quote
        @_inline_meta
        @boundscheck if (index < 1 || index > $(N * N))
            throw(BoundsError(a, index))
        end
        return typeof(a)(SVector{L, T}(tuple($(exprs...))))
    end
end

# needed because it is used in convert.jl and the generic fallback is slow
@generated function Tuple(a::SSymmetricCompact{N}) where N
    exprs = [:(a[$i]) for i = 1 : N^2]
    quote
        @_inline_meta
        tuple($(exprs...))
    end
end

LinAlg.ishermitian(a::SSymmetricCompact{N, T}) where {N,T <: Real} = true
LinAlg.ishermitian(a::SSymmetricCompact) = all(isreal, a.lowertriangle)
LinAlg.issymmetric(a::SSymmetricCompact) = true

# TODO: factorize

# TODO: a.lowertriangle == b.lowertriangle is slow (used by SDiagonal). SMatrix etc. actually use AbstractArray fallback (also slow)
@inline ==(a::SSymmetricCompact, b::SSymmetricCompact) = mapreduce(==, (x, y) -> x && y, a.lowertriangle, b.lowertriangle)
@generated function _map(f, ::Size{S}, a::SSymmetricCompact...) where {S}
    N = S[1]
    L = triangularnumber(N)
    exprs = Vector{Expr}(L)
    for i ∈ 1:L
        tmp = [:(a[$j][$i]) for j ∈ 1:length(a)]
        exprs[i] = :(f($(tmp...)))
    end
    return quote
        @_inline_meta
        @inbounds return SSymmetricCompact(SVector(tuple($(exprs...))))
    end
end

# Scalar-array. TODO: overload broadcast instead, once API has stabilized a bit
@inline +(a::Number, b::SSymmetricCompact) = SSymmetricCompact(a + b.lowertriangle)
@inline +(a::SSymmetricCompact, b::Number) = SSymmetricCompact(a.lowertriangle + b)

@inline -(a::Number, b::SSymmetricCompact) = SSymmetricCompact(a - b.lowertriangle)
@inline -(a::SSymmetricCompact, b::Number) = SSymmetricCompact(a.lowertriangle - b)

@inline *(a::Number, b::SSymmetricCompact) = SSymmetricCompact(a * b.lowertriangle)
@inline *(a::SSymmetricCompact, b::Number) = SSymmetricCompact(a.lowertriangle * b)

@inline /(a::SSymmetricCompact, b::Number) = SSymmetricCompact(a.lowertriangle / b)
@inline \(a::Number, b::SSymmetricCompact) = SSymmetricCompact(a \ b.lowertriangle)

# TODO: operations With UniformScaling

@inline transpose(a::SSymmetricCompact) = SSymmetricCompact(transpose.(a.lowertriangle))
@inline adjoint(a::SSymmetricCompact) = conj(a)

#TODO: one, eye

@generated function _one(::Size{S}, ::Type{SSC}) where {S, SSC <: SSymmetricCompact}
    N = S[1]
    L = triangularnumber(N)
    T = eltype(SSC)
    if T == Any
        T = Float64
    end
    exprs = Vector{Expr}(L)
    i = 0
    for col = 1 : N, row = col : N
        exprs[i += 1] = row == col ? :(one($T)) : :(zero($T))
    end
    quote
        @_inline_meta
        return SSymmetricCompact(SVector(tuple($(exprs...))))
    end
end

@inline _eye(s::Size{S}, t::Type{SSC}) where {S, SM <: SSymmetricCompact} = _one(s, t)

# _fill covers fill, zeros, and ones:
@generated function _fill(val, ::Size{s}, ::Type{SSC}) where {s, SSC <: SSymmetricCompact}
    N = s[1]
    L = triangularnumber(N)
    v = [:val for i = 1:L]
    return quote
        @_inline_meta
        $SSC(SVector(tuple($(v...))))
    end
end

@generated function _rand(randfun, rng::AbstractRNG, ::Type{SSC}) where {N, SSC <: SSymmetricCompact{N}}
    T = eltype(SSC)
    if T == Any
        T = Float64
    end
    L = triangularnumber(N)
    v = [:(randfun(rng, $T)) for i = 1:L]
    return quote
        @_inline_meta
        $SSC(SVector(tuple($(v...))))
    end
end

@inline rand(rng::AbstractRNG, ::Type{SSC}) where {SSC <: SSymmetricCompact} = _rand(rand, rng, SSC)
@inline randn(rng::AbstractRNG, ::Type{SSC}) where {SSC <: SSymmetricCompact} = _rand(randn, rng, SSC)
@inline randexp(rng::AbstractRNG, ::Type{SSC}) where {SSC <: SSymmetricCompact} = _rand(randexp, rng, SSC)
