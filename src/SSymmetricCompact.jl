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

@inline function (::Type{SSymmetricCompact{N, T}})(lowertriangle::SVector{L}) where {N, T, L}
    SSymmetricCompact{N, T, L}(lowertriangle)
end

@inline function (::Type{SSymmetricCompact{N}})(lowertriangle::SVector{L, T}) where {N, T, L}
    SSymmetricCompact{N, T, L}(lowertriangle)
end

@generated function SSymmetricCompact(lowertriangle::SVector{L, T}) where {T, L}
    N = div(isqrt(8 * L + 1) - 1, 2) # from quadratic formula
    quote
        @_inline_meta
        SSymmetricCompact{$N, T, L}(lowertriangle)
    end
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

@generated function (::Type{SSymmetricCompact{N, T}})(a::Tuple) where {N, T}
    L = div(N * (N + 1), 2)
    quote
        @_inline_meta
        SSymmetricCompact{N, T, $L}(a)
    end
end

@inline (::Type{SSymmetricCompact{N}})(a::NTuple{M, T}) where {N, T, M} = SSymmetricCompact{N, T}(a)
@inline SSymmetricCompact(a::StaticMatrix{N, N, T}) where {N, T} = SSymmetricCompact{N, T}(a)

@inline (::Type{SSC})(a::SSymmetricCompact) where {SSC<:SSymmetricCompact} = SSC(a.lowertriangle)
@inline (::Type{SSC})(a::SSC) where {SSC<:SSymmetricCompact} = SSC(a.lowertriangle)

lowertriangletype(::Type{SSymmetricCompact{N, T, L}}) where {N, T, L} = SVector{L, T}
lowertriangletype(::Type{<:SSymmetricCompact}) = SVector

@inline (::Type{SSC})(a::AbstractVector) where {SSC <: SSymmetricCompact} = SSC(convert(lowertriangletype(SSC), a))
@inline (::Type{SSC})(a::Tuple) where {SSC <: SSymmetricCompact} = SSymmetricCompact(convert(lowertriangletype(SSC), a))

convert(::Type{SSC}, a::SSC) where {SSC<:SSymmetricCompact} = a
# TODO: more convert methods?

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

LinAlg.ishermitian(a::SSymmetricCompact{N, T}) where {N,T<:Real} = true
LinAlg.ishermitian(a::SSymmetricCompact) = all(isreal, a.lowertriangle)
LinAlg.issymmetric(a::SSymmetricCompact) = true

# TODO: factorize

@inline ==(a::SSymmetricCompact, b::SSymmetricCompact) = a.lowertriangle == b.lowertriangle
@inline -(a::SSymmetricCompact) = SSymmetricCompact(-a.lowertriangle)
@inline +(a::SSymmetricCompact, b::SSymmetricCompact) = SSymmetricCompact(a.lowertriangle + b.lowertriangle)
@inline -(a::SSymmetricCompact, b::SSymmetricCompact) = SSymmetricCompact(a.lowertriangle - b.lowertriangle)

@inline *(x::Number, a::SSymmetricCompact) = SSymmetricCompact(x * a.lowertriangle)
@inline *(a::SSymmetricCompact, x::Number) = SSymmetricCompact(a.lowertriangle * x)
@inline /(a::SSymmetricCompact, x::Number)= SSymmetricCompact(a.lowertriangle / x)

@inline conj(a::SSymmetricCompact) = SSymmetricCompact(conj(a.lowertriangle))
@inline transpose(a::SSymmetricCompact) = a
@inline adjoint(a::SSymmetricCompact) = conj(a)

#TODO: eye, one, zero