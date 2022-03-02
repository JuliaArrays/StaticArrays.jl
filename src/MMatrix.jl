"""
    MMatrix{S1, S2, T, L}(undef)
    MMatrix{S1, S2, T, L}(x::NTuple{L, T})
    MMatrix{S1, S2, T, L}(x1, x2, x3, ...)

Construct a statically-sized, mutable matrix `MMatrix`. The data may optionally
be provided upon construction and can be mutated later. The `L` parameter is the
`length` of the array and is always equal to `S1 * S2`. Constructors may drop
the `L`, `T` and even `S2` parameters if they are inferrable from the input
(e.g. `L` is always inferrable from `S1` and `S2`).

    MMatrix{S1, S2}(mat::Matrix)

Construct a statically-sized, mutable matrix of dimensions `S1 × S2` using the data from
`mat`. The parameters `S1` and `S2` are mandatory since the size of `mat` is
unknown to the compiler (the element type may optionally also be specified).
"""
const MMatrix{S1, S2, T, L} = MArray{Tuple{S1, S2}, T, 2, L}

@generated function (::Type{MMatrix{S1}})(x::NTuple{L}) where {S1,L}
    S2 = div(L, S1)
    if S1*S2 != L
        throw(DimensionMismatch("Incorrect matrix sizes. $S1 does not divide $L elements"))
    end
    return quote
        $(Expr(:meta, :inline))
        T = eltype(typeof(x))
        MMatrix{S1, $S2, T, L}(x)
    end
end

@generated function (::Type{MMatrix{S1,S2}})(x::NTuple{L}) where {S1,S2,L}
    return quote
        $(Expr(:meta, :inline))
        T = eltype(typeof(x))
        MMatrix{S1, S2, T, L}(x)
    end
end

@generated function (::Type{MMatrix{S1,S2,T}})(x::NTuple{L}) where {S1,S2,T,L}
    return quote
        $(Expr(:meta, :inline))
        MMatrix{S1, S2, T, L}(x)
    end
end

@generated function (::Type{MMatrix{S1,S2,T}})(::UndefInitializer) where {S1,S2,T}
    return quote
        $(Expr(:meta, :inline))
        MMatrix{S1, S2, T, $(S1*S2)}(undef)
    end
end

@inline convert(::Type{MMatrix{S1,S2}}, a::StaticArray{<:Tuple, T}) where {S1,S2,T} = MMatrix{S1,S2,T}(Tuple(a))
@inline MMatrix(a::StaticMatrix{N,M,T}) where {N,M,T} = MMatrix{N,M,T}(Tuple(a))

# Some more advanced constructor-like functions
@inline one(::Type{MMatrix{N}}) where {N} = one(MMatrix{N,N})


#####################
## MMatrix methods ##
#####################

macro MMatrix(ex)
    esc(static_matrix_gen(MMatrix, ex, __module__))
end