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

@generated function (::Type{MMatrix{S1,S2,T}})(::UndefInitializer) where {S1,S2,T}
    return quote
        $(Expr(:meta, :inline))
        MMatrix{S1, S2, T, $(S1*S2)}(undef)
    end
end

# Some more advanced constructor-like functions
@inline one(::Type{MMatrix{N}}) where {N} = one(MMatrix{N,N})


#####################
## MMatrix methods ##
#####################

"""
    @MMatrix [a b c d]
    @MMatrix [[a, b];[c, d]]
    @MMatrix [i+j for i in 1:2, j in 1:2]
    @MMatrix ones(2, 2)

A convenience macro to construct `MMatrix`.
See [`@SArray`](@ref) for detailed features.
"""
macro MMatrix(ex)
    esc(static_matrix_gen(MMatrix, ex, __module__))
end