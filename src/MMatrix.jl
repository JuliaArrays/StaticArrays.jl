
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
    static_matrix_gen(MMatrix, ex, __module__)
end