# LU decomposition
function lu(A::StaticMatrix, pivot::Union{Type{Val{false}},Type{Val{true}}}=Val{true})
    L,U,p = _lu(Size(A), A, pivot)
    (L,U,p)
end

# For the square version, return explicit lower and upper triangular matrices.
# We would do this for the rectangular case too, but Base doesn't support that.
function lu(A::StaticMatrix{N,N}, pivot::Union{Type{Val{false}},Type{Val{true}}}=Val{true}) where {N}
    L,U,p = _lu(Size(A), A, pivot)
    (LowerTriangular(L), UpperTriangular(U), p)
end


@inline function _lu(::Size{S}, A::StaticMatrix, pivot) where {S}
    # For now, just call through to Base.
    # TODO: statically sized LU without allocations!
    f = lufact(Matrix(A), pivot)
    T = eltype(A)
    # Trick to get the output eltype - can't rely on the result of f[:L] as
    # it's not type inferrable.
    T2 = typeof((one(T)*zero(T) + zero(T))/one(T))
    L = similar_type(A, T2, Size(Size(A)[1], diagsize(A)))(f[:L])
    U = similar_type(A, T2, Size(diagsize(A), Size(A)[2]))(f[:U])
    p = similar_type(A, Int, Size(Size(A)[1]))(f[:p])
    (L,U,p)
end

# Base.lufact() interface is fairly inherently type unstable.  Punt on
# implementing that, for now...
