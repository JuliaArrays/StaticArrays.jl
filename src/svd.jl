# Singular Value Decomposition

# We need our own SVD factorization struct, as Base.LinAlg.SVD assumes
# Base.Vector for `S`, and that the `U` and `Vt` have the same
struct SVD{T,TU,TS,TVt} <: Factorization{T}
    U::TU
    S::TS
    Vt::TVt
end
SVD(U::AbstractArray{T}, S::AbstractVector, Vt::AbstractArray{T}) where {T} = SVD{T,typeof(U),typeof(S),typeof(Vt)}(U, S, Vt)

getindex(::SVD, ::Symbol) = error("In order to avoid type instability, StaticArrays.SVD doesn't support indexing the output of svdfact with a symbol.  Instead, you can access the fields of the factorization directly as f.U, f.S, and f.Vt")

function svdvals(A::StaticMatrix)
    sv = svdvals(Matrix(A))
    # We should be using `T2=eltype(sv)`, but it's not inferrable for complex
    # eltypes.  See https://github.com/JuliaLang/julia/pull/22443
    T = eltype(A)
    T2 = promote_type(Float32, real(typeof(one(T)/norm(one(T)))))
    similar_type(A, T2, Size(diagsize(A)))(sv)
end

function svdfact(A::StaticMatrix)
    # "Thin" SVD only for now.
    f = svdfact(Matrix(A))
    U = similar_type(A, eltype(f.U), Size(Size(A)[1], diagsize(A)))(f.U)
    S = similar_type(A, eltype(f.S), Size(diagsize(A)))(f.S)
    Vt = similar_type(A, eltype(f.Vt), Size(diagsize(A), Size(A)[2]))(f.Vt)
    SVD(U,S,Vt)
end

function svd(A::StaticMatrix)
    # Need our own version of `svd()`, as `Base` passes the `thin` argument
    # which makes the resulting dimensions uninferrable.
    f = svdfact(A)
    (f.U, f.S, f.Vt')
end
