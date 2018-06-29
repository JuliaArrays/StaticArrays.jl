# Singular Value Decomposition

# We need our own SVD factorization struct, as LinearAlgebra.SVD assumes
# Base.Vector for `S`, and that the `U` and `Vt` have the same
struct SVD{T,TU,TS,TVt} <: Factorization{T}
    U::TU
    S::TS
    Vt::TVt
end
SVD(U::AbstractArray{T}, S::AbstractVector, Vt::AbstractArray{T}) where {T} = SVD{T,typeof(U),typeof(S),typeof(Vt)}(U, S, Vt)

@inline function Base.getproperty(F::SVD, s::Symbol)
    if s === :V
        return getfield(F, :Vt)'
    else
        return getfield(F, s)
    end
end

# iteration for destructuring into components
Base.iterate(S::SVD) = (S.U, Val(:S))
Base.iterate(S::SVD, ::Val{:S}) = (S.S, Val(:V))
Base.iterate(S::SVD, ::Val{:V}) = (S.V, Val(:done))
Base.iterate(S::SVD, ::Val{:done}) = nothing

function svdvals(A::StaticMatrix)
    sv = svdvals(Matrix(A))
    # We should be using `T2=eltype(sv)`, but it's not inferrable for complex
    # eltypes.  See https://github.com/JuliaLang/julia/pull/22443
    T = eltype(A)
    T2 = promote_type(Float32, real(typeof(one(T)/norm(one(T)))))
    similar_type(A, T2, Size(diagsize(A)))(sv)
end

function svd(A::StaticMatrix)
    # "Thin" SVD only for now.
    f = svd(Matrix(A))
    U = similar_type(A, eltype(f.U), Size(Size(A)[1], diagsize(A)))(f.U)
    S = similar_type(A, eltype(f.S), Size(diagsize(A)))(f.S)
    Vt = similar_type(A, eltype(f.Vt), Size(diagsize(A), Size(A)[2]))(f.Vt)
    SVD(U,S,Vt)
end
