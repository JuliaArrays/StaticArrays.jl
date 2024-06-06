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
    # We should be using `T2=eltype(sv)`, but it's not inferable for complex
    # eltypes.  See https://github.com/JuliaLang/julia/pull/22443
    T = eltype(A)
    T2 = promote_type(Float32, real(typeof(one(T)/norm(one(T)))))
    similar_type(A, T2, Size(diagsize(A)))(sv)
end

# `@inline` annotation is required to propagate `full` as constant to `_svd`
@inline svd(A::StaticMatrix; full=Val(false)) = _svd(A, full)

# Allow plain Bool in addition to Val
# Required inline as of version 1.5 to ensure Bool usage like svd(A,
# full=false) is constant-propagated
@inline _svd(A, full) = _svd(A, Val(convert(Bool, full)))

function _svd(A, full::Val{false})
    f = svd(Matrix(A), full=false)
    U = similar_type(A,  eltype(f.U),  Size(Size(A)[1], diagsize(A)))(f.U)
    S = similar_type(A,  eltype(f.S),  Size(diagsize(A)))(f.S)
    Vt = similar_type(A, eltype(f.Vt), Size(diagsize(A), Size(A)[2]))(f.Vt)
    SVD(U,S,Vt)
end

function _svd(A, full::Val{true})
    f = svd(Matrix(A), full=true)
    U = similar_type(A,  eltype(f.U),  Size(Size(A)[1], Size(A)[1]))(f.U)
    S = similar_type(A,  eltype(f.S),  Size(diagsize(A)))(f.S)
    Vt = similar_type(A, eltype(f.Vt), Size(Size(A)[2], Size(A)[2]))(f.Vt)
    SVD(U,S,Vt)
end

function \(F::SVD, B::StaticVecOrMat)
    sthresh = eps(F.S[1])
    Sinv = map(s->s < sthresh ? zero(1/sthresh) : 1/s, F.S)
    return transposemult(F.Vt, diagmult(Sinv, transposemult(F.U, B)))
end

transposemult(U, B) = transposemult(Size(U), Size(B), U, B)
function transposemult(sU, sB, U, B)
    sU[1] == sB[1] && return U'*B
    return U[SOneTo(sB[1]),:]'*B
end
diagmult(d, B) = diagmult(Size(d), Size(B), d, B)
function diagmult(sd, sB, d, B)
    sd[1] == sB[1] && return Diagonal(d)*B
    ind = SOneTo(sd[1])
    return isa(B, AbstractVector) ? Diagonal(d)*B[ind] : Diagonal(d)*B[ind,:]
end

