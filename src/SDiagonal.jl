# Originally contributed by D. Getz (https://github.com/getzdan), M. Schauer
# at https://github.com/mschauer/Bridge.jl under MIT License

import Base: ==, -, +, *, /, \, abs, real, imag, conj

const SDiagonal = Diagonal{T,SVector{N,T}} where {N,T}
SDiagonal(x...) = Diagonal(SVector(x...))

# this is to deal with convert.jl
#@inline (::Type{SDiagonal{N,T}})(a::AbstractVector) where {N,T} = Diagonal(SVector{N,T}(a))
@inline (::Type{SDiagonal{N,T}})(a::Tuple) where {N,T} = Diagonal(SVector{N,T}(a))
@inline (::Type{SDiagonal{N}})(a::Tuple) where {N} = Diagonal(SVector{N}(a))

SDiagonal(a::SVector) = Diagonal(a)
SDiagonal(a::StaticMatrix{N,N,T}) where {N,T} = Diagonal(diag(a))

size(::Type{<:SDiagonal{N}}) where {N} = (N,N)
size(::Type{<:SDiagonal{N}}, d::Int) where {N} = d > 2 ? 1 : N

# define specific methods to avoid allocating mutable arrays
*(A::StaticMatrix, D::SDiagonal) = A .* transpose(D.diag)
*(D::SDiagonal, A::StaticMatrix) = D.diag .* A
\(D::SDiagonal, b::AbstractVector) = D.diag .\ b
\(D::SDiagonal, b::StaticVector) = D.diag .\ b # catch ambiguity

\(D::SDiagonal, B::StaticMatrix) = D.diag .\ B
/(B::StaticMatrix, D::SDiagonal) = B ./ transpose(D.diag)
\(Da::SDiagonal, Db::SDiagonal) = SDiagonal(Db.diag ./ Da.diag)
/(Da::SDiagonal, Db::SDiagonal) = SDiagonal(Da.diag ./ Db.diag )

\(D::Diagonal, B::StaticMatrix) = ldiv!(D, Matrix(B))

# override to avoid copying
diag(D::SDiagonal) = D.diag

# SDiagonal(I::UniformScaling) methods to replace eye
(::Type{SDiagonal{N}})(I::UniformScaling) where {N} = SDiagonal{N}(ntuple(x->I.λ, Val(N)))
(::Type{SDiagonal{N,T}})(I::UniformScaling) where {N,T} = SDiagonal{N,T}(ntuple(x->I.λ, Val(N)))

one(::Type{SDiagonal{N,T}}) where {N,T} = SDiagonal(ones(SVector{N,T}))
one(::SDiagonal{N,T}) where {N,T} = SDiagonal(ones(SVector{N,T}))

Base.zero(::SDiagonal{N,T}) where {N,T} = SDiagonal(zeros(SVector{N,T}))

function LinearAlgebra.cholesky(D::SDiagonal)
    any(x -> x < 0, D.diag) && throw(LinearAlgebra.PosDefException(1))
    C = sqrt.(D.diag)
    return Cholesky(SDiagonal(C), 'U', 0)
end

@generated function check_singular(D::SDiagonal{N}) where {N}
    quote
    Base.Cartesian.@nexprs $N i->(@inbounds iszero(D.diag[i]) && throw(LinearAlgebra.SingularException(i)))
    end
end
function inv(D::SDiagonal)
    check_singular(D)
    SDiagonal(inv.(D.diag))
end
