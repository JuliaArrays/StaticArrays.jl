# Originally contributed by D. Getz (https://github.com/getzdan), M. Schauer
# at https://github.com/mschauer/Bridge.jl under MIT License

import Base: ==, -, +, *, /, \, abs, real, imag, conj

@generated function scalem(a::StaticMatrix{M,N}, b::StaticVector{N}) where {M, N}
    expr = vec([:(a[$j,$i]*b[$i]) for j=1:M, i=1:N])
    :(@_inline_meta; let val1 = ($(expr[1])); similar_type(SMatrix{M,N},typeof(val1))(val1, $(expr[2:end]...)); end)
end
@generated function scalem(a::StaticVector{M}, b::StaticMatrix{M, N}) where {M, N}
    expr = vec([:(b[$j,$i]*a[$j]) for j=1:M, i=1:N])
    :(@_inline_meta; let val1 = ($(expr[1])); similar_type(SMatrix{M,N},typeof(val1))(val1, $(expr[2:end]...)); end)
end

struct SDiagonal{N,T} <: StaticMatrix{N,N,T}
    diag::SVector{N,T}
    SDiagonal{N,T}(diag::SVector{N,T}) where {N,T} = new(diag)
end
diagtype(::Type{SDiagonal{N,T}}) where {N, T} = SVector{N,T}
diagtype(::Type{SDiagonal{N}}) where {N} = SVector{N}
diagtype(::Type{SDiagonal}) = SVector

# this is to deal with convert.jl
@inline (::Type{SD})(a::AbstractVector) where {SD <: SDiagonal} = SDiagonal(convert(diagtype(SD), a))
@inline (::Type{SD})(a::Tuple) where {SD <: SDiagonal} = SDiagonal(convert(diagtype(SD), a))
@inline SDiagonal(a::SVector{N,T}) where {N,T} = SDiagonal{N,T}(a)

@generated function SDiagonal(a::StaticMatrix{N,N,T}) where {N,T}
    expr = [:(a[$i,$i]) for i=1:N]
    :(SDiagonal{N,T}($(expr...)))
end

convert(::Type{SDiagonal{N,T}}, D::SDiagonal{N,T}) where {N,T} = D
convert(::Type{SDiagonal{N,T}}, D::SDiagonal{N}) where {N,T} = SDiagonal{N,T}(convert(SVector{N,T}, D.diag))

function getindex(D::SDiagonal{N,T}, i::Int, j::Int) where {N,T}
    @boundscheck checkbounds(D, i, j)
    @inbounds return ifelse(i == j, D.diag[i], zero(T))
end

# avoid linear indexing?
@propagate_inbounds function getindex(D::SDiagonal{N,T}, k::Int) where {N,T}
    i, j = CartesianIndices(size(D))[k].I
    D[i,j]
end

ishermitian(D::SDiagonal{N, T}) where {N,T<:Real} = true
ishermitian(D::SDiagonal) = all(D.diag .== real(D.diag))
issymmetric(D::SDiagonal) = true
isposdef(D::SDiagonal) = all(D.diag .> 0)

factorize(D::SDiagonal) = D

==(Da::SDiagonal, Db::SDiagonal) = Da.diag == Db.diag
-(A::SDiagonal) = SDiagonal(-A.diag)
+(Da::SDiagonal, Db::SDiagonal) = SDiagonal(Da.diag + Db.diag)
-(Da::SDiagonal, Db::SDiagonal) = SDiagonal(Da.diag - Db.diag)
-(A::SDiagonal, B::SMatrix) = typeof(B)(I)*A - B

*(x::T, D::SDiagonal) where {T<:Number} = SDiagonal(x * D.diag)
*(D::SDiagonal, x::T) where {T<:Number} = SDiagonal(D.diag * x)
/(D::SDiagonal, x::T) where {T<:Number} = SDiagonal(D.diag / x)
*(Da::SDiagonal, Db::SDiagonal) = SDiagonal(Da.diag .* Db.diag)
*(D::SDiagonal, V::AbstractVector) = D.diag .* V
*(D::SDiagonal, V::StaticVector) = D.diag .* V
*(A::StaticMatrix, D::SDiagonal) = scalem(A,D.diag)
*(D::SDiagonal, A::StaticMatrix) = scalem(D.diag,A)
\(D::SDiagonal, b::AbstractVector) = D.diag .\ b
\(D::SDiagonal, b::StaticVector) = D.diag .\ b # catch ambiguity

conj(D::SDiagonal) = SDiagonal(conj(D.diag))
transpose(D::SDiagonal) = D
adjoint(D::SDiagonal) = conj(D)

diag(D::SDiagonal) = D.diag
tr(D::SDiagonal) = sum(D.diag)
det(D::SDiagonal) = prod(D.diag)
logdet(D::SDiagonal{N,T}) where {N,T<:Real} = sum(log.(D.diag))
function logdet(D::SDiagonal{N,T}) where {N,T<:Complex} #Make sure branch cut is correct
    x = sum(log.(D.diag))
    -pi<imag(x)<pi ? x : real(x)+(mod2pi(imag(x)+pi)-pi)*im
end

# SDiagonal(I::UniformScaling) methods to replace eye
(::Type{SD})(I::UniformScaling) where {N,SD<:SDiagonal{N}} = SD(ntuple(x->I.λ, Val(N)))
# deprecate eye, keep around for as long as LinearAlgebra.eye exists
@static if isdefined(LinearAlgebra, :eye)
    @deprecate eye(::Type{SDiagonal{N,T}}) where {N,T} SDiagonal{N,T}(I)
end

one(::Type{SDiagonal{N,T}}) where {N,T} = SDiagonal(ones(SVector{N,T}))
one(::SDiagonal{N,T}) where {N,T} = SDiagonal(ones(SVector{N,T}))
Base.zero(::SDiagonal{N,T}) where {N,T} = SDiagonal(zeros(SVector{N,T}))
exp(D::SDiagonal) = SDiagonal(exp.(D.diag))
log(D::SDiagonal) = SDiagonal(log.(D.diag))
sqrt(D::SDiagonal) = SDiagonal(sqrt.(D.diag))
function LinearAlgebra.cholesky(D::SDiagonal)
    any(x -> x < 0, D.diag) && throw(LinearAlgebra.PosDefException(1))
    C = sqrt.(D.diag)
    return Cholesky(SDiagonal(C), 'U', 0)
end

\(D::SDiagonal, B::StaticMatrix) = scalem(1 ./ D.diag, B)
/(B::StaticMatrix, D::SDiagonal) = scalem(1 ./ D.diag, B)
\(Da::SDiagonal, Db::SDiagonal) = SDiagonal(Db.diag ./ Da.diag)
/(Da::SDiagonal, Db::SDiagonal) = SDiagonal(Da.diag ./ Db.diag )

@generated function check_singular(D::SDiagonal{N}) where {N}
    quote
    Base.Cartesian.@nexprs $N i->(@inbounds iszero(D.diag[i]) && throw(LinearAlgebra.SingularException(i)))
    end
end

function inv(D::SDiagonal)
    check_singular(D)
    SDiagonal(inv.(D.diag))
end
