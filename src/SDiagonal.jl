# Originally contributed by D. Getz (https://github.com/getzdan), M. Schauer
# at https://github.com/mschauer/Bridge.jl under MIT License

import Base: getindex,setindex!,==,-,+,*,/,\,transpose,ctranspose,convert, size, abs, real, imag, conj, eye, inv
import Base.LinAlg: ishermitian, issymmetric, isposdef, factorize, diag, trace, det, logdet, expm, logm, sqrtm

@generated function scalem{T, M, N}(a::SMatrix{M,N, T}, b::SVector{N, T})
    expr = vec([:(a[$j,$i]*b[$i]) for j=1:M, i=1:N])
    :(SMatrix{M,N,T}($(expr...)))
end
@generated function scalem{T, M, N}(a::SVector{M,T}, b::SMatrix{M, N, T})
    expr = vec([:(b[$j,$i]*a[$j]) for j=1:M, i=1:N])
    :(SMatrix{M,N,T}($(expr...)))
end

struct SDiagonal{N,T} <: StaticMatrix{N, N, T}
    diag::SVector{N,T}
    SDiagonal{N,T}(diag::SVector{N,T}) where {N,T} = new(diag)
end    

# this is to deal with convert.jl
@inline (::Type{SDiagonal})(a::AbstractVector) = SDiagonal(SVector(a)) 
@inline (::Type{SDiagonal}){N,T}(a::SVector{N,T}) = SDiagonal{N,T}(a) 

@generated function SDiagonal{N,T}(a::SMatrix{N,N,T})
    expr = [:(a[$i,$i]) for i=1:N]
    :(SDiagonal{N,T}($(expr...)))
end

function \{T,M}(D::SDiagonal, b::SVector{M,T} )
    D.diag .* b
end

convert{N,T}(::Type{SDiagonal{N,T}}, D::SDiagonal{N,T}) = D
convert{N,T}(::Type{SDiagonal{N,T}}, D::SDiagonal) = SDiagonal{N,T}(convert(SVector{N,T}, D.diag))

size(D::SDiagonal) = (length(D.diag),length(D.diag))

function size(D::SDiagonal,d::Integer)
    if d<1
        throw(ArgumentError("dimension must be ≥ 1, got $d"))
    end
    return d<=2 ? length(D.diag) : 1
end

function getindex{T}(D::SDiagonal{T}, i::Int, j::Int)  
    if i == j
        D.diag[i]
    else
        zero(T)
    end
end


ishermitian{T<:Real}(D::SDiagonal{T}) = true
ishermitian(D::SDiagonal) = all(D.diag .== real(D.diag))
issym(D::SDiagonal) = true
isposdef(D::SDiagonal) = all(D.diag .> 0)

factorize(D::SDiagonal) = D

abs(D::SDiagonal) = SDiagonal(abs(D.diag))
real(D::SDiagonal) = SDiagonal(real(D.diag))
imag(D::SDiagonal) = SDiagonal(imag(D.diag))

==(Da::SDiagonal, Db::SDiagonal) = Da.diag == Db.diag
-(A::SDiagonal) = SDiagonal(-A.diag)
+(Da::SDiagonal, Db::SDiagonal) = SDiagonal(Da.diag + Db.diag)
-(Da::SDiagonal, Db::SDiagonal) = SDiagonal(Da.diag - Db.diag)
-(A::SDiagonal, B::SMatrix) = eye(typeof(B))*A - B


*{T<:Number}(x::T, D::SDiagonal) = SDiagonal(x * D.diag)
*{T<:Number}(D::SDiagonal, x::T) = SDiagonal(D.diag * x)
/{T<:Number}(D::SDiagonal, x::T) = SDiagonal(D.diag / x)
*(Da::SDiagonal, Db::SDiagonal) = SDiagonal(Da.diag .* Db.diag)
*(D::SDiagonal, V::SVector) = D.diag .* V
*(V::SVector, D::SDiagonal) = D.diag .* V
*(A::SMatrix, D::SDiagonal) = scalem(A,D.diag)
*(D::SDiagonal, A::SMatrix) = scalem(D.diag,A)

/(Da::SDiagonal, Db::SDiagonal) = SDiagonal(Da.diag ./ Db.diag )

conj(D::SDiagonal) = SDiagonal(conj(D.diag))
transpose(D::SDiagonal) = D
ctranspose(D::SDiagonal) = conj(D)

diag(D::SDiagonal) = D.diag
trace(D::SDiagonal) = sum(D.diag)
det(D::SDiagonal) = prod(D.diag)
logdet{N,T<:Real}(D::SDiagonal{N,T}) = sum(log.(D.diag))
function logdet{N,T<:Complex}(D::SDiagonal{N,T}) #Make sure branch cut is correct
    x = sum(log.(D.diag))
    -pi<imag(x)<pi ? x : real(x)+(mod2pi(imag(x)+pi)-pi)*im
end


eye{N,T}(::Type{SDiagonal{N,T}}) = SDiagonal(one(SVector{n,Int}))

expm(D::SDiagonal) = SDiagonal(exp.(D.diag))
logm(D::SDiagonal) = SDiagonal(log.(D.diag))
sqrtm(D::SDiagonal) = SDiagonal(sqrt.(D.diag))

\(D::SDiagonal, B::SMatrix) = scalem(1 ./ D.diag, B)
/(B::SMatrix, D::SDiagonal) = scalem(1 ./ D.diag, B)
\(Da::SDiagonal, Db::SDiagonal) = SDiagonal(Db.diag ./ Da.diag)

function inv{N,T}(D::SDiagonal{N,T})
    for i = 1:length(D.diag)
        if D.diag[i] == zero(T)
            throw(SingularException(i))
        end
    end
    SDiagonal(one(T)./D.diag)
end

