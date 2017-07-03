# Originally contributed by D. Getz (https://github.com/getzdan), M. Schauer
# at https://github.com/mschauer/Bridge.jl under MIT License

import Base: getindex,setindex!,==,-,+,*,/,\,transpose,ctranspose,convert, size, abs, real, imag, conj, eye, inv
import Base.LinAlg: ishermitian, issymmetric, isposdef, factorize, diag, trace, det, logdet, expm, logm, sqrtm

@generated function scalem{M, N}(a::StaticMatrix{M,N}, b::StaticVector{N})
    expr = vec([:(a[$j,$i]*b[$i]) for j=1:M, i=1:N])
    :(let val1 = ($(expr[1])); similar_type(SMatrix{M,N},typeof(val1))(val1, $(expr[2:end]...)); end)
end
@generated function scalem{M, N}(a::StaticVector{M}, b::StaticMatrix{M, N})
    expr = vec([:(b[$j,$i]*a[$j]) for j=1:M, i=1:N])
    :(let val1 = ($(expr[1])); similar_type(SMatrix{M,N},typeof(val1))(val1, $(expr[2:end]...)); end)
end

struct SDiagonal{N,T} <: StaticMatrix{N, N, T}
    diag::SVector{N,T}
    SDiagonal{N,T}(diag::SVector{N,T}) where {N,T} = new(diag)
end    
diagtype{N,T}(::Type{SDiagonal{N,T}}) = SVector{N,T}
diagtype{N}(::Type{SDiagonal{N}}) = SVector{N}
diagtype(::Type{SDiagonal}) = SVector

# this is to deal with convert.jl
@inline (::Type{SD})(a::AbstractVector) where {SD <: SDiagonal} = SDiagonal(convert(diagtype(SD), a))
@inline (::Type{SD})(a::Tuple) where {SD <: SDiagonal} = SDiagonal(convert(diagtype(SD), a))
@inline (::Type{SDiagonal}){N,T}(a::SVector{N,T}) = SDiagonal{N,T}(a) 

@generated function SDiagonal{N,T}(a::StaticMatrix{N,N,T})
    expr = [:(a[$i,$i]) for i=1:N]
    :(SDiagonal{N,T}($(expr...)))
end

convert{N,T}(::Type{SDiagonal{N,T}}, D::SDiagonal{N,T}) = D
convert{N,T}(::Type{SDiagonal{N,T}}, D::SDiagonal{N}) = SDiagonal{N,T}(convert(SVector{N,T}, D.diag))

Base.@propagate_inbounds function getindex{N,T}(D::SDiagonal{N,T}, i::Int, j::Int)  
    @boundscheck checkbounds(D, i, j)
    @inbounds return ifelse(i == j, D.diag[i], zero(T))
end

# avoid linear indexing?
Base.@propagate_inbounds function getindex{N,T}(D::SDiagonal{N,T}, k::Int) 
    i, j = ind2sub(size(D), k)
    D[i,j]
end

ishermitian{T<:Real}(D::SDiagonal{T}) = true
ishermitian(D::SDiagonal) = all(D.diag .== real(D.diag))
issymmetric(D::SDiagonal) = true
isposdef(D::SDiagonal) = all(D.diag .> 0)

factorize(D::SDiagonal) = D

==(Da::SDiagonal, Db::SDiagonal) = Da.diag == Db.diag
-(A::SDiagonal) = SDiagonal(-A.diag)
+(Da::SDiagonal, Db::SDiagonal) = SDiagonal(Da.diag + Db.diag)
-(Da::SDiagonal, Db::SDiagonal) = SDiagonal(Da.diag - Db.diag)
-(A::SDiagonal, B::SMatrix) = eye(typeof(B))*A - B

*{T<:Number}(x::T, D::SDiagonal) = SDiagonal(x * D.diag)
*{T<:Number}(D::SDiagonal, x::T) = SDiagonal(D.diag * x)
/{T<:Number}(D::SDiagonal, x::T) = SDiagonal(D.diag / x)
*(Da::SDiagonal, Db::SDiagonal) = SDiagonal(Da.diag .* Db.diag)
*(D::SDiagonal, V::AbstractVector) = D.diag .* V
*(D::SDiagonal, V::StaticVector) = D.diag .* V
*(A::StaticMatrix, D::SDiagonal) = scalem(A,D.diag)
*(D::SDiagonal, A::StaticMatrix) = scalem(D.diag,A)
\(D::SDiagonal, b::AbstractVector) = D.diag .\ b
\(D::SDiagonal, b::StaticVector) = D.diag .\ b # catch ambiguity

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

eye{N,T}(::Type{SDiagonal{N,T}}) = SDiagonal(ones(SVector{N,T}))

expm(D::SDiagonal) = SDiagonal(exp.(D.diag))
logm(D::SDiagonal) = SDiagonal(log.(D.diag))
sqrtm(D::SDiagonal) = SDiagonal(sqrt.(D.diag))

\(D::SDiagonal, B::StaticMatrix) = scalem(1 ./ D.diag, B)
/(B::StaticMatrix, D::SDiagonal) = scalem(1 ./ D.diag, B)
\(Da::SDiagonal, Db::SDiagonal) = SDiagonal(Db.diag ./ Da.diag)
/(Da::SDiagonal, Db::SDiagonal) = SDiagonal(Da.diag ./ Db.diag )


@generated function check_singular{N,T}(D::SDiagonal{N,T})
    expr = Expr(:block)
    for i=1:N
        push!(expr.args, :(@inbounds iszero(D.diag[$i]) && throw(Base.LinAlg.SingularException($i))))
    end
    expr
end

function inv{N,T}(D::SDiagonal{N,T})
    check_singular(D)
    SDiagonal(inv.(D.diag))
end

