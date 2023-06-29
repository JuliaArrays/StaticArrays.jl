const StaticULT{TA} = Union{UpperTriangular{TA,<:StaticMatrix},LowerTriangular{TA,<:StaticMatrix},UnitUpperTriangular{TA,<:StaticMatrix},UnitLowerTriangular{TA,<:StaticMatrix}}

@inline Base.:*(A::Adjoint{<:Any,<:StaticVector}, B::StaticULT{<:Any}) =
    adjoint(adjoint(B) * adjoint(A))
@inline Base.:*(A::Transpose{<:Any,<:StaticVector}, B::StaticULT{<:Any}) =
    transpose(transpose(B) * transpose(A))
@inline Base.:*(A::StaticULT{<:Any}, B::Adjoint{<:Any,<:StaticVector}) =
    adjoint(adjoint(B) * adjoint(A))
@inline Base.:*(A::StaticULT{<:Any}, B::Transpose{<:Any,<:StaticVector}) =
    transpose(transpose(B) * transpose(A))

@inline Base.:\(A::StaticULT, B::StaticVecOrMatLike) = _A_ldiv_B(Size(A), Size(B), A, B)
@inline Base.:/(A::StaticVecOrMatLike, B::StaticULT) = transpose(transpose(B) \ transpose(A))

@generated function _A_ldiv_B(::Size{sa}, ::Size{sb}, A::StaticULT{TA}, B::StaticVecOrMatLike{TB}) where {sa,sb,TA,TB}
    m = sb[1]
    n = length(sb) > 1 ? sb[2] : 1
    if m != sa[1]
        throw(DimensionMismatch("right hand side B needs first dimension of size $(sa[1]), has size $m"))
    end

    X = [Symbol("X_$(i)_$(j)") for i = 1:m, j = 1:n]

    isunitdiag = A <: Union{UnitUpperTriangular, UnitLowerTriangular}
    isupper = A <: Union{UnitUpperTriangular, UpperTriangular}

    j_range = isupper ? (m:-1:1) : (1:m)

    init = gen_by_access(B, :B) do access_b
        init_exprs = [:($(X[i,j]) = $(uplo_access(sb, :b, i, j, access_b))) for i = 1:m, j = 1:n]
        code = Expr(:block, init_exprs...)
        for k = 1:n
            for j = j_range
                if !isunitdiag && k == 1
                    push!(code.args, :(A.data[$(LinearIndices(sa)[j, j])] == zero(A.data[$(LinearIndices(sa)[j, j])]) && throw(LinearAlgebra.SingularException($j))))
                end
                if isunitdiag
                    push!(code.args, :($(X[j,k]) = oneunit(TA) \ $(X[j,k])))
                else
                    push!(code.args, :($(X[j,k]) = A.data[$(LinearIndices(sa)[j, j])] \ $(X[j,k])))
                end
                i_range = isupper ? (j-1:-1:1) : (j+1:m)
                for i = i_range
                    push!(code.args, :($(X[i,k]) -= A.data[$(LinearIndices(sa)[i, j])]*$(X[j,k])))
                end
            end
        end
        return code
    end

    return quote
        @_inline_meta
        b = mul_parent(B)
        Tb = TB
        @inbounds $init
        TAB = typeof((zero(TA)*zero(TB) + zero(TA)*zero(TB))/one(TA))
        @inbounds return similar_type(B, TAB)(tuple($(X...)))
    end
end

function _first_zero_on_diagonal(A::StaticULT)
    _first_zero_on_diagonal(A.data)
end
