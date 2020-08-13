@inline transpose(A::LinearAlgebra.LowerTriangular{<:Any,<:StaticMatrix}) =
    LinearAlgebra.UpperTriangular(transpose(A.data))
@inline adjoint(A::LinearAlgebra.LowerTriangular{<:Any,<:StaticMatrix}) =
    LinearAlgebra.UpperTriangular(adjoint(A.data))
@inline transpose(A::LinearAlgebra.UpperTriangular{<:Any,<:StaticMatrix}) =
    LinearAlgebra.LowerTriangular(transpose(A.data))
@inline adjoint(A::LinearAlgebra.UpperTriangular{<:Any,<:StaticMatrix}) =
    LinearAlgebra.LowerTriangular(adjoint(A.data))
@inline Base.:*(A::Adjoint{<:Any,<:StaticVector}, B::LinearAlgebra.AbstractTriangular{<:Any,<:StaticMatrix}) =
    adjoint(adjoint(B) * adjoint(A))
@inline Base.:*(A::Transpose{<:Any,<:StaticVector}, B::LinearAlgebra.AbstractTriangular{<:Any,<:StaticMatrix}) =
    transpose(transpose(B) * transpose(A))
@inline Base.:*(A::LinearAlgebra.AbstractTriangular{<:Any,<:StaticMatrix}, B::Adjoint{<:Any,<:StaticVector}) =
    adjoint(adjoint(B) * adjoint(A))
@inline Base.:*(A::LinearAlgebra.AbstractTriangular{<:Any,<:StaticMatrix}, B::Transpose{<:Any,<:StaticVector}) =
    transpose(transpose(B) * transpose(A))

const StaticULT = Union{UpperTriangular{<:Any,<:StaticMatrix},LowerTriangular{<:Any,<:StaticMatrix}}

@inline Base.:\(A::StaticULT, B::StaticVecOrMat) = _A_ldiv_B(Size(A), Size(B), A, B)

@generated function _A_ldiv_B(::Size{sa}, ::Size{sb}, A::UpperTriangular{<:TA,<:StaticMatrix}, B::StaticVecOrMat{TB}) where {sa,sb,TA,TB}
    m = sb[1]
    n = length(sb) > 1 ? sb[2] : 1
    if m != sa[1]
        throw(DimensionMismatch("right hand side B needs first dimension of size $(sa[1]), has size $m"))
    end

    X = [Symbol("X_$(i)_$(j)") for i = 1:m, j = 1:n]
    init = [:($(X[i,j]) = B[$(LinearIndices(sb)[i, j])]) for i = 1:m, j = 1:n]

    code = Expr(:block)
    for k = 1:n
        for j = m:-1:1
            if k == 1
                push!(code.args, :(A.data[$(LinearIndices(sa)[j, j])] == zero(A.data[$(LinearIndices(sa)[j, j])]) && throw(LinearAlgebra.SingularException($j))))
            end
            push!(code.args, :($(X[j,k]) = A.data[$(LinearIndices(sa)[j, j])] \ $(X[j,k])))
            for i = j-1:-1:1
                push!(code.args, :($(X[i,k]) -= A.data[$(LinearIndices(sa)[i, j])]*$(X[j,k])))
            end
        end
    end

    return quote
        @_inline_meta
        @inbounds $(Expr(:block, init...))
        @inbounds $code
        TAB = typeof((zero(TA)*zero(TB) + zero(TA)*zero(TB))/one(TA))
        @inbounds return similar_type(B, TAB)(tuple($(X...)))
    end
end

@generated function _A_ldiv_B(::Size{sa}, ::Size{sb}, A::LowerTriangular{<:TA,<:StaticMatrix}, B::StaticVecOrMat{TB}) where {sa,sb,TA,TB}
    m = sb[1]
    n = length(sb) > 1 ? sb[2] : 1
    if m != sa[1]
        throw(DimensionMismatch("right hand side B needs first dimension of size $(sa[1]), has size $m"))
    end

    X = [Symbol("X_$(i)_$(j)") for i = 1:m, j = 1:n]
    init = [:($(X[i,j]) = B[$(LinearIndices(sb)[i, j])]) for i = 1:m, j = 1:n]

    code = Expr(:block)
    for k = 1:n
        for j = 1:m
            if k == 1
                push!(code.args, :(A.data[$(LinearIndices(sa)[j, j])] == zero(A.data[$(LinearIndices(sa)[j, j])]) && throw(LinearAlgebra.SingularException($j))))
            end
            push!(code.args, :($(X[j,k]) = A.data[$(LinearIndices(sa)[j, j])] \ $(X[j,k])))
            for i = j+1:m
                push!(code.args, :($(X[i,k]) -= A.data[$(LinearIndices(sa)[i, j])]*$(X[j,k])))
            end
        end
    end

    return quote
        @_inline_meta
        @inbounds $(Expr(:block, init...))
        @inbounds $code
        TAB = typeof((zero(TA)*zero(TB) + zero(TA)*zero(TB))/one(TA))
        @inbounds return similar_type(B, TAB)(tuple($(X...)))
    end
end

@generated function _Ac_ldiv_B(::Size{sa}, ::Size{sb}, A::UpperTriangular{<:TA,<:StaticMatrix}, B::StaticVecOrMat{TB}) where {sa,sb,TA,TB}
    m = sb[1]
    n = length(sb) > 1 ? sb[2] : 1
    if m != sa[1]
        throw(DimensionMismatch("right hand side B needs first dimension of size $(sa[1]), has size $m"))
    end

    X = [Symbol("X_$(i)_$(j)") for i = 1:m, j = 1:n]

    code = Expr(:block)
    for k = 1:n
        for j = 1:m
            ex = :(B[$(LinearIndices(sb)[j, k])])
            for i = 1:j-1
                ex = :($ex - A.data[$(LinearIndices(sa)[i, j])]'*$(X[i,k]))
            end
            if k == 1
                push!(code.args, :(A.data[$(LinearIndices(sa)[j, j])] == zero(A.data[$(LinearIndices(sa)[j, j])]) && throw(LinearAlgebra.SingularException($j))))
            end
            push!(code.args, :($(X[j,k]) = A.data[$(LinearIndices(sa)[j, j])]' \ $ex))
        end
    end

    return quote
        @_inline_meta
        @inbounds $code
        TAB = typeof((zero(TA)*zero(TB) + zero(TA)*zero(TB))/one(TA))
        @inbounds return similar_type(B, TAB)(tuple($(X...)))
    end
end

@generated function _At_ldiv_B(::Size{sa}, ::Size{sb}, A::UpperTriangular{<:TA,<:StaticMatrix}, B::StaticVecOrMat{TB}) where {sa,sb,TA,TB}
    m = sb[1]
    n = length(sb) > 1 ? sb[2] : 1
    if m != sa[1]
        throw(DimensionMismatch("right hand side B needs first dimension of size $(sa[1]), has size $m"))
    end

    X = [Symbol("X_$(i)_$(j)") for i = 1:m, j = 1:n]

    code = Expr(:block)
    for k = 1:n
        for j = 1:m
            ex = :(B[$(LinearIndices(sb)[j, k])])
            for i = 1:j-1
                ex = :($ex - A.data[$(LinearIndices(sa)[i, j])]*$(X[i,k]))
            end
            if k == 1
                push!(code.args, :(A.data[$(LinearIndices(sa)[j, j])] == zero(A.data[$(LinearIndices(sa)[j, j])]) && throw(LinearAlgebra.SingularException($j))))
            end
            push!(code.args, :($(X[j,k]) = A.data[$(LinearIndices(sa)[j, j])] \ $ex))
        end
    end

    return quote
        @_inline_meta
        @inbounds $code
        TAB = typeof((zero(TA)*zero(TB) + zero(TA)*zero(TB))/one(TA))
        @inbounds return similar_type(B, TAB)(tuple($(X...)))
    end
end

@generated function _Ac_ldiv_B(::Size{sa}, ::Size{sb}, A::LowerTriangular{<:TA,<:StaticMatrix}, B::StaticVecOrMat{TB}) where {sa,sb,TA,TB}
    m = sb[1]
    n = length(sb) > 1 ? sb[2] : 1
    if m != sa[1]
        throw(DimensionMismatch("right hand side B needs first dimension of size $(sa[1]), has size $m"))
    end

    X = [Symbol("X_$(i)_$(j)") for i = 1:m, j = 1:n]

    code = Expr(:block)
    for k = 1:n
        for j = m:-1:1
            ex = :(B[$(LinearIndices(sb)[j, k])])
            for i = m:-1:j+1
                ex = :($ex - A.data[$(LinearIndices(sa)[i, j])]'*$(X[i,k]))
            end
            if k == 1
                push!(code.args, :(A.data[$(LinearIndices(sa)[j, j])] == zero(A.data[$(LinearIndices(sa)[j, j])]) && throw(LinearAlgebra.SingularException($j))))
            end
            push!(code.args, :($(X[j,k]) = A.data[$(LinearIndices(sa)[j, j])]' \ $ex))
        end
    end

    return quote
        @_inline_meta
        @inbounds $code
        TAB = typeof((zero(TA)*zero(TB) + zero(TA)*zero(TB))/one(TA))
        @inbounds return similar_type(B, TAB)(tuple($(X...)))
    end
end

@generated function _At_ldiv_B(::Size{sa}, ::Size{sb}, A::LowerTriangular{<:TA,<:StaticMatrix}, B::StaticVecOrMat{TB}) where {sa,sb,TA,TB}
    m = sb[1]
    n = length(sb) > 1 ? sb[2] : 1
    if m != sa[1]
        throw(DimensionMismatch("right hand side B needs first dimension of size $(sa[1]), has size $m"))
    end

    X = [Symbol("X_$(i)_$(j)") for i = 1:m, j = 1:n]

    code = Expr(:block)
    for k = 1:n
        for j = m:-1:1
            ex = :(B[$(LinearIndices(sb)[j, k])])
            for i = m:-1:j+1
                ex = :($ex - A.data[$(LinearIndices(sa)[i, j])]*$(X[i,k]))
            end
            if k == 1
                push!(code.args, :(A.data[$(LinearIndices(sa)[j, j])] == zero(A.data[$(LinearIndices(sa)[j, j])]) && throw(LinearAlgebra.SingularException($j))))
            end
            push!(code.args, :($(X[j,k]) = A.data[$(LinearIndices(sa)[j, j])] \ $ex))
        end
    end

    return quote
        @_inline_meta
        @inbounds $code
        TAB = typeof((zero(TA)*zero(TB) + zero(TA)*zero(TB))/one(TA))
        @inbounds return similar_type(B, TAB)(tuple($(X...)))
    end
end
