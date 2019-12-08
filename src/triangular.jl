@inline transpose(A::LinearAlgebra.LowerTriangular{<:Any,<:StaticMatrix}) =
    LinearAlgebra.UpperTriangular(transpose(A.data))
@inline adjoint(A::LinearAlgebra.LowerTriangular{<:Any,<:StaticMatrix}) =
    LinearAlgebra.UpperTriangular(adjoint(A.data))
@inline transpose(A::LinearAlgebra.UpperTriangular{<:Any,<:StaticMatrix}) =
    LinearAlgebra.LowerTriangular(transpose(A.data))
@inline adjoint(A::LinearAlgebra.UpperTriangular{<:Any,<:StaticMatrix}) =
    LinearAlgebra.LowerTriangular(adjoint(A.data))
@inline Base.:*(A::Adjoint{<:Any,<:StaticVecOrMat}, B::LinearAlgebra.AbstractTriangular{<:Any,<:StaticMatrix}) =
    adjoint(adjoint(B) * adjoint(A))
@inline Base.:*(A::Transpose{<:Any,<:StaticVecOrMat}, B::LinearAlgebra.AbstractTriangular{<:Any,<:StaticMatrix}) =
    transpose(transpose(B) * transpose(A))
@inline Base.:*(A::LinearAlgebra.AbstractTriangular{<:Any,<:StaticMatrix}, B::Adjoint{<:Any,<:StaticVecOrMat}) =
    adjoint(adjoint(B) * adjoint(A))
@inline Base.:*(A::LinearAlgebra.AbstractTriangular{<:Any,<:StaticMatrix}, B::Transpose{<:Any,<:StaticVecOrMat}) =
    transpose(transpose(B) * transpose(A))

const StaticULT = Union{UpperTriangular{<:Any,<:StaticMatrix},LowerTriangular{<:Any,<:StaticMatrix}}

@inline Base.:*(A::LinearAlgebra.AbstractTriangular{<:Any,<:StaticMatrix}, B::StaticVecOrMat) = _A_mul_B(Size(A), Size(B), A, B)
@inline Base.:*(A::StaticVecOrMat, B::LinearAlgebra.AbstractTriangular{<:Any,<:StaticMatrix}) = _A_mul_B(Size(A), Size(B), A, B)
@inline Base.:*(A::StaticULT, B::StaticULT) = _A_mul_B(Size(A), Size(B), A, B)
@inline Base.:\(A::StaticULT, B::StaticVecOrMat) = _A_ldiv_B(Size(A), Size(B), A, B)


@generated function _A_mul_B(::Size{sa}, ::Size{sb}, A::UpperTriangular{TA,<:StaticMatrix}, B::StaticVecOrMat{TB}) where {sa,sb,TA,TB}
    m = sb[1]
    n = length(sb) > 1 ? sb[2] : 1
    if m != sa[1]
        throw(DimensionMismatch("right hand side B needs first dimension of size $(sa[1]), has size $m"))
    end

    X = [Symbol("X_$(i)_$(j)") for i = 1:m, j = 1:n]

    code = Expr(:block)
    for j = 1:n
        for i = 1:m
            ex = :(A.data[$(LinearIndices(sa)[i, i])]*B[$(LinearIndices(sb)[i, j])])
            for k = i+1:m
                ex = :($ex + A.data[$(LinearIndices(sa)[i, k])]*B[$(LinearIndices(sb)[k, j])])
            end
            push!(code.args, :($(X[i,j]) = $ex))
        end
    end

    return quote
        @_inline_meta
        @inbounds $code
        TAB = promote_op(matprod, TA, TB)
        return similar_type(B, TAB)(tuple($(X...)))
    end
end

@generated function _Ac_mul_B(::Size{sa}, ::Size{sb}, A::UpperTriangular{TA,<:StaticMatrix}, B::StaticVecOrMat{TB}) where {sa,sb,TA,TB}
    m = sb[1]
    n = length(sb) > 1 ? sb[2] : 1
    if m != sa[1]
        throw(DimensionMismatch("right hand side B needs first dimension of size $(sa[1]), has size $m"))
    end

    X = [Symbol("X_$(i)_$(j)") for i = 1:m, j = 1:n]

    code = Expr(:block)
    for j = 1:n
        for i = m:-1:1
            ex = :(A.data[$(LinearIndices(sa)[i, i])]'*B[$(LinearIndices(sb)[i, j])])
            for k = 1:i-1
                ex = :($ex + A.data[$(LinearIndices(sa)[k, i])]'*B[$(LinearIndices(sb)[k, j])])
            end
            push!(code.args, :($(X[i,j]) = $ex))
        end
    end

    return quote
        @_inline_meta
        @inbounds $code
        TAB = promote_op(matprod, TA, TB)
        return similar_type(B, TAB)(tuple($(X...)))
    end
end

@generated function _At_mul_B(::Size{sa}, ::Size{sb}, A::UpperTriangular{TA,<:StaticMatrix}, B::StaticVecOrMat{TB}) where {sa,sb,TA,TB}
    m = sb[1]
    n = length(sb) > 1 ? sb[2] : 1
    if m != sa[1]
        throw(DimensionMismatch("right hand side B needs first dimension of size $(sa[1]), has size $m"))
    end

    X = [Symbol("X_$(i)_$(j)") for i = 1:m, j = 1:n]

    code = Expr(:block)
    for j = 1:n
        for i = m:-1:1
            ex = :(transpose(A.data[$(LinearIndices(sa)[i, i])])*B[$(LinearIndices(sb)[i, j])])
            for k = 1:i-1
                ex = :($ex + transpose(A.data[$(LinearIndices(sa)[k, i])])*B[$(LinearIndices(sb)[k, j])])
            end
            push!(code.args, :($(X[i,j]) = $ex))
        end
    end

    return quote
        @_inline_meta
        @inbounds $code
        TAB = promote_op(matprod, TA, TB)
        return similar_type(B, TAB)(tuple($(X...)))
    end
end

@generated function _A_mul_B(::Size{sa}, ::Size{sb}, A::LowerTriangular{TA,<:StaticMatrix}, B::StaticVecOrMat{TB}) where {sa,sb,TA,TB}
    m = sb[1]
    n = length(sb) > 1 ? sb[2] : 1
    if m != sa[1]
        throw(DimensionMismatch("right hand side B needs first dimension of size $(sa[1]), has size $m"))
    end

    X = [Symbol("X_$(i)_$(j)") for i = 1:m, j = 1:n]

    code = Expr(:block)
    for j = 1:n
        for i = m:-1:1
            ex = :(A.data[$(LinearIndices(sa)[i, i])]*B[$(LinearIndices(sb)[i, j])])
            for k = 1:i-1
                ex = :($ex + A.data[$(LinearIndices(sa)[i, k])]*B[$(LinearIndices(sb)[k, j])])
            end
            push!(code.args, :($(X[i,j]) = $ex))
        end
    end

    return quote
        @_inline_meta
        @inbounds $code
        TAB = promote_op(matprod, TA, TB)
        return similar_type(B, TAB)(tuple($(X...)))
    end
end

@generated function _Ac_mul_B(::Size{sa}, ::Size{sb}, A::LowerTriangular{TA,<:StaticMatrix}, B::StaticVecOrMat{TB}) where {sa,sb,TA,TB}
    m = sb[1]
    n = length(sb) > 1 ? sb[2] : 1
    if m != sa[1]
        throw(DimensionMismatch("right hand side B needs first dimension of size $(sa[1]), has size $m"))
    end

    X = [Symbol("X_$(i)_$(j)") for i = 1:m, j = 1:n]

    code = Expr(:block)
    for j = 1:n
        for i = 1:m
            ex = :(A.data[$(LinearIndices(sa)[i, i])]'*B[$(LinearIndices(sb)[i, j])])
            for k = i+1:m
                ex = :($ex + A.data[$(LinearIndices(sa)[k, i])]'*B[$(LinearIndices(sb)[k, j])])
            end
            push!(code.args, :($(X[i,j]) = $ex))
        end
    end

    return quote
        @_inline_meta
        @inbounds $code
        TAB = promote_op(matprod, TA, TB)
        return similar_type(B, TAB)(tuple($(X...)))
    end
end

@generated function _At_mul_B(::Size{sa}, ::Size{sb}, A::LowerTriangular{TA,<:StaticMatrix}, B::StaticVecOrMat{TB}) where {sa,sb,TA,TB}
    m = sb[1]
    n = length(sb) > 1 ? sb[2] : 1
    if m != sa[1]
        throw(DimensionMismatch("right hand side B needs first dimension of size $(sa[1]), has size $m"))
    end

    X = [Symbol("X_$(i)_$(j)") for i = 1:m, j = 1:n]

    code = Expr(:block)
    for j = 1:n
        for i = 1:m
            ex = :(transpose(A.data[$(LinearIndices(sa)[i, i])])*B[$(LinearIndices(sb)[i, j])])
            for k = i+1:m
                ex = :($ex + transpose(A.data[$(LinearIndices(sa)[k, i])])*B[$(LinearIndices(sb)[k, j])])
            end
            push!(code.args, :($(X[i,j]) = $ex))
        end
    end

    return quote
        @_inline_meta
        @inbounds $code
        TAB = promote_op(matprod, TA, TB)
        return similar_type(B, TAB)(tuple($(X...)))
    end
end

@generated function _A_mul_B(::Size{sa}, ::Size{sb}, A::StaticArray{<:Tuple,TA}, B::UpperTriangular{TB,<:StaticMatrix}) where {sa,sb,TA,TB}
    m = sa[1]
    if length(sa) == 1
        n = 1
    else
        n = sa[2]
    end
    if sb[1] != n
        throw(DimensionMismatch("right hand side B needs first dimension of size $n, has size $(sb[1])"))
    end

    X = [Symbol("X_$(i)_$(j)") for i = 1:m, j = 1:n]

    code = Expr(:block)
    for i = 1:m
        for j = n:-1:1
            ex = :(A[$(LinearIndices(sa)[i, j])]*B[$(LinearIndices(sb)[j, j])])
            for k = 1:j-1
                ex = :($ex + A[$(LinearIndices(sa)[i, k])]*B.data[$(LinearIndices(sb)[k, j])])
            end
            push!(code.args, :($(X[i,j]) = $ex))
        end
    end

    return quote
        @_inline_meta
        @inbounds $code
        TAB = promote_op(matprod, TA, TB)
        return similar_type(A, TAB, Size($m,$n))(tuple($(X...)))
    end
end

@generated function _A_mul_Bc(::Size{sa}, ::Size{sb}, A::StaticArray{<:Tuple,TA}, B::UpperTriangular{TB,<:StaticMatrix}) where {sa,sb,TA,TB}
    m = sa[1]
    if length(sa) == 1
        n = 1
    else
        n = sa[2]
    end
    if sb[1] != n
        throw(DimensionMismatch("right hand side B needs first dimension of size $n, has size $(sb[1])"))
    end

    X = [Symbol("X_$(i)_$(j)") for i = 1:m, j = 1:n]

    code = Expr(:block)
    for i = 1:m
        for j = 1:n
            ex = :(A[$(LinearIndices(sa)[i, j])]*B[$(LinearIndices(sb)[j, j])]')
            for k = j+1:n
                ex = :($ex + A[$(LinearIndices(sa)[i, k])]*B.data[$(LinearIndices(sb)[j, k])]')
            end
            push!(code.args, :($(X[i,j]) = $ex))
        end
    end

    return quote
        @_inline_meta
        @inbounds $code
        TAB = promote_op(matprod, TA, TB)
        return similar_type(A, TAB, Size($m, $n))(tuple($(X...)))
    end
end

@generated function _A_mul_Bt(::Size{sa}, ::Size{sb}, A::StaticMatrix{<:Any,<:Any,TA}, B::UpperTriangular{TB,<:StaticMatrix}) where {sa,sb,TA,TB}
    m, n = sa[1], sa[2]
    if sb[1] != n
        throw(DimensionMismatch("right hand side B needs first dimension of size $n, has size $(sb[1])"))
    end

    X = [Symbol("X_$(i)_$(j)") for i = 1:m, j = 1:n]

    code = Expr(:block)
    for i = 1:m
        for j = 1:n
            ex = :(A[$(LinearIndices(sa)[i, j])]*transpose(B[$(LinearIndices(sb)[j, j])]))
            for k = j+1:n
                ex = :($ex + A[$(LinearIndices(sa)[i, k])]*transpose(B.data[$(LinearIndices(sb)[j, k])]))
            end
            push!(code.args, :($(X[i,j]) = $ex))
        end
    end

    return quote
        @_inline_meta
        @inbounds $code
        TAB = promote_op(matprod, TA, TB)
        return similar_type(A, TAB)(tuple($(X...)))
    end
end

@generated function _A_mul_B(::Size{sa}, ::Size{sb}, A::StaticArray{<:Tuple,TA}, B::LowerTriangular{TB,<:StaticMatrix}) where {sa,sb,TA,TB}
    m = sa[1]
    if length(sa) == 1
        n = 1
    else
        n = sa[2]
    end
    if sb[1] != n
        throw(DimensionMismatch("right hand side B needs first dimension of size $n, has size $(sb[1])"))
    end

    X = [Symbol("X_$(i)_$(j)") for i = 1:m, j = 1:n]

    code = Expr(:block)
    for i = 1:m
        for j = 1:n
            ex = :(A[$(LinearIndices(sa)[i, j])]*B[$(LinearIndices(sb)[j, j])])
            for k = j+1:n
                ex = :($ex + A[$(LinearIndices(sa)[i, k])]*B.data[$(LinearIndices(sb)[k, j])])
            end
            push!(code.args, :($(X[i,j]) = $ex))
        end
    end

    return quote
        @_inline_meta
        @inbounds $code
        TAB = promote_op(matprod, TA, TB)
        return similar_type(A, TAB, Size($m,$n))(tuple($(X...)))
    end
end

@generated function _A_mul_Bc(::Size{sa}, ::Size{sb}, A::StaticArray{<:Tuple,TA}, B::LowerTriangular{TB,<:StaticMatrix}) where {sa,sb,TA,TB}
    m = sa[1]
    if length(sa) == 1
        n = 1
    else
        n = sa[2]
    end
    if sb[1] != n
        throw(DimensionMismatch("right hand side B needs first dimension of size $n, has size $(sb[1])"))
    end

    X = [Symbol("X_$(i)_$(j)") for i = 1:m, j = 1:n]

    code = Expr(:block)
    for i = 1:m
        for j = n:-1:1
            ex = :(A[$(LinearIndices(sa)[i, j])]*B[$(LinearIndices(sb)[j, j])]')
            for k = 1:j-1
                ex = :($ex + A[$(LinearIndices(sa)[i, k])]*B.data[$(LinearIndices(sb)[j, k])]')
            end
            push!(code.args, :($(X[i,j]) = $ex))
        end
    end

    return quote
        @_inline_meta
        @inbounds $code
        TAB = promote_op(matprod, TA, TB)
        return similar_type(A, TAB, Size($m,$n))(tuple($(X...)))
    end
end

@generated function _A_mul_Bt(::Size{sa}, ::Size{sb}, A::StaticMatrix{<:Any,<:Any,TA}, B::LowerTriangular{TB,<:StaticMatrix}) where {sa,sb,TA,TB}
    m, n = sa[1], sa[2]
    if sb[1] != n
        throw(DimensionMismatch("right hand side B needs first dimension of size $n, has size $(sb[1])"))
    end

    X = [Symbol("X_$(i)_$(j)") for i = 1:m, j = 1:n]

    code = Expr(:block)
    for i = 1:m
        for j = n:-1:1
            ex = :(A[$(LinearIndices(sa)[i, j])]*transpose(B[$(LinearIndices(sb)[j, j])]))
            for k = 1:j-1
                ex = :($ex + A[$(LinearIndices(sa)[i, k])]*transpose(B.data[$(LinearIndices(sb)[j, k])]))
            end
            push!(code.args, :($(X[i,j]) = $ex))
        end
    end

    return quote
        @_inline_meta
        @inbounds $code
        TAB = promote_op(matprod, TA, TB)
        return similar_type(A, TAB)(tuple($(X...)))
    end
end

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

@generated function _A_mul_B(::Size{sa}, ::Size{sb}, A::UpperTriangular{<:TA,<:StaticMatrix}, B::UpperTriangular{<:TB,<:StaticMatrix}) where {sa,sb,TA,TB}
    n = sa[1]
    if n != sb[1]
        throw(DimensionMismatch("left and right-hand must have same sizes, got $(n) and $(sb[1])"))
    end

    X = [Symbol("X_$(i)_$(j)") for i = 1:n, j = 1:n]

    TAB = promote_op(*, eltype(TA), eltype(TB))
    z = zero(TAB)

    code = Expr(:block)
    for j = 1:n
        for i = 1:n
            if i > j
                push!(code.args, :($(X[i,j]) = $z))
            else
                ex = :(A.data[$(LinearIndices(sa)[i,i])] * B.data[$(LinearIndices(sb)[i,j])])
                for k = i+1:j
                    ex = :($ex + A.data[$(LinearIndices(sa)[i,k])] * B.data[$(LinearIndices(sb)[k,j])])
                end
                push!(code.args, :($(X[i,j]) = $ex))
            end
        end
    end

    return quote
        @_inline_meta
        @inbounds $code
        return UpperTriangular(similar_type(B.data, $TAB)(tuple($(X...))))
    end

end

@generated function _A_mul_B(::Size{sa}, ::Size{sb}, A::LowerTriangular{<:TA,<:StaticMatrix}, B::LowerTriangular{<:TB,<:StaticMatrix}) where {sa,sb,TA,TB}
    n = sa[1]
    if n != sb[1]
        throw(DimensionMismatch("left and right-hand must have same sizes, got $(n) and $(sb[1])"))
    end

    X = [Symbol("X_$(i)_$(j)") for i = 1:n, j = 1:n]

    TAB = promote_op(*, eltype(TA), eltype(TB))
    z = zero(TAB)

    code = Expr(:block)
    for j = 1:n
        for i = 1:n
            if i < j
                push!(code.args, :($(X[i,j]) = $z))
            else
                ex = :(A.data[$(LinearIndices(sa)[i,j])] * B.data[$(LinearIndices(sb)[j,j])])
                for k = j+1:i
                    ex = :($ex + A.data[$(LinearIndices(sa)[i,k])] * B.data[$(LinearIndices(sb)[k,j])])
                end
                push!(code.args, :($(X[i,j]) = $ex))
            end
        end
    end

    return quote
        @_inline_meta
        @inbounds $code
        return LowerTriangular(similar_type(B.data, $TAB)(tuple($(X...))))
    end

end


@generated function _A_mul_B(::Size{sa}, ::Size{sb}, A::UpperTriangular{<:TA,<:StaticMatrix}, B::LowerTriangular{<:TB,<:StaticMatrix}) where {sa,sb,TA,TB}
    n = sa[1]
    if n != sb[1]
        throw(DimensionMismatch("left and right-hand must have same sizes, got $(n) and $(sb[1])"))
    end

    X = [Symbol("X_$(i)_$(j)") for i = 1:n, j = 1:n]

    code = Expr(:block)
    for j = 1:n
        for i = 1:n
            k1 = max(i,j)
            ex = :(A.data[$(LinearIndices(sa)[i,k1])] * B.data[$(LinearIndices(sb)[k1,j])])
            for k = k1+1:n
                ex = :($ex + A.data[$(LinearIndices(sa)[i,k])] * B.data[$(LinearIndices(sb)[k,j])])
            end
            push!(code.args, :($(X[i,j]) = $ex))
        end
    end

    return quote
        @_inline_meta
        @inbounds $code
        TAB = promote_op(*, eltype(TA), eltype(TB))
        return similar_type(B.data, TAB)(tuple($(X...)))
    end

end

@generated function _A_mul_B(::Size{sa}, ::Size{sb}, A::LowerTriangular{<:TA,<:StaticMatrix}, B::UpperTriangular{<:TB,<:StaticMatrix}) where {sa,sb,TA,TB}
    n = sa[1]
    if n != sb[1]
        throw(DimensionMismatch("left and right-hand must have same sizes, got $(n) and $(sb[1])"))
    end

    X = [Symbol("X_$(i)_$(j)") for i = 1:n, j = 1:n]

    code = Expr(:block)
    for j = 1:n
        for i = 1:n
            ex = :(A.data[$(LinearIndices(sa)[i,1])] * B.data[$(LinearIndices(sb)[1,j])])
            for k = 2:min(i,j)
                ex = :($ex + A.data[$(LinearIndices(sa)[i,k])] * B.data[$(LinearIndices(sb)[k,j])])
            end
            push!(code.args, :($(X[i,j]) = $ex))
        end
    end

    return quote
        @_inline_meta
        @inbounds $code
        TAB = promote_op(*, eltype(TA), eltype(TB))
        return similar_type(B.data, TAB)(tuple($(X...)))
    end

end
