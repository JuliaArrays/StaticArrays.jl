@inline Size(A::LinearAlgebra.AbstractTriangular{<:Any,<:StaticMatrix}) = Size(A.data)

@static if VERSION < v"0.7-"
    import Base: Ac_mul_B, At_mul_B, A_mul_Bc, A_mul_Bt, At_mul_Bt, Ac_mul_Bc
    import Base: Ac_ldiv_B, At_ldiv_B


    # TODO add specialized op(AbstractTriangular, AbstractTriangular) methods
    # TODO add A*_rdiv_B* methods
    @inline Ac_mul_B(A::LinearAlgebra.AbstractTriangular{<:Any,<:StaticMatrix}, B::StaticVecOrMat) = _Ac_mul_B(Size(A), Size(B), A, B)
    @inline A_mul_Bc(A::StaticVecOrMat, B::LinearAlgebra.AbstractTriangular{<:Any,<:StaticMatrix}) = _A_mul_Bc(Size(A), Size(B), A, B)
    @inline At_mul_B(A::LinearAlgebra.AbstractTriangular{<:Any,<:StaticMatrix}, B::StaticVecOrMat) = _At_mul_B(Size(A), Size(B), A, B)
    @inline A_mul_Bt(A::StaticMatrix, B::LinearAlgebra.AbstractTriangular{<:Any,<:StaticMatrix}) = _A_mul_Bt(Size(A), Size(B), A, B)

    # Specializations for Adjoint
    @inline Base.:*(rowvec::Adjoint{<:Any,<:StaticVector}, A::LinearAlgebra.AbstractTriangular{<:Any,<:StaticMatrix}) = transpose(A * transpose(rowvec))
    @inline Base.:*(A::LinearAlgebra.AbstractTriangular{<:Any,<:StaticMatrix}, rowvec::RowVector{<:Any,<:StaticVector}) = transpose(rowvec.vec * A)
    @inline A_mul_Bt(rowvec::Adjoint{<:Any,<:StaticVector}, A::LinearAlgebra.AbstractTriangular{<:Any,<:StaticMatrix}) = transpose(A * transpose(rowvec))
    @inline A_mul_Bt(A::LinearAlgebra.AbstractTriangular{<:Any,<:StaticMatrix}, rowvec::Adjoint{<:Any,<:StaticVector}) = A * transpose(rowvec)
    @inline At_mul_B(A::LinearAlgebra.AbstractTriangular{<:Any,<:StaticMatrix}, rowvec::RowVector{<:Any,<:StaticVector}) = transpose(rowvec.vec * A)
    @inline At_mul_Bt(A::LinearAlgebra.AbstractTriangular{<:Any,<:StaticMatrix}, rowvec::Adjoint{<:Any,<:StaticVector}) = transpose(A) * transpose(rowvec)
    @inline At_mul_Bt(rowvec::RowVector{<:Any,<:StaticVector}, A::LinearAlgebra.AbstractTriangular{<:Any,<:StaticMatrix}) = rowvec.vec * transpose(A)
    @inline A_mul_Bc(rowvec::RowVector{<:Any,<:StaticVector}, A::LinearAlgebra.AbstractTriangular{<:Any,<:StaticMatrix}) = transpose(conj(A * adjoint(rowvec)))
    @inline A_mul_Bc(A::LinearAlgebra.AbstractTriangular{<:Any,<:StaticMatrix}, rowvec::Adjoint{<:Any,<:StaticVector}) = A * adjoint(rowvec)
    @inline Ac_mul_B(A::LinearAlgebra.AbstractTriangular{<:Any,<:StaticMatrix}, rowvec::RowVector{<:Any,<:StaticVector}) = adjoint(conj(rowvec.vec) * A)
    @inline Ac_mul_Bc(A::LinearAlgebra.AbstractTriangular{<:Any,<:StaticMatrix}, rowvec::Adjoint{<:Any,<:StaticVector}) = A' * adjoint(rowvec)
    @inline Ac_mul_Bc(rowvec::RowVector{<:Any,<:StaticVector}, A::LinearAlgebra.AbstractTriangular{<:Any,<:StaticMatrix}) = conj(rowvec.vec) * A'

    Ac_mul_B(A::StaticMatrix, B::LinearAlgebra.AbstractTriangular{<:Any,<:StaticMatrix}) = (*)(adjoint(A), B)
    At_mul_B(A::StaticMatrix, B::LinearAlgebra.AbstractTriangular{<:Any,<:StaticMatrix}) = (*)(transpose(A), B)
    A_mul_Bc(A::LinearAlgebra.AbstractTriangular{<:Any,<:StaticMatrix}, B::StaticMatrix) = (*)(A, adjoint(B))
    A_mul_Bt(A::LinearAlgebra.AbstractTriangular{<:Any,<:StaticMatrix}, B::StaticMatrix) = (*)(A, transpose(B))
    Ac_mul_Bc(A::LinearAlgebra.AbstractTriangular{<:Any,<:StaticMatrix}, B::StaticMatrix) = Ac_mul_B(A, B')
    Ac_mul_Bc(A::StaticMatrix, B::LinearAlgebra.AbstractTriangular{<:Any,<:StaticMatrix}) = A_mul_Bc(A', B)
    At_mul_Bt(A::LinearAlgebra.AbstractTriangular{<:Any,<:StaticMatrix}, B::StaticMatrix) = At_mul_B(A, transpose(B))
    At_mul_Bt(A::StaticMatrix, B::LinearAlgebra.AbstractTriangular{<:Any,<:StaticMatrix}) = A_mul_Bt(transpose(A), B)

    @inline Ac_ldiv_B(A::Union{UpperTriangular{<:Any,<:StaticMatrix},LowerTriangular{<:Any,<:StaticMatrix}}, B::StaticVecOrMat) = _Ac_ldiv_B(Size(A), Size(B), A, B)
    @inline At_ldiv_B(A::Union{UpperTriangular{<:Any,<:StaticMatrix},LowerTriangular{<:Any,<:StaticMatrix}}, B::StaticVecOrMat) = _At_ldiv_B(Size(A), Size(B), A, B)
else
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
end

@inline Base.:*(A::LinearAlgebra.AbstractTriangular{<:Any,<:StaticMatrix}, B::StaticVecOrMat) = _A_mul_B(Size(A), Size(B), A, B)
@inline Base.:*(A::StaticVecOrMat, B::LinearAlgebra.AbstractTriangular{<:Any,<:StaticMatrix}) = _A_mul_B(Size(A), Size(B), A, B)
@inline Base.:\(A::Union{UpperTriangular{<:Any,<:StaticMatrix},LowerTriangular{<:Any,<:StaticMatrix}}, B::StaticVecOrMat) = _A_ldiv_B(Size(A), Size(B), A, B)


@generated function _A_mul_B(::Size{sa}, ::Size{sb}, A::UpperTriangular{TA,<:StaticMatrix}, B::StaticVecOrMat{TB}) where {sa,sb,TA,TB}
    m = sb[1]
    n = length(sb) > 1 ? sb[2] : 1
    if m != sa[1]
        throw(DimensionMismatch("right hand side B needs first dimension of size $(sa[1]), has size $m"))
    end

    X = [Symbol("X_$(i)_$(j)") for i = 1:m, j = 1:n]

    code = quote end
    for j = 1:n
        for i = 1:m
            ex = :(A.data[$(sub2ind(sa,i,i))]*B[$(sub2ind(sb,i,j))])
            for k = i+1:m
                ex = :($ex + A.data[$(sub2ind(sa,i,k))]*B[$(sub2ind(sb,k,j))])
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

    code = quote end
    for j = 1:n
        for i = m:-1:1
            ex = :(A.data[$(sub2ind(sa,i,i))]'*B[$(sub2ind(sb,i,j))])
            for k = 1:i-1
                ex = :($ex + A.data[$(sub2ind(sa,k,i))]'*B[$(sub2ind(sb,k,j))])
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

    code = quote end
    for j = 1:n
        for i = m:-1:1
            ex = :(transpose(A.data[$(sub2ind(sa,i,i))])*B[$(sub2ind(sb,i,j))])
            for k = 1:i-1
                ex = :($ex + transpose(A.data[$(sub2ind(sa,k,i))])*B[$(sub2ind(sb,k,j))])
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

    code = quote end
    for j = 1:n
        for i = m:-1:1
            ex = :(A.data[$(sub2ind(sa,i,i))]*B[$(sub2ind(sb,i,j))])
            for k = 1:i-1
                ex = :($ex + A.data[$(sub2ind(sa,i,k))]*B[$(sub2ind(sb,k,j))])
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

    code = quote end
    for j = 1:n
        for i = 1:m
            ex = :(A.data[$(sub2ind(sa,i,i))]'*B[$(sub2ind(sb,i,j))])
            for k = i+1:m
                ex = :($ex + A.data[$(sub2ind(sa,k,i))]'*B[$(sub2ind(sb,k,j))])
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

    code = quote end
    for j = 1:n
        for i = 1:m
            ex = :(transpose(A.data[$(sub2ind(sa,i,i))])*B[$(sub2ind(sb,i,j))])
            for k = i+1:m
                ex = :($ex + transpose(A.data[$(sub2ind(sa,k,i))])*B[$(sub2ind(sb,k,j))])
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

@generated function _A_mul_B(::Size{sa}, ::Size{sb}, A::StaticArray{<:Any,TA}, B::UpperTriangular{TB,<:StaticMatrix}) where {sa,sb,TA,TB}
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

    code = quote end
    for i = 1:m
        for j = n:-1:1
            ex = :(A[$(sub2ind(sa,i,j))]*B[$(sub2ind(sb,j,j))])
            for k = 1:j-1
                ex = :($ex + A[$(sub2ind(sa,i,k))]*B.data[$(sub2ind(sb,k,j))])
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

@generated function _A_mul_Bc(::Size{sa}, ::Size{sb}, A::StaticArray{<:Any,TA}, B::UpperTriangular{TB,<:StaticMatrix}) where {sa,sb,TA,TB}
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

    code = quote end
    for i = 1:m
        for j = 1:n
            ex = :(A[$(sub2ind(sa,i,j))]*B[$(sub2ind(sb,j,j))]')
            for k = j+1:n
                ex = :($ex + A[$(sub2ind(sa,i,k))]*B.data[$(sub2ind(sb,j,k))]')
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

    code = quote end
    for i = 1:m
        for j = 1:n
            ex = :(A[$(sub2ind(sa,i,j))]*transpose(B[$(sub2ind(sb,j,j))]))
            for k = j+1:n
                ex = :($ex + A[$(sub2ind(sa,i,k))]*transpose(B.data[$(sub2ind(sb,j,k))]))
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

@generated function _A_mul_B(::Size{sa}, ::Size{sb}, A::StaticArray{<:Any,TA}, B::LowerTriangular{TB,<:StaticMatrix}) where {sa,sb,TA,TB}
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

    code = quote end
    for i = 1:m
        for j = 1:n
            ex = :(A[$(sub2ind(sa,i,j))]*B[$(sub2ind(sb,j,j))])
            for k = j+1:n
                ex = :($ex + A[$(sub2ind(sa,i,k))]*B.data[$(sub2ind(sb,k,j))])
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

@generated function _A_mul_Bc(::Size{sa}, ::Size{sb}, A::StaticArray{<:Any,TA}, B::LowerTriangular{TB,<:StaticMatrix}) where {sa,sb,TA,TB}
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

    code = quote end
    for i = 1:m
        for j = n:-1:1
            ex = :(A[$(sub2ind(sa,i,j))]*B[$(sub2ind(sb,j,j))]')
            for k = 1:j-1
                ex = :($ex + A[$(sub2ind(sa,i,k))]*B.data[$(sub2ind(sb,j,k))]')
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

    code = quote end
    for i = 1:m
        for j = n:-1:1
            ex = :(A[$(sub2ind(sa,i,j))]*transpose(B[$(sub2ind(sb,j,j))]))
            for k = 1:j-1
                ex = :($ex + A[$(sub2ind(sa,i,k))]*transpose(B.data[$(sub2ind(sb,j,k))]))
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
    init = [:($(X[i,j]) = B[$(sub2ind(sb,i,j))]) for i = 1:m, j = 1:n]

    code = quote end
    for k = 1:n
        for j = m:-1:1
            if k == 1
                push!(code.args, :(A.data[$(sub2ind(sa,j,j))] == zero(A.data[$(sub2ind(sa,j,j))]) && throw(LinearAlgebra.SingularException($j))))
            end
            push!(code.args, :($(X[j,k]) = A.data[$(sub2ind(sa,j,j))] \ $(X[j,k])))
            for i = j-1:-1:1
                push!(code.args, :($(X[i,k]) -= A.data[$(sub2ind(sa,i,j))]*$(X[j,k])))
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
    init = [:($(X[i,j]) = B[$(sub2ind(sb,i,j))]) for i = 1:m, j = 1:n]

    code = quote end
    for k = 1:n
        for j = 1:m
            if k == 1
                push!(code.args, :(A.data[$(sub2ind(sa,j,j))] == zero(A.data[$(sub2ind(sa,j,j))]) && throw(LinearAlgebra.SingularException($j))))
            end
            push!(code.args, :($(X[j,k]) = A.data[$(sub2ind(sa,j,j))] \ $(X[j,k])))
            for i = j+1:m
                push!(code.args, :($(X[i,k]) -= A.data[$(sub2ind(sa,i,j))]*$(X[j,k])))
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

    code = quote end
    for k = 1:n
        for j = 1:m
            ex = :(B[$(sub2ind(sb,j,k))])
            for i = 1:j-1
                ex = :($ex - A.data[$(sub2ind(sa,i,j))]'*$(X[i,k]))
            end
            if k == 1
                push!(code.args, :(A.data[$(sub2ind(sa,j,j))] == zero(A.data[$(sub2ind(sa,j,j))]) && throw(LinearAlgebra.SingularException($j))))
            end
            push!(code.args, :($(X[j,k]) = A.data[$(sub2ind(sa,j,j))]' \ $ex))
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

    code = quote end
    for k = 1:n
        for j = 1:m
            ex = :(B[$(sub2ind(sb,j,k))])
            for i = 1:j-1
                ex = :($ex - A.data[$(sub2ind(sa,i,j))]*$(X[i,k]))
            end
            if k == 1
                push!(code.args, :(A.data[$(sub2ind(sa,j,j))] == zero(A.data[$(sub2ind(sa,j,j))]) && throw(LinearAlgebra.SingularException($j))))
            end
            push!(code.args, :($(X[j,k]) = A.data[$(sub2ind(sa,j,j))] \ $ex))
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

    code = quote end
    for k = 1:n
        for j = m:-1:1
            ex = :(B[$(sub2ind(sb,j,k))])
            for i = m:-1:j+1
                ex = :($ex - A.data[$(sub2ind(sa,i,j))]'*$(X[i,k]))
            end
            if k == 1
                push!(code.args, :(A.data[$(sub2ind(sa,j,j))] == zero(A.data[$(sub2ind(sa,j,j))]) && throw(LinearAlgebra.SingularException($j))))
            end
            push!(code.args, :($(X[j,k]) = A.data[$(sub2ind(sa,j,j))]' \ $ex))
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

    code = quote end
    for k = 1:n
        for j = m:-1:1
            ex = :(B[$(sub2ind(sb,j,k))])
            for i = m:-1:j+1
                ex = :($ex - A.data[$(sub2ind(sa,i,j))]*$(X[i,k]))
            end
            if k == 1
                push!(code.args, :(A.data[$(sub2ind(sa,j,j))] == zero(A.data[$(sub2ind(sa,j,j))]) && throw(LinearAlgebra.SingularException($j))))
            end
            push!(code.args, :($(X[j,k]) = A.data[$(sub2ind(sa,j,j))] \ $ex))
        end
    end

    return quote
        @_inline_meta
        @inbounds $code
        TAB = typeof((zero(TA)*zero(TB) + zero(TA)*zero(TB))/one(TA))
        @inbounds return similar_type(B, TAB)(tuple($(X...)))
    end
end
