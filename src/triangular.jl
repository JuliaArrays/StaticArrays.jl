import Base: *, Ac_mul_B, At_mul_B, A_mul_Bc, A_mul_Bt, At_mul_Bt, Ac_mul_Bc
import Base: \, Ac_ldiv_B, At_ldiv_B

@inline Size(A::Base.LinAlg.AbstractTriangular{<:Any,<:StaticMatrix}) = Size(A.data)

# TODO add specialized op(AbstractTriangular, AbstractTriangular) methods
# TODO add A*_rdiv_B* methods
@inline *(A::Base.LinAlg.AbstractTriangular{<:Any,<:StaticMatrix}, B::StaticVecOrMat) = _A_mul_B(Size(A), Size(B), A, B)
@inline *(A::StaticMatrix, B::Base.LinAlg.AbstractTriangular{<:Any,<:StaticMatrix}) = _A_mul_B(Size(A), Size(B), A, B)
@inline Ac_mul_B(A::Base.LinAlg.AbstractTriangular{<:Any,<:StaticMatrix}, B::StaticVecOrMat) = _Ac_mul_B(Size(A), Size(B), A, B)
@inline A_mul_Bc(A::StaticMatrix, B::Base.LinAlg.AbstractTriangular{<:Any,<:StaticMatrix}) = _A_mul_Bc(Size(A), Size(B), A, B)
@inline At_mul_B(A::Base.LinAlg.AbstractTriangular{<:Any,<:StaticMatrix}, B::StaticVecOrMat) = _At_mul_B(Size(A), Size(B), A, B)
@inline A_mul_Bt(A::StaticMatrix, B::Base.LinAlg.AbstractTriangular{<:Any,<:StaticMatrix}) = _A_mul_Bt(Size(A), Size(B), A, B)

# Specializations for RowVector
@inline *(rowvec::RowVector{<:Any,<:StaticVector}, A::Base.LinAlg.AbstractTriangular{<:Any,<:StaticMatrix}) = transpose(A * transpose(rowvec))
@inline A_mul_Bt(rowvec::RowVector{<:Any,<:StaticVector}, A::Base.LinAlg.AbstractTriangular{<:Any,<:StaticMatrix}) = transpose(A * transpose(rowvec))
@inline A_mul_Bt(A::Base.LinAlg.AbstractTriangular{<:Any,<:StaticMatrix}, rowvec::RowVector{<:Any,<:StaticVector}) = A * transpose(rowvec)
@inline At_mul_Bt(A::Base.LinAlg.AbstractTriangular{<:Any,<:StaticMatrix}, rowvec::RowVector{<:Any,<:StaticVector}) = A.' * transpose(rowvec)
@inline A_mul_Bc(rowvec::RowVector{<:Any,<:StaticVector}, A::Base.LinAlg.AbstractTriangular{<:Any,<:StaticMatrix}) = ctranspose(A * ctranspose(rowvec))
@inline A_mul_Bc(A::Base.LinAlg.AbstractTriangular{<:Any,<:StaticMatrix}, rowvec::RowVector{<:Any,<:StaticVector}) = A * ctranspose(rowvec)
@inline Ac_mul_Bc(A::Base.LinAlg.AbstractTriangular{<:Any,<:StaticMatrix}, rowvec::RowVector{<:Any,<:StaticVector}) = A' * ctranspose(rowvec)

Ac_mul_B(A::StaticMatrix, B::Base.LinAlg.AbstractTriangular{<:Any,<:StaticMatrix}) = (*)(ctranspose(A), B)
At_mul_B(A::StaticMatrix, B::Base.LinAlg.AbstractTriangular{<:Any,<:StaticMatrix}) = (*)(transpose(A), B)
A_mul_Bc(A::Base.LinAlg.AbstractTriangular{<:Any,<:StaticMatrix}, B::StaticMatrix) = (*)(A, ctranspose(B))
A_mul_Bt(A::Base.LinAlg.AbstractTriangular{<:Any,<:StaticMatrix}, B::StaticMatrix) = (*)(A, transpose(B))
Ac_mul_Bc(A::Base.LinAlg.AbstractTriangular{<:Any,<:StaticMatrix}, B::StaticMatrix) = Ac_mul_B(A, B')
Ac_mul_Bc(A::StaticMatrix, B::Base.LinAlg.AbstractTriangular{<:Any,<:StaticMatrix}) = A_mul_Bc(A', B)
At_mul_Bt(A::Base.LinAlg.AbstractTriangular{<:Any,<:StaticMatrix}, B::StaticMatrix) = At_mul_B(A, B.')
At_mul_Bt(A::StaticMatrix, B::Base.LinAlg.AbstractTriangular{<:Any,<:StaticMatrix}) = A_mul_Bt(A.', B)

@inline \(A::Union{UpperTriangular{<:Any,<:StaticMatrix},LowerTriangular{<:Any,<:StaticMatrix}}, B::StaticVecOrMat) = _A_ldiv_B(Size(A), Size(B), A, B)
@inline Ac_ldiv_B(A::Union{UpperTriangular{<:Any,<:StaticMatrix},LowerTriangular{<:Any,<:StaticMatrix}}, B::StaticVecOrMat) = _Ac_ldiv_B(Size(A), Size(B), A, B)
@inline At_ldiv_B(A::Union{UpperTriangular{<:Any,<:StaticMatrix},LowerTriangular{<:Any,<:StaticMatrix}}, B::StaticVecOrMat) = _At_ldiv_B(Size(A), Size(B), A, B)

for f in (:_A_mul_B, :_Ac_mul_B, :_At_mul_B)
    @eval begin
        @generated function $f(::Size{sa}, ::Size{sb}, A::UpperTriangular{TA,<:StaticMatrix}, B::StaticVecOrMat{TB}) where {sa,sb,TA,TB}
            m = sb[1]
            n = length(sb) > 1 ? sb[2] : 1
            if m != sa[1]
                throw(DimensionMismatch("right hand side B needs first dimension of size $(sa[1]), has size $m"))
            end

            TAB = promote_op(matprod, TA, TB)
            X = [Symbol("X_$(i)_$(j)") for i = 1:m, j = 1:n]

            code = quote end
            for j = 1:n
                if $(f == :_A_mul_B)
                    for i = 1:m
                        ex = :(A.data[$(sub2ind(sa,i,i))]*B[$(sub2ind(sb,i,j))])
                        for k = i+1:m
                            ex = :($ex + A.data[$(sub2ind(sa,i,k))]*B[$(sub2ind(sb,k,j))])
                        end
                        push!(code.args, :($(X[i,j]) = $ex))
                    end
                elseif $(f == :_Ac_mul_B)
                    for i = m:-1:1
                        ex = :(A.data[$(sub2ind(sa,i,i))]'*B[$(sub2ind(sb,i,j))])
                        for k = 1:i-1
                            ex = :($ex + A.data[$(sub2ind(sa,k,i))]'*B[$(sub2ind(sb,k,j))])
                        end
                        push!(code.args, :($(X[i,j]) = $ex))
                    end
                else # $(f == :_At_mul_B)
                    for i = m:-1:1
                        ex = :(A.data[$(sub2ind(sa,i,i))].'*B[$(sub2ind(sb,i,j))])
                        for k = 1:i-1
                            ex = :($ex + A.data[$(sub2ind(sa,k,i))].'*B[$(sub2ind(sb,k,j))])
                        end
                        push!(code.args, :($(X[i,j]) = $ex))
                    end
                end
            end

            return quote
                @_inline_meta
                @inbounds $code
                return similar_type(B, $TAB)(tuple($(X...)))
            end
        end
    end
end

for f in (:_A_mul_B, :_Ac_mul_B, :_At_mul_B)
    @eval begin
        @generated function $f(::Size{sa}, ::Size{sb}, A::LowerTriangular{TA,<:StaticMatrix}, B::StaticVecOrMat{TB}) where {sa,sb,TA,TB}
            m = sb[1]
            n = length(sb) > 1 ? sb[2] : 1
            if m != sa[1]
                throw(DimensionMismatch("right hand side B needs first dimension of size $(sa[1]), has size $m"))
            end

            TAB = promote_op(matprod, TA, TB)
            X = [Symbol("X_$(i)_$(j)") for i = 1:m, j = 1:n]

            code = quote end
            for j = 1:n
                if $(f == :_A_mul_B)
                    for i = m:-1:1
                        ex = :(A.data[$(sub2ind(sa,i,i))]*B[$(sub2ind(sb,i,j))])
                        for k = 1:i-1
                            ex = :($ex + A.data[$(sub2ind(sa,i,k))]*B[$(sub2ind(sb,k,j))])
                        end
                        push!(code.args, :($(X[i,j]) = $ex))
                    end
                elseif $(f == :_Ac_mul_B)
                    for i = 1:m
                        ex = :(A.data[$(sub2ind(sa,i,i))]'*B[$(sub2ind(sb,i,j))])
                        for k = i+1:m
                            ex = :($ex + A.data[$(sub2ind(sa,k,i))]'*B[$(sub2ind(sb,k,j))])
                        end
                        push!(code.args, :($(X[i,j]) = $ex))
                    end
                else # $(f == :_At_mul_B)
                    for i = 1:m
                        ex = :(A.data[$(sub2ind(sa,i,i))].'*B[$(sub2ind(sb,i,j))])
                        for k = i+1:m
                            ex = :($ex + A.data[$(sub2ind(sa,k,i))].'*B[$(sub2ind(sb,k,j))])
                        end
                        push!(code.args, :($(X[i,j]) = $ex))
                    end
                end
            end

            return quote
                @_inline_meta
                @inbounds $code
                return similar_type(B, $TAB)(tuple($(X...)))
            end
        end
    end
end

for f in (:_A_mul_B, :_A_mul_Bc, :_A_mul_Bt)
    @eval begin
        @generated function $f(::Size{sa}, ::Size{sb}, A::StaticMatrix{<:Any,<:Any,TA}, B::UpperTriangular{TB,<:StaticMatrix}) where {sa,sb,TA,TB}
            m, n = sa[1], sa[2]
            if sb[1] != n
                throw(DimensionMismatch("right hand side B needs first dimension of size $n, has size $(sb[1])"))
            end

            TAB = promote_op(matprod, TA, TB)
            X = [Symbol("X_$(i)_$(j)") for i = 1:m, j = 1:n]

            code = quote end
            for i = 1:m
                if $(f == :_A_mul_B)
                    for j = n:-1:1
                        ex = :(A[$(sub2ind(sa,i,j))]*B[$(sub2ind(sb,j,j))])
                        for k = 1:j-1
                            ex = :($ex + A[$(sub2ind(sa,i,k))]*B.data[$(sub2ind(sb,k,j))])
                        end
                        push!(code.args, :($(X[i,j]) = $ex))
                    end
                elseif $(f == :_A_mul_Bc)
                    for j = 1:n
                        ex = :(A[$(sub2ind(sa,i,j))]*B[$(sub2ind(sb,j,j))]')
                        for k = j+1:n
                            ex = :($ex + A[$(sub2ind(sa,i,k))]*B.data[$(sub2ind(sb,j,k))]')
                        end
                        push!(code.args, :($(X[i,j]) = $ex))
                    end
                else # $(f == :_A_mul_Bt)
                    for j = 1:n
                        ex = :(A[$(sub2ind(sa,i,j))]*B[$(sub2ind(sb,j,j))].')
                        for k = j + 1:n
                            ex = :($ex + A[$(sub2ind(sa,i,k))]*B.data[$(sub2ind(sb,j,k))].')
                        end
                        push!(code.args, :($(X[i,j]) = $ex))
                    end
                end
            end

            return quote
                @_inline_meta
                @inbounds $code
                return similar_type(A, $TAB)(tuple($(X...)))
            end
        end
    end
end

for f in (:_A_mul_B, :_A_mul_Bc, :_A_mul_Bt)
    @eval begin
        @generated function $f(::Size{sa}, ::Size{sb}, A::StaticMatrix{<:Any,<:Any,TA}, B::LowerTriangular{TB,<:StaticMatrix}) where {sa,sb,TA,TB}
            m, n = sa[1], sa[2]
            if sb[1] != n
                throw(DimensionMismatch("right hand side B needs first dimension of size $n, has size $(sb[1])"))
            end

            TAB = promote_op(matprod, TA, TB)
            X = [Symbol("X_$(i)_$(j)") for i = 1:m, j = 1:n]

            code = quote end
            for i = 1:m
                if $(f == :_A_mul_B)
                    for j = 1:n
                        ex = :(A[$(sub2ind(sa,i,j))]*B[$(sub2ind(sb,j,j))])
                        for k = j+1:n
                            ex = :($ex + A[$(sub2ind(sa,i,k))]*B.data[$(sub2ind(sb,k,j))])
                        end
                        push!(code.args, :($(X[i,j]) = $ex))
                    end
                elseif $(f == :_A_mul_Bc)
                    for j = n:-1:1
                        ex = :(A[$(sub2ind(sa,i,j))]*B[$(sub2ind(sb,j,j))]')
                        for k = 1:j-1
                            ex = :($ex + A[$(sub2ind(sa,i,k))]*B.data[$(sub2ind(sb,j,k))]')
                        end
                        push!(code.args, :($(X[i,j]) = $ex))
                    end
                else # $(f == :_A_mul_Bt)
                    for j = n:-1:1
                        ex = :(A[$(sub2ind(sa,i,j))]*B[$(sub2ind(sb,j,j))].')
                        for k = 1:j-1
                            ex = :($ex + A[$(sub2ind(sa,i,k))]*B.data[$(sub2ind(sb,j,k))].')
                        end
                        push!(code.args, :($(X[i,j]) = $ex))
                    end
                end
            end

            return quote
                @_inline_meta
                @inbounds $code
                return similar_type(A, $TAB)(tuple($(X...)))
            end
        end
    end
end

for t in (UpperTriangular, LowerTriangular)
    @eval begin
        @generated function _A_ldiv_B(::Size{sa}, ::Size{sb}, A::$t{<:TA,<:StaticMatrix}, B::StaticVecOrMat{TB}) where {sa,sb,TA,TB}
            m = sb[1]
            n = length(sb) > 1 ? sb[2] : 1
            if m != sa[1]
                throw(DimensionMismatch("right hand side B needs first dimension of size $(sa[1]), has size $m"))
            end

            TAB = typeof((zero(TA)*zero(TB) + zero(TA)*zero(TB))/one(TA))
            X = [Symbol("X_$(i)_$(j)") for i = 1:m, j = 1:n]
            init = [:($(X[i,j]) = B[$(sub2ind(sb,i,j))]) for i = 1:m, j = 1:n]

            code = quote end
            for k = 1:n
                for j = $(t == UpperTriangular ? :(m:-1:1) : :(1:m))
                    if k == 1
                        push!(code.args, :(A.data[$(sub2ind(sa,j,j))] == zero(A.data[$(sub2ind(sa,j,j))]) && throw(Base.LinAlg.SingularException($j))))
                    end
                    push!(code.args, :($(X[j,k]) = A.data[$(sub2ind(sa,j,j))] \ $(X[j,k])))
                    for i = $(t == UpperTriangular ? :(j-1:-1:1) : :(j+1:m))
                        push!(code.args, :($(X[i,k]) -= A.data[$(sub2ind(sa,i,j))]*$(X[j,k])))
                    end
                end
            end

            return quote
                @_inline_meta
                @inbounds $(Expr(:block, init...))
                @inbounds $code
                @inbounds return similar_type(B, $TAB)(tuple($(X...)))
            end
        end
    end
end

for f in (:_Ac_ldiv_B, :_At_ldiv_B)
    @eval begin
        @generated function $f(::Size{sa}, ::Size{sb}, A::UpperTriangular{<:TA,<:StaticMatrix}, B::StaticVecOrMat{TB}) where {sa,sb,TA,TB}
            m = sb[1]
            n = length(sb) > 1 ? sb[2] : 1
            if m != sa[1]
                throw(DimensionMismatch("right hand side B needs first dimension of size $(sa[1]), has size $m"))
            end

            TAB = typeof((zero(TA)*zero(TB) + zero(TA)*zero(TB))/one(TA))
            X = [Symbol("X_$(i)_$(j)") for i = 1:m, j = 1:n]

            code = quote end
            for k = 1:n
                if $(f == :_Ac_ldiv_B)
                    for j = 1:m
                        ex = :(B[$(sub2ind(sb,j,k))])
                        for i = 1:j-1
                            ex = :($ex - A.data[$(sub2ind(sa,i,j))]'*$(X[i,k]))
                        end
                        if k == 1
                            push!(code.args, :(A.data[$(sub2ind(sa,j,j))] == zero(A.data[$(sub2ind(sa,j,j))]) && throw(Base.LinAlg.SingularException($j))))
                        end
                        push!(code.args, :($(X[j,k]) = A.data[$(sub2ind(sa,j,j))]' \ $ex))
                    end
                else # $(f == :_At_ldiv_B)
                    for j = 1:m
                        ex = :(B[$(sub2ind(sb,j,k))])
                        for i = 1:j-1
                            ex = :($ex - A.data[$(sub2ind(sa,i,j))]*$(X[i,k]))
                        end
                        if k == 1
                            push!(code.args, :(A.data[$(sub2ind(sa,j,j))] == zero(A.data[$(sub2ind(sa,j,j))]) && throw(Base.LinAlg.SingularException($j))))
                        end
                        push!(code.args, :($(X[j,k]) = A.data[$(sub2ind(sa,j,j))] \ $ex))
                    end
                end
            end

            return quote
                @_inline_meta
                @inbounds $code
                @inbounds return similar_type(B, $TAB)(tuple($(X...)))
            end
        end
    end
end

for f in (:_Ac_ldiv_B, :_At_ldiv_B)
    @eval begin
        @generated function $f(::Size{sa}, ::Size{sb}, A::LowerTriangular{<:TA,<:StaticMatrix}, B::StaticVecOrMat{TB}) where {sa,sb,TA,TB}
            m = sb[1]
            n = length(sb) > 1 ? sb[2] : 1
            if m != sa[1]
                throw(DimensionMismatch("right hand side B needs first dimension of size $(sa[1]), has size $m"))
            end

            TAB = typeof((zero(TA)*zero(TB) + zero(TA)*zero(TB))/one(TA))
            X = [Symbol("X_$(i)_$(j)") for i = 1:m, j = 1:n]

            code = quote end
            for k = 1:n
                if $(f == :_Ac_ldiv_B)
                    for j = m:-1:1
                        ex = :(B[$(sub2ind(sb,j,k))])
                        for i = m:-1:j+1
                            ex = :($ex - A.data[$(sub2ind(sa,i,j))]'*$(X[i,k]))
                        end
                        if k == 1
                            push!(code.args, :(A.data[$(sub2ind(sa,j,j))] == zero(A.data[$(sub2ind(sa,j,j))]) && throw(Base.LinAlg.SingularException($j))))
                        end
                        push!(code.args, :($(X[j,k]) = A.data[$(sub2ind(sa,j,j))]' \ $ex))
                    end
                else # $(f == :_At_ldiv_B)
                    for j = m:-1:1
                        ex = :(B[$(sub2ind(sb,j,k))])
                        for i = m:-1:j+1
                            ex = :($ex - A.data[$(sub2ind(sa,i,j))]*$(X[i,k]))
                        end
                        if k == 1
                            push!(code.args, :(A.data[$(sub2ind(sa,j,j))] == zero(A.data[$(sub2ind(sa,j,j))]) && throw(Base.LinAlg.SingularException($j))))
                        end
                        push!(code.args, :($(X[j,k]) = A.data[$(sub2ind(sa,j,j))] \ $ex))
                    end
                end
            end

            return quote
                @_inline_meta
                @inbounds $code
                @inbounds return similar_type(B, $TAB)(tuple($(X...)))
            end
        end
    end
end
