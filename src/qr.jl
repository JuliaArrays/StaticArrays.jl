# define our own struct since LinearAlgebra.QR are restricted to Matrix
struct QR{Q,R,P}
    Q::Q
    R::R
    p::P
end

# iteration for destructuring into components
Base.iterate(S::QR) = (S.Q, Val(:R))
Base.iterate(S::QR, ::Val{:R}) = (S.R, Val(:p))
Base.iterate(S::QR, ::Val{:p}) = (S.p, Val(:done))
Base.iterate(S::QR, ::Val{:done}) = nothing

pivot_options = if isdefined(LinearAlgebra, :PivotingStrategy) # introduced in Julia v1.7
    (:(Val{true}), :(Val{false}), :NoPivot, :ColumnNorm)
else
    (:(Val{true}), :(Val{false}))
end
for pv in pivot_options
    @eval begin
        @inline function qr(A::StaticMatrix, pivot::$pv)
            QRp = _qr(Size(A), A, pivot)
            if length(QRp) === 2
                # create an identity permutation since that is cheap,
                # and much safer since, in the case of isbits types, we can't
                # safely leave the field undefined.
                p = identity_perm(QRp[2])
                return QR(QRp[1], QRp[2], p)
            else # length(QRp) === 3
                return QR(QRp[1], QRp[2], QRp[3])
            end 
        end
    end
end
"""
    qr(A::StaticMatrix,
       pivot::Union{Val{true}, Val{false}, LinearAlgebra.PivotingStrategy} = Val(false))

Compute the QR factorization of `A`. The factors can be obtained by iteration:

```jldoctest qr
julia> A = @SMatrix rand(3,4);

julia> Q, R = qr(A);

julia> Q * R ≈ A
true
```

or by using `getfield`:

```jldoctest qr
julia> F = qr(A);

julia> F.Q * F.R ≈ A
true
```
"""
qr(A::StaticMatrix) = qr(A, Val(false))

function identity_perm(R::StaticMatrix{N,M,T}) where {N,M,T}
    return similar_type(R, Int, Size((M,)))(ntuple(x -> x, Val{M}()))
end

_qreltype(::Type{T}) where T = typeof(zero(T)/sqrt(abs2(one(T))))

@generated function _qr(::Size{sA}, A::StaticMatrix{<:Any, <:Any, TA},
                        pivot = Val(false)) where {sA, TA}

    SizeQ = Size( sA[1], diagsize(Size(A)) )
    SizeR = Size( diagsize(Size(A)), sA[2] )

    if pivot === Val{true} || (isdefined(LinearAlgebra, :PivotingStrategy) && pivot === ColumnNorm)
        _pivot = isdefined(LinearAlgebra, :PivotingStrategy) ? ColumnNorm() : Val(true)
        return quote
            @_inline_meta
            Q0, R0, p0 = qr(Matrix(A), $(_pivot))
            T = _qreltype(TA)
            return similar_type(A, T, $(SizeQ))(Matrix(Q0)),
                   similar_type(A, T, $(SizeR))(R0),
                   similar_type(A, Int, $(Size(sA[2])))(p0)
        end
    else
        if (sA[1]*sA[1] + sA[1]*sA[2])÷2 * diagsize(Size(A)) < 17*17*17
            return quote
                @_inline_meta
                return qr_unrolled(Size(A), A, Val(false))
            end
        else
            _pivot = isdefined(LinearAlgebra, :PivotingStrategy) ? NoPivot() : Val(false)
            return quote
                @_inline_meta
                Q0R0 = qr(Matrix(A), $(_pivot))
                Q0, R0 = Matrix(Q0R0.Q), Q0R0.R
                T = _qreltype(TA)
                return similar_type(A, T, $(SizeQ))(Q0),
                       similar_type(A, T, $(SizeR))(R0)
            end
        end
    end
end


# Compute the QR decomposition of `A` such that `A = Q*R`
# by Householder reflections without pivoting.
#
# For original source code see below.
@generated function qr_unrolled(::Size{sA}, A::StaticMatrix{<:Any, <:Any, TA}, pivot::Val{false}) where {sA, TA}
    m, n = sA[1], sA[2]

    Q = [Symbol("Q_$(i)_$(j)") for i = 1:m, j = 1:m]
    R = [Symbol("R_$(i)_$(j)") for i = 1:m, j = 1:n]

    initQ = [:($(Q[i, j]) = $(i == j ? one : zero)(T)) for i = 1:m, j = 1:m]  # Q .= I
    initR = [:($(R[i, j]) = T(A[$i, $j])) for i = 1:m, j = 1:n]               # R .= A

    code = quote end
    for k = 1:min(m - 1 + !(TA<:Real), n)
        #x = view(R, k:m, k)
        #τk = reflector!(x)
        push!(code.args, :(ξ1 = $(R[k, k])))
        ex = :(normu = abs2(ξ1))
        for i = k+1:m
            ex = :($ex + abs2($(R[i, k])))
        end
        push!(code.args, :(normu = sqrt($ex)))
        push!(code.args, :(ν = copysign(normu, real(ξ1))))
        push!(code.args, :(ξ1 += ν))
        push!(code.args, :(invξ1 = ξ1 == zero(T) ? zero(T) : inv(ξ1)))
        push!(code.args, :($(R[k, k]) = -ν))
        for i = k+1:m
            push!(code.args, :($(R[i, k]) *= invξ1))
        end
        push!(code.args, :(τk = ν == zero(T) ? zero(T) : ξ1/ν))

        #reflectorApply!(x, τk, view(R, k:m, k+1:n))
        for j = k+1:n
            ex = :($(R[k, j]))
            for i = k+1:m
                ex = :($ex + $(R[i, k])'*$(R[i, j]))
            end
            push!(code.args, :(vRj = τk'*$ex))
            push!(code.args, :($(R[k, j]) -= vRj))
            for i = k+1:m
                push!(code.args, :($(R[i, j]) -= $(R[i, k])*vRj))
            end
        end

        #reflectorApplyRight!(x, τk, view(Q, 1:m, k:m))
        for i = 1:m
            ex = :($(Q[i, k]))
            for j = k+1:m
                ex = :($ex + $(Q[i, j])*$(R[j, k]))
            end
            push!(code.args, :(Qiv = $ex*τk))
            push!(code.args, :($(Q[i, k]) -= Qiv))
            for j = k+1:m
                push!(code.args, :($(Q[i, j]) -= Qiv*$(R[j, k])'))
            end
        end

        for i = k+1:m
            push!(code.args, :($(R[i, k]) = zero(T)))
        end
    end

    # truncate Q and R sizes in LAPACK consilient way
    mQ, nQ = m, min(m, n)
    mR, nR = min(m, n), n

    return quote
        @_inline_meta
        T = _qreltype(TA)
        @inbounds $(Expr(:block, initQ...))
        @inbounds $(Expr(:block, initR...))
        @inbounds $code
        @inbounds return similar_type(A, T, $(Size(mQ, nQ)))( tuple($(Q[1:mQ, 1:nQ]...)) ),
                         similar_type(A, T, $(Size(mR, nR)))( tuple($(R[1:mR, 1:nR]...)) )
    end

end


## Source for @generated qr_unrolled() function above.
## Derived from base/linalg/qr.jl
## thin=true version of QR
#function qr_unrolled(A::StaticMatrix{<:Any, <:Any, TA}) where {TA}
#    m, n = size(A)
#    T = _qreltype(TA)
#    Q = MMatrix{m,m,T,m*m}(I)
#    R = MMatrix{m,n,T,m*n}(A)
#    for k = 1:min(m - 1 + !(TA<:Real), n)
#        #x = view(R, k:m, k)
#        #τk = reflector!(x)
#        ξ1 = R[k, k]
#        normu = abs2(ξ1)
#        for i = k+1:m
#            normu += abs2(R[i, k])
#        end
#        normu = sqrt(normu)
#        ν = copysign(normu, real(ξ1))
#        ξ1 += ν
#        invξ1 = ξ1 == zero(T) ? zero(T) : inv(ξ1)
#        R[k, k] = -ν
#        for i = k+1:m
#            R[i, k] *= invξ1
#        end
#        τk = ν == zero(T) ? zero(T) : ξ1/ν
#
#        #reflectorApply!(x, τk, view(R, k:m, k+1:n))
#        for j = k+1:n
#            vRj = R[k, j]
#            for i = k+1:m
#                vRj += R[i, k]'*R[i, j]
#            end
#            vRj = τk'*vRj
#            R[k, j] -= vRj
#            for i = k+1:m
#                R[i, j] -= R[i, k]*vRj
#            end
#        end
#
#        #reflectorApplyRight!(x, τk, view(Q, 1:m, k:m))
#        for i = 1:m
#            Qiv = Q[i, k]
#            for j = k+1:m
#                Qiv += Q[i, j]*R[j, k]
#            end
#            Qiv = Qiv*τk
#            Q[i, k] -= Qiv
#            for j = k+1:m
#                Q[i, j] -= Qiv*R[j, k]'
#            end
#        end
#
#        for i = k+1:m
#            R[i, k] = zero(T)
#        end
#
#    end
#    if m > n
#        return (similar_type(A, T, Size(m, n))(Q[1:m,1:n]), similar_type(A, T, Size(n, n))(R[1:n,1:n]))
#    else
#        return (similar_type(A, T, Size(m, m))(Q), similar_type(A, T, Size(n, n))(R))
#    end
#end

