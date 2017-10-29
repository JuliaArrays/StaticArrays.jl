_thin_must_hold(thin) = 
    thin || throw(ArgumentError("For the sake of type stability, `thin = true` must hold."))
import Base.qr


@inline function qr(A::StaticMatrix, pivot::Type{Val{true}}; thin::Bool=true)
    _thin_must_hold(thin)
    return qr(Size(A), A, pivot, Val{true})
end

@generated function qr(SA::Size{sA}, A::StaticMatrix{<:Any, <:Any, TA}, pivot::Type{Val{true}}, thin::Union{Type{Val{false}}, Type{Val{true}}}) where {sA, TA}
    mQ = nQ = mR = sA[1]
    nR = sA[2]
    sA[1] > sA[2] && (mR = sA[2])
    sA[1] > sA[2] && thin <: Type{Val{true}} && (nQ = sA[2])
    T = arithmetic_closure(TA)
    QT = similar_type(A, T, Size(mQ, nQ))
    RT = similar_type(A, T, Size(mR, nR))
    PT = similar_type(A, Int, Size(sA[2]))
    return quote
        @_inline_meta
        Q0, R0, p0 = Base.qr(Matrix(A), pivot)
        return $QT(Q0), $RT(R0), $PT(p0)
    end
end


@inline function qr(A::StaticMatrix, pivot::Type{Val{false}}; thin::Bool=true)
    _thin_must_hold(thin)
    return qr(Size(A), A, pivot, Val{true})
end

@generated function qr(SA::Size{sA}, A::StaticMatrix{<:Any, <:Any, TA}, pivot::Type{Val{false}}, thin::Union{Type{Val{false}}, Type{Val{true}}}) where {sA, TA}
    if sA[1] < 17 && sA[2] < 17
        return quote
            @_inline_meta
            return qr_householder_unrolled(SA, A, thin)
        end
    else
        mQ = nQ = mR = sA[1]
        nR = sA[2]
        sA[1] > sA[2] && (mR = sA[2])
        sA[1] > sA[2] && thin <: Type{Val{true}} && (nQ = sA[2])
        T = arithmetic_closure(TA)
        QT = similar_type(A, T, Size(mQ, nQ))
        RT = similar_type(A, T, Size(mR, nR))
        return quote
            @_inline_meta
            Q0, R0 = Base.qr(Matrix(A), pivot)
            return $QT(Q0), $RT(R0)
        end
    end
end


# Compute the QR decomposition of `A` such that `A = Q*R`
# by Householder reflections without pivoting.
#
# `thin=true` (reduced) method will produce `Q` and `R` in truncated form,
# in the case of `thin=false` Q is full, but R is still reduced, see [`qr`](@ref).
#
# For original source code see below.
@generated function qr_householder_unrolled(::Size{sA}, A::StaticMatrix{<:Any, <:Any, TA}, thin::Union{Type{Val{false}},Type{Val{true}}}) where {sA, TA}
    mQ = nQ = mR = m = sA[1]
    nR = n = sA[2]
    # truncate Q and R for thin case
    m > n && (mR = n)
    m > n && thin <: Type{Val{true}} && (nQ = n)

    Q = [Symbol("Q_$(i)_$(j)") for i = 1:m, j = 1:m]
    R = [Symbol("R_$(i)_$(j)") for i = 1:m, j = 1:n]

    initQ = [:($(Q[i, j]) = $(i == j ? one : zero)(T)) for i = 1:m, j = 1:m]  # Q .= eye(A)
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

    return quote
        @_inline_meta
        T = arithmetic_closure(TA)
        @inbounds $(Expr(:block, initQ...))
        @inbounds $(Expr(:block, initR...))
        @inbounds $code
        @inbounds return similar_type(A, T, $(Size(mQ,nQ)))(tuple($(Q[1:mQ,1:nQ]...))),
                         similar_type(A, T, $(Size(mR,nR)))(tuple($(R[1:mR,1:nR]...)))
    end

end

## source for @generated function above
## derived from base/linalg/qr.jl
## thin version of QR
#function qr_householder_unrolled(A::StaticMatrix{<:Any, <:Any, TA}) where {TA}
#    m, n = size(A)
#    T = arithmetic_closure(TA)
#    Q = eye(MMatrix{m,m,T,m*m})
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

