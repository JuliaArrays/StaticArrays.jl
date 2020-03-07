import LinearAlgebra: BlasFloat, matprod, mul!


# Manage dispatch of * and mul!
# TODO Adjoint? (Inner product?)

@inline *(A::StaticMatrix, B::AbstractVector) = _mul(Size(A), A, B)
@inline *(A::StaticMatrix, B::StaticVector) = _mul(Size(A), Size(B), A, B)
@inline *(A::StaticMatrix, B::StaticMatrix) = _mul(Size(A), Size(B), A, B)
@inline *(A::StaticVector, B::StaticMatrix) = *(reshape(A, Size(Size(A)[1], 1)), B)
@inline *(A::StaticVector, B::Transpose{<:Any, <:StaticVector}) = _mul(Size(A), Size(B), A, B)
@inline *(A::StaticVector, B::Adjoint{<:Any, <:StaticVector}) = _mul(Size(A), Size(B), A, B)
@inline *(A::StaticArray{Tuple{N,1},<:Any,2}, B::Adjoint{<:Any,<:StaticVector}) where {N} = vec(A) * B
@inline *(A::StaticArray{Tuple{N,1},<:Any,2}, B::Transpose{<:Any,<:StaticVector}) where {N} = vec(A) * B


# Implementations

@generated function _mul(::Size{sa}, a::StaticMatrix{<:Any, <:Any, Ta}, b::AbstractVector{Tb}) where {sa, Ta, Tb}
    if sa[2] != 0
        exprs = [reduce((ex1,ex2) -> :(+($ex1,$ex2)), [:(a[$(LinearIndices(sa)[k, j])]*b[$j]) for j = 1:sa[2]]) for k = 1:sa[1]]
    else
        exprs = [:(zero(T)) for k = 1:sa[1]]
    end

    return quote
        @_inline_meta
        if length(b) != sa[2]
            throw(DimensionMismatch("Tried to multiply arrays of size $sa and $(size(b))"))
        end
        T = promote_op(matprod,Ta,Tb)
        @inbounds return similar_type(b, T, Size(sa[1]))(tuple($(exprs...)))
    end
end

@generated function _mul(::Size{sa}, ::Size{sb}, a::StaticMatrix{<:Any, <:Any, Ta}, b::StaticVector{<:Any, Tb}) where {sa, sb, Ta, Tb}
    if sb[1] != sa[2]
        throw(DimensionMismatch("Tried to multiply arrays of size $sa and $sb"))
    end

    if sa[2] != 0
        exprs = [reduce((ex1,ex2) -> :(+($ex1,$ex2)), [:(a[$(LinearIndices(sa)[k, j])]*b[$j]) for j = 1:sa[2]]) for k = 1:sa[1]]
    else
        exprs = [:(zero(T)) for k = 1:sa[1]]
    end

    return quote
        @_inline_meta
        T = promote_op(matprod,Ta,Tb)
        @inbounds return similar_type(b, T, Size(sa[1]))(tuple($(exprs...)))
    end
end

# outer product
@generated function _mul(::Size{sa}, ::Size{sb}, a::StaticVector{<: Any, Ta},
        b::Union{Transpose{Tb, <:StaticVector}, Adjoint{Tb, <:StaticVector}}) where {sa, sb, Ta, Tb}
    newsize = (sa[1], sb[2])
    exprs = [:(a[$i]*b[$j]) for i = 1:sa[1], j = 1:sb[2]]

    return quote
        @_inline_meta
        T = promote_op(*, Ta, Tb)
        @inbounds return similar_type(b, T, Size($newsize))(tuple($(exprs...)))
    end
end

@generated function _mul(Sa::Size{sa}, Sb::Size{sb}, a::StaticMatrix{<:Any, <:Any, Ta}, b::StaticMatrix{<:Any, <:Any, Tb}) where {sa, sb, Ta, Tb}
    # Heuristic choice for amount of codegen
    if sa[1]*sa[2]*sb[2] <= 8*8*8
        return quote
            @_inline_meta
            return mul_unrolled(Sa, Sb, a, b)
        end
    elseif sa[1] <= 14 && sa[2] <= 14 && sb[2] <= 14
        return quote
            @_inline_meta
            return mul_unrolled_chunks(Sa, Sb, a, b)
        end
    else
        return quote
            @_inline_meta
            return mul_loop(Sa, Sb, a, b)
        end
    end
end

@generated function _mul(Sa::Size{sa}, Sb::Size{sb}, a::Union{SizedMatrix{T}, MMatrix{T}, MArray{T}}, b::Union{SizedMatrix{T}, MMatrix{T}, MArray{T}}) where {sa, sb, T <: BlasFloat}
    S = Size(sa[1], sb[2])

    # Heuristic choice between BLAS and explicit unrolling (or chunk-based unrolling)
    if sa[1]*sa[2]*sb[2] >= 14*14*14
        Sa = TSize{size(S),false}()
        Sb = TSize{sa,false}()
        Sc = TSize{sb,false}()
        _add = MulAddMul(true,false)
        return quote
            @_inline_meta
            C = similar(a, T, $S)
            mul_blas!($Sa, C, $Sa, $Sb, a, b, $_add)
            return C
        end
    elseif sa[1]*sa[2]*sb[2] < 8*8*8
        return quote
            @_inline_meta
            return mul_unrolled(Sa, Sb, a, b)
        end
    elseif sa[1] <= 14 && sa[2] <= 14 && sb[2] <= 14
        return quote
            @_inline_meta
            return similar_type(a, T, $S)(mul_unrolled_chunks(Sa, Sb, a, b))
        end
    else
        return quote
            @_inline_meta
            return mul_loop(Sa, Sb, a, b)
        end
    end
end

@generated function mul_unrolled(::Size{sa}, ::Size{sb}, a::StaticMatrix{<:Any, <:Any, Ta}, b::StaticMatrix{<:Any, <:Any, Tb}) where {sa, sb, Ta, Tb}
    if sb[1] != sa[2]
        throw(DimensionMismatch("Tried to multiply arrays of size $sa and $sb"))
    end

    S = Size(sa[1], sb[2])

    if sa[2] != 0
        exprs = [reduce((ex1,ex2) -> :(+($ex1,$ex2)), [:(a[$(LinearIndices(sa)[k1, j])]*b[$(LinearIndices(sb)[j, k2])]) for j = 1:sa[2]]) for k1 = 1:sa[1], k2 = 1:sb[2]]
    else
        exprs = [:(zero(T)) for k1 = 1:sa[1], k2 = 1:sb[2]]
    end

    return quote
        @_inline_meta
        T = promote_op(matprod,Ta,Tb)
        @inbounds return similar_type(a, T, $S)(tuple($(exprs...)))
    end
end


@generated function mul_loop(::Size{sa}, ::Size{sb}, a::StaticMatrix{<:Any, <:Any, Ta}, b::StaticMatrix{<:Any, <:Any, Tb}) where {sa, sb, Ta, Tb}
    if sb[1] != sa[2]
        throw(DimensionMismatch("Tried to multiply arrays of size $sa and $sb"))
    end

    S = Size(sa[1], sb[2])

    tmps = [Symbol("tmp_$(k1)_$(k2)") for k1 = 1:sa[1], k2 = 1:sb[2]]
    exprs_init = [:($(tmps[k1,k2])  = a[$k1] * b[1 + $((k2-1) * sb[1])]) for k1 = 1:sa[1], k2 = 1:sb[2]]
    exprs_loop = [:($(tmps[k1,k2]) += a[$(k1-sa[1]) + $(sa[1])*j] * b[j + $((k2-1) * sb[1])]) for k1 = 1:sa[1], k2 = 1:sb[2]]

    return quote
        @_inline_meta
        T = promote_op(matprod,Ta,Tb)

        @inbounds $(Expr(:block, exprs_init...))
        for j = 2:$(sa[2])
            @inbounds $(Expr(:block, exprs_loop...))
        end
        @inbounds return similar_type(a, T, $S)(tuple($(tmps...)))
    end
end

# Concatenate a series of matrix-vector multiplications
# Each function is N^2 not N^3 - aids in compile time.
@generated function mul_unrolled_chunks(::Size{sa}, ::Size{sb}, a::StaticMatrix{<:Any, <:Any, Ta}, b::StaticMatrix{<:Any, <:Any, Tb}) where {sa, sb, Ta, Tb}
    if sb[1] != sa[2]
        throw(DimensionMismatch("Tried to multiply arrays of size $sa and $sb"))
    end

    S = Size(sa[1], sb[2])

    # Do a custom b[:, k2] to return a SVector (an isbitstype type) rather than (possibly) a mutable type. Avoids allocation == faster
    tmp_type_in = :(SVector{$(sb[1]), T})
    tmp_type_out = :(SVector{$(sa[1]), T})
    vect_exprs = [:($(Symbol("tmp_$k2"))::$tmp_type_out = partly_unrolled_multiply(TSize(a), TSize($(sb[1])), a,
                    $(Expr(:call, tmp_type_in, [Expr(:ref, :b, LinearIndices(sb)[i, k2]) for i = 1:sb[1]]...)))::$tmp_type_out)
                  for k2 = 1:sb[2]]

    exprs = [:($(Symbol("tmp_$k2"))[$k1]) for k1 = 1:sa[1], k2 = 1:sb[2]]

    return quote
        @_inline_meta
        T = promote_op(matprod,Ta,Tb)
        $(Expr(:block,
            vect_exprs...,
            :(@inbounds return similar_type(a, T, $S)(tuple($(exprs...))))
        ))
    end
end

#
