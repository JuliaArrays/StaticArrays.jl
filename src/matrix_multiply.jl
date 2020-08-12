import LinearAlgebra: BlasFloat, matprod, mul!


# Manage dispatch of * and mul!
# TODO Adjoint? (Inner product?)

const StaticMatMulLike{s1, s2, T} = Union{
    StaticMatrix{s1, s2, T},
    Symmetric{T, <:StaticMatrix{s1, s2, T}},
    Hermitian{T, <:StaticMatrix{s1, s2, T}},
    LowerTriangular{T, <:StaticMatrix{s1, s2, T}},
    UpperTriangular{T, <:StaticMatrix{s1, s2, T}}}

@inline *(A::StaticMatMulLike, B::AbstractVector) = _mul(Size(A), A, B)
@inline *(A::StaticMatMulLike, B::StaticVector) = _mul(Size(A), Size(B), A, B)
@inline *(A::StaticMatMulLike, B::StaticMatMulLike) = _mul(Size(A), Size(B), A, B)
@inline *(A::StaticVector, B::StaticMatMulLike) = *(reshape(A, Size(Size(A)[1], 1)), B)
@inline *(A::StaticVector, B::Transpose{<:Any, <:StaticVector}) = _mul(Size(A), Size(B), A, B)
@inline *(A::StaticVector, B::Adjoint{<:Any, <:StaticVector}) = _mul(Size(A), Size(B), A, B)
@inline *(A::StaticArray{Tuple{N,1},<:Any,2}, B::Adjoint{<:Any,<:StaticVector}) where {N} = vec(A) * B
@inline *(A::StaticArray{Tuple{N,1},<:Any,2}, B::Transpose{<:Any,<:StaticVector}) where {N} = vec(A) * B

function gen_by_access(expr_gen, a::Type{<:StaticMatrix}, asym = :a)
    return expr_gen(:any)
end
function gen_by_access(expr_gen, a::Type{<:Symmetric{<:Any, <:StaticMatrix}}, asym = :a)
    return quote
        if $(asym).uplo == 'U'
            $(expr_gen(:up))
        else
            $(expr_gen(:lo))
        end
    end
end
function gen_by_access(expr_gen, a::Type{<:Hermitian{<:Any, <:StaticMatrix}}, asym = :a)
    return quote
        if $(asym).uplo == 'U'
            $(expr_gen(:up_herm))
        else
            $(expr_gen(:lo_herm))
        end
    end
end
function gen_by_access(expr_gen, a::Type{<:UpperTriangular{<:Any, <:StaticMatrix}}, asym = :a)
    return expr_gen(:upper_triangular)
end
function gen_by_access(expr_gen, a::Type{<:LowerTriangular{<:Any, <:StaticMatrix}}, asym = :a)
    return expr_gen(:lower_triangular)
end
function gen_by_access(expr_gen, a::Type{<:StaticMatrix}, b::Type)
    return quote
        return $(gen_by_access(b, :b) do access_b
            expr_gen(:any, access_b)
        end)
    end
end
function gen_by_access(expr_gen, a::Type{<:Symmetric{<:Any, <:StaticMatrix}}, b::Type)
    return quote
        if a.uplo == 'U'
            return $(gen_by_access(b, :b) do access_b
                expr_gen(:up, access_b)
            end)
        else
            return $(gen_by_access(b, :b) do access_b
                expr_gen(:lo, access_b)
            end)
        end
    end
end
function gen_by_access(expr_gen, a::Type{<:Hermitian{<:Any, <:StaticMatrix}}, b::Type)
    return quote
        if a.uplo == 'U'
            return $(gen_by_access(b, :b) do access_b
                expr_gen(:up_herm, access_b)
            end)
        else
            return $(gen_by_access(b, :b) do access_b
                expr_gen(:lo_herm, access_b)
            end)
        end
    end
end
function gen_by_access(expr_gen, a::Type{<:UpperTriangular{<:Any, <:StaticMatrix}}, b::Type)
    return quote
        return $(gen_by_access(b, :b) do access_b
            expr_gen(:upper_triangular, access_b)
        end)
    end
end
function gen_by_access(expr_gen, a::Type{<:LowerTriangular{<:Any, <:StaticMatrix}}, b::Type)
    return quote
        return $(gen_by_access(b, :b) do access_b
            expr_gen(:lower_triangular, access_b)
        end)
    end
end

"""
    mul_result_structure(a::Type, b::Type)

Get a structure wrapper that should be applied to the result of multiplication of matrices
of given types (a*b). 
"""
function mul_result_structure(a, b)
    return identity
end
function mul_result_structure(::UpperTriangular{<:Any, <:StaticMatrix}, ::UpperTriangular{<:Any, <:StaticMatrix})
    return UpperTriangular
end
function mul_result_structure(::LowerTriangular{<:Any, <:StaticMatrix}, ::LowerTriangular{<:Any, <:StaticMatrix})
    return LowerTriangular
end

function uplo_access(sa, asym, k, j, uplo)
    if uplo == :any
        return :($asym[$(LinearIndices(sa)[k, j])])
    elseif uplo == :up
        if k <= j
            return :($asym[$(LinearIndices(sa)[k, j])])
        else
            return :($asym[$(LinearIndices(sa)[j, k])])
        end
    elseif uplo == :lo
        if k >= j
            return :($asym[$(LinearIndices(sa)[k, j])])
        else
            return :($asym[$(LinearIndices(sa)[j, k])])
        end
    elseif uplo == :up_herm
        if k <= j
            return :($asym[$(LinearIndices(sa)[k, j])])
        else
            return :(adjoint($asym[$(LinearIndices(sa)[j, k])]))
        end
    elseif uplo == :lo_herm
        if k >= j
            return :($asym[$(LinearIndices(sa)[k, j])])
        else
            return :(adjoint($asym[$(LinearIndices(sa)[j, k])]))
        end
    elseif uplo == :upper_triangular
        if k <= j
            return :($asym[$(LinearIndices(sa)[k, j])])
        else
            return :(zero(T))
        end
    elseif uplo == :lower_triangular
        if k >= j
            return :($asym[$(LinearIndices(sa)[k, j])])
        else
            return :(zero(T))
        end
    end
end

# Implementations

function mul_smat_vec_exprs(sa, access_a)
    return [reduce((ex1,ex2) -> :(+($ex1,$ex2)), [:($(uplo_access(sa, :a, k, j, access_a))*b[$j]) for j = 1:sa[2]]) for k = 1:sa[1]]
end

@generated function _mul(::Size{sa}, a::StaticMatMulLike{<:Any, <:Any, Ta}, b::AbstractVector{Tb}) where {sa, Ta, Tb}
    if sa[2] != 0
        retexpr = gen_by_access(a) do access_a
            exprs = mul_smat_vec_exprs(sa, access_a)
            return :(@inbounds return similar_type(b, T, Size(sa[1]))(tuple($(exprs...))))
        end
    else
        exprs = [:(zero(T)) for k = 1:sa[1]]
        retexpr = :(@inbounds return similar_type(b, T, Size(sa[1]))(tuple($(exprs...))))
    end

    return quote
        @_inline_meta
        if length(b) != sa[2]
            throw(DimensionMismatch("Tried to multiply arrays of size $sa and $(size(b))"))
        end
        T = promote_op(matprod,Ta,Tb)
        $retexpr
    end
end

@generated function _mul(::Size{sa}, ::Size{sb}, a::StaticMatMulLike{<:Any, <:Any, Ta}, b::StaticVector{<:Any, Tb}) where {sa, sb, Ta, Tb}
    if sb[1] != sa[2]
        throw(DimensionMismatch("Tried to multiply arrays of size $sa and $sb"))
    end

    if sa[2] != 0
        retexpr = gen_by_access(a) do access_a
            exprs = mul_smat_vec_exprs(sa, access_a)
            return :(@inbounds similar_type(b, T, Size(sa[1]))(tuple($(exprs...))))
        end
    else
        exprs = [:(zero(T)) for k = 1:sa[1]]
        retexpr = :(@inbounds return similar_type(b, T, Size(sa[1]))(tuple($(exprs...))))
    end

    return quote
        @_inline_meta
        T = promote_op(matprod,Ta,Tb)
        $retexpr
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

@generated function _mul(Sa::Size{sa}, Sb::Size{sb}, a::StaticMatMulLike{<:Any, <:Any, Ta}, b::StaticMatMulLike{<:Any, <:Any, Tb}) where {sa, sb, Ta, Tb}
    # Heuristic choice for amount of codegen
    if sa[1]*sa[2]*sb[2] <= 8*8*8 || !(a <: StaticMatrix) || !(b <: StaticMatrix)
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

@generated function mul_unrolled(::Size{sa}, ::Size{sb}, a::StaticMatMulLike{<:Any, <:Any, Ta}, b::StaticMatMulLike{<:Any, <:Any, Tb}) where {sa, sb, Ta, Tb}
    if sb[1] != sa[2]
        throw(DimensionMismatch("Tried to multiply arrays of size $sa and $sb"))
    end

    S = Size(sa[1], sb[2])

    if sa[2] != 0
        retexpr = gen_by_access(a, b) do access_a, access_b
            exprs = [reduce((ex1,ex2) -> :(+($ex1,$ex2)),
                [:($(uplo_access(sa, :a, k1, j, access_a))*$(uplo_access(sb, :b, j, k2, access_b))) for j = 1:sa[2]]
                ) for k1 = 1:sa[1], k2 = 1:sb[2]]
            return :((mul_result_structure(a, b))(similar_type(a, T, $S)(tuple($(exprs...)))))
        end
    else
        exprs = [:(zero(T)) for k1 = 1:sa[1], k2 = 1:sb[2]]
        retexpr = :(return (mul_result_structure(a, b))(similar_type(a, T, $S)(tuple($(exprs...)))))
    end

    return quote
        @_inline_meta
        T = promote_op(matprod,Ta,Tb)
        @inbounds $retexpr
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
