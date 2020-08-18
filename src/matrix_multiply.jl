import LinearAlgebra: BlasFloat, matprod, mul!


# Manage dispatch of * and mul!
# TODO Adjoint? (Inner product?)

# *(A::StaticMatMulLike, B::AbstractVector) causes an ambiguity with SparseArrays
@inline *(A::StaticMatrix, B::AbstractVector) = _mul(Size(A), A, B)
@inline *(A::StaticMatMulLike, B::StaticVector) = _mul(Size(A), Size(B), A, B)
@inline *(A::StaticMatrix, B::StaticVector) = _mul(Size(A), Size(B), A, B)
@inline *(A::StaticMatMulLike, B::StaticMatMulLike) = _mul(Size(A), Size(B), A, B)
@inline *(A::StaticVector, B::StaticMatMulLike) = *(reshape(A, Size(Size(A)[1], 1)), B)
@inline *(A::StaticVector, B::Transpose{<:Any, <:StaticVector}) = _mul(Size(A), Size(B), A, B)
@inline *(A::StaticVector, B::Adjoint{<:Any, <:StaticVector}) = _mul(Size(A), Size(B), A, B)
@inline *(A::StaticArray{Tuple{N,1},<:Any,2}, B::Adjoint{<:Any,<:StaticVector}) where {N} = vec(A) * B
@inline *(A::StaticArray{Tuple{N,1},<:Any,2}, B::Transpose{<:Any,<:StaticVector}) where {N} = vec(A) * B

"""
    gen_by_access(expr_gen, a::Type{<:AbstractArray}, asym = :wrapped_a)

Statically generate outer code for fully unrolled multiplication loops.
Returned code does wrapper-specific tests (for example if a symmetric matrix view is
`U` or `L`) and the body of the if expression is then generated by function `expr_gen`.
The function `expr_gen` receives access pattern description symbol as its argument
and this symbol is then consumed by uplo_access to generate the right code for matrix
element access.

The name of the matrix to test is indicated by `asym`.
"""
function gen_by_access(expr_gen, a::Type{<:StaticVecOrMat}, asym = :wrapped_a)
    return expr_gen(:any)
end
function gen_by_access(expr_gen, a::Type{<:Symmetric{<:Any, <:StaticMatrix}}, asym = :wrapped_a)
    return quote
        if $(asym).uplo == 'U'
            $(expr_gen(:up))
        else
            $(expr_gen(:lo))
        end
    end
end
function gen_by_access(expr_gen, a::Type{<:Hermitian{<:Any, <:StaticMatrix}}, asym = :wrapped_a)
    return quote
        if $(asym).uplo == 'U'
            $(expr_gen(:up_herm))
        else
            $(expr_gen(:lo_herm))
        end
    end
end
function gen_by_access(expr_gen, a::Type{<:UpperTriangular{<:Any, <:StaticMatrix}}, asym = :wrapped_a)
    return expr_gen(:upper_triangular)
end
function gen_by_access(expr_gen, a::Type{<:LowerTriangular{<:Any, <:StaticMatrix}}, asym = :wrapped_a)
    return expr_gen(:lower_triangular)
end
function gen_by_access(expr_gen, a::Type{<:UnitUpperTriangular{<:Any, <:StaticMatrix}}, asym = :wrapped_a)
    return expr_gen(:unit_upper_triangular)
end
function gen_by_access(expr_gen, a::Type{<:UnitLowerTriangular{<:Any, <:StaticMatrix}}, asym = :wrapped_a)
    return expr_gen(:unit_lower_triangular)
end
function gen_by_access(expr_gen, a::Type{<:Transpose{<:Any, <:StaticVecOrMat}}, asym = :wrapped_a)
    return expr_gen(:transpose)
end
function gen_by_access(expr_gen, a::Type{<:Adjoint{<:Any, <:StaticVecOrMat}}, asym = :wrapped_a)
    return expr_gen(:adjoint)
end
"""
    gen_by_access(expr_gen, a::Type{<:AbstractArray}, b::Type{<:AbstractArray})

Simiar to gen_by_access with only one type argument. The difference is that tests for both
arrays of type `a` and `b` are generated and `expr_gen` receives two access arguments,
first for matrix `a` and the second for matrix `b`.
"""
function gen_by_access(expr_gen, a::Type{<:StaticMatrix}, b::Type)
    return quote
        return $(gen_by_access(b, :wrapped_b) do access_b
            expr_gen(:any, access_b)
        end)
    end
end
function gen_by_access(expr_gen, a::Type{<:Symmetric{<:Any, <:StaticMatrix}}, b::Type)
    return quote
        if wrapped_a.uplo == 'U'
            return $(gen_by_access(b, :wrapped_b) do access_b
                expr_gen(:up, access_b)
            end)
        else
            return $(gen_by_access(b, :wrapped_b) do access_b
                expr_gen(:lo, access_b)
            end)
        end
    end
end
function gen_by_access(expr_gen, a::Type{<:Hermitian{<:Any, <:StaticMatrix}}, b::Type)
    return quote
        if wrapped_a.uplo == 'U'
            return $(gen_by_access(b, :wrapped_b) do access_b
                expr_gen(:up_herm, access_b)
            end)
        else
            return $(gen_by_access(b, :wrapped_b) do access_b
                expr_gen(:lo_herm, access_b)
            end)
        end
    end
end
function gen_by_access(expr_gen, a::Type{<:UpperTriangular{<:Any, <:StaticMatrix}}, b::Type)
    return quote
        return $(gen_by_access(b, :wrapped_b) do access_b
            expr_gen(:upper_triangular, access_b)
        end)
    end
end
function gen_by_access(expr_gen, a::Type{<:LowerTriangular{<:Any, <:StaticMatrix}}, b::Type)
    return quote
        return $(gen_by_access(b, :wrapped_b) do access_b
            expr_gen(:lower_triangular, access_b)
        end)
    end
end
function gen_by_access(expr_gen, a::Type{<:UnitUpperTriangular{<:Any, <:StaticMatrix}}, b::Type)
    return quote
        return $(gen_by_access(b, :wrapped_b) do access_b
            expr_gen(:unit_upper_triangular, access_b)
        end)
    end
end
function gen_by_access(expr_gen, a::Type{<:UnitLowerTriangular{<:Any, <:StaticMatrix}}, b::Type)
    return quote
        return $(gen_by_access(b, :wrapped_b) do access_b
            expr_gen(:unit_lower_triangular, access_b)
        end)
    end
end
function gen_by_access(expr_gen, a::Type{<:Transpose{<:Any, <:StaticMatrix}}, b::Type)
    return quote
        return $(gen_by_access(b, :wrapped_b) do access_b
            expr_gen(:transpose, access_b)
        end)
    end
end
function gen_by_access(expr_gen, a::Type{<:Adjoint{<:Any, <:StaticMatrix}}, b::Type)
    return quote
        return $(gen_by_access(b, :wrapped_b) do access_b
            expr_gen(:adjoint, access_b)
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

"""
    uplo_access(sa, asym, k, j, uplo)

Generate code for matrix element access, for a matrix of size `sa` locally referred to
as `asym` in the context where the result will be used. Both indices `k` and `j` need to be
statically known for this function to work. `uplo` is the access pattern mode generated
by the `gen_by_access` function.
"""
function uplo_access(sa, asym, k, j, uplo)
    TAsym = Symbol("T"*string(asym))
    if uplo == :any
        return :($asym[$(LinearIndices(sa)[k, j])])
    elseif uplo == :up
        if k < j
            return :($asym[$(LinearIndices(sa)[k, j])])
        elseif k == j
            return :(LinearAlgebra.symmetric($asym[$(LinearIndices(sa)[k, j])], :U))
        else
            return :(transpose($asym[$(LinearIndices(sa)[j, k])]))
        end
    elseif uplo == :lo
        if k > j
            return :($asym[$(LinearIndices(sa)[k, j])])
        elseif k == j
            return :(LinearAlgebra.symmetric($asym[$(LinearIndices(sa)[k, j])], :L))
        else
            return :(transpose($asym[$(LinearIndices(sa)[j, k])]))
        end
    elseif uplo == :up_herm
        if k < j
            return :($asym[$(LinearIndices(sa)[k, j])])
        elseif k == j
            return :(LinearAlgebra.hermitian($asym[$(LinearIndices(sa)[k, j])], :U))
        else
            return :(adjoint($asym[$(LinearIndices(sa)[j, k])]))
        end
    elseif uplo == :lo_herm
        if k > j
            return :($asym[$(LinearIndices(sa)[k, j])])
        elseif k == j
            return :(LinearAlgebra.hermitian($asym[$(LinearIndices(sa)[k, j])], :L))
        else
            return :(adjoint($asym[$(LinearIndices(sa)[j, k])]))
        end
    elseif uplo == :upper_triangular
        if k <= j
            return :($asym[$(LinearIndices(sa)[k, j])])
        else
            return :(zero($TAsym))
        end
    elseif uplo == :lower_triangular
        if k >= j
            return :($asym[$(LinearIndices(sa)[k, j])])
        else
            return :(zero($TAsym))
        end
    elseif uplo == :unit_upper_triangular
        if k < j
            return :($asym[$(LinearIndices(sa)[k, j])])
        elseif k == j
            return :(oneunit($TAsym))
        else
            return :(zero($TAsym))
        end
    elseif uplo == :unit_lower_triangular
        if k > j
            return :($asym[$(LinearIndices(sa)[k, j])])
        elseif k == j 
            return :(oneunit($TAsym))
        else
            return :(zero($TAsym))
        end
    elseif uplo == :upper_hessenberg
        if k <= j+1
            return :($asym[$(LinearIndices(sa)[k, j])])
        else
            return :(zero($TAsym))
        end
    elseif uplo == :transpose
        return :(transpose($asym[$(LinearIndices(reverse(sa))[j, k])]))
    elseif uplo == :adjoint
        return :(adjoint($asym[$(LinearIndices(reverse(sa))[j, k])]))
    else
        error("Unknown uplo: $uplo")
    end
end

# Implementations

function mul_smat_vec_exprs(sa, access_a)
    return [combine_products([:($(uplo_access(sa, :a, k, j, access_a))*b[$j]) for j = 1:sa[2]]) for k = 1:sa[1]]
end

@generated function _mul(::Size{sa}, wrapped_a::StaticMatMulLike{<:Any, <:Any, Ta}, b::AbstractVector{Tb}) where {sa, Ta, Tb}
    if sa[2] != 0
        retexpr = gen_by_access(wrapped_a) do access_a
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
        a = mul_parent(wrapped_a)
        $retexpr
    end
end

@generated function _mul(::Size{sa}, ::Size{sb}, wrapped_a::StaticMatMulLike{<:Any, <:Any, Ta}, b::StaticVector{<:Any, Tb}) where {sa, sb, Ta, Tb}
    if sb[1] != sa[2]
        throw(DimensionMismatch("Tried to multiply arrays of size $sa and $sb"))
    end

    if sa[2] != 0
        retexpr = gen_by_access(wrapped_a) do access_a
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
        a = mul_parent(wrapped_a)
        $retexpr
    end
end

# outer product
@generated function _mul(::Size{sa}, ::Size{sb}, a::StaticVector{<: Any, Ta},
        b::Union{Transpose{Tb, <:StaticVector}, Adjoint{Tb, <:StaticVector}}) where {sa, sb, Ta, Tb}
    newsize = (sa[1], sb[2])
    conjugate_b = b <: Adjoint
    if conjugate_b
        exprs = [:(a[$i] * adjoint(b[$j])) for i = 1:sa[1], j = 1:sb[2]]
    else
        exprs = [:(a[$i] * transpose(b[$j])) for i = 1:sa[1], j = 1:sb[2]]
    end
    
    return quote
        @_inline_meta
        T = promote_op(*, Ta, Tb)
        @inbounds return similar_type(b, T, Size($newsize))(tuple($(exprs...)))
    end
end

_unstatic_array(::Type{TSA}) where {S, T, N, TSA<:StaticArray{S,T,N}} = AbstractArray{T,N}
for TWR in [Adjoint, Transpose, Symmetric, Hermitian, LowerTriangular, UpperTriangular]
    @eval _unstatic_array(::Type{$TWR{T,TSA}}) where {S, T, N, TSA<:StaticArray{S,T,N}} = $TWR{T,<:AbstractArray{T,N}}
end

function combine_products(expr_list)
    filtered = filter(expr_list) do expr
        if expr.head != :call || expr.args[1] != :*
            error("expected call to *")
        end
        for arg in expr.args[2:end]
            if isa(arg, Expr) && arg.head == :call && arg.args[1] == :zero
                return false
            end
        end
        return true
    end
    if isempty(filtered)
        return :(zero(T))
    else
        return reduce((ex1,ex2) -> :(+($ex1,$ex2)), filtered)
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
    elseif a <: StaticMatrix && b <:StaticMatrix
        return quote
            @_inline_meta
            return mul_loop(Sa, Sb, a, b)
        end
    else
        # we don't have any special code for handling this case so let's fall back to
        # the generic implementation of matrix multiplication
        return quote
            @_inline_meta
            return invoke(*, Tuple{$(_unstatic_array(a)),$(_unstatic_array(b))}, a, b)
        end
    end
end

@generated function mul_unrolled(::Size{sa}, ::Size{sb}, wrapped_a::StaticMatMulLike{<:Any, <:Any, Ta}, wrapped_b::StaticMatMulLike{<:Any, <:Any, Tb}) where {sa, sb, Ta, Tb}
    if sb[1] != sa[2]
        throw(DimensionMismatch("Tried to multiply arrays of size $sa and $sb"))
    end

    S = Size(sa[1], sb[2])

    if sa[2] != 0
        retexpr = gen_by_access(wrapped_a, wrapped_b) do access_a, access_b
            exprs = [combine_products([:($(uplo_access(sa, :a, k1, j, access_a))*$(uplo_access(sb, :b, j, k2, access_b))) for j = 1:sa[2]]
                ) for k1 = 1:sa[1], k2 = 1:sb[2]]
            return :((mul_result_structure(wrapped_a, wrapped_b))(similar_type(a, T, $S)(tuple($(exprs...)))))
        end
    else
        exprs = [:(zero(T)) for k1 = 1:sa[1], k2 = 1:sb[2]]
        retexpr = :(return (mul_result_structure(wrapped_a, wrapped_b))(similar_type(a, T, $S)(tuple($(exprs...)))))
    end

    return quote
        @_inline_meta
        T = promote_op(matprod,Ta,Tb)
        a = mul_parent(wrapped_a)
        b = mul_parent(wrapped_b)
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
@generated function mul_unrolled_chunks(::Size{sa}, ::Size{sb}, wrapped_a::StaticMatMulLike{<:Any, <:Any, Ta}, wrapped_b::StaticMatMulLike{<:Any, <:Any, Tb}) where {sa, sb, Ta, Tb}
    if sb[1] != sa[2]
        throw(DimensionMismatch("Tried to multiply arrays of size $sa and $sb"))
    end

    S = Size(sa[1], sb[2])

    # Do a custom b[:, k2] to return a SVector (an isbitstype type) rather than (possibly) a mutable type. Avoids allocation == faster
    tmp_type_in = :(SVector{$(sb[1]), T})
    tmp_type_out = :(SVector{$(sa[1]), T})

    retexpr = gen_by_access(wrapped_a, wrapped_b) do access_a, access_b
        vect_exprs = [:($(Symbol("tmp_$k2"))::$tmp_type_out = partly_unrolled_multiply($(Size{sa}()), $(Size{(sb[1],)}()),
            a, $(Expr(:call, tmp_type_in, [uplo_access(sb, :b, i, k2, access_b) for i = 1:sb[1]]...)), $(Val(access_a)))::$tmp_type_out) for k2 = 1:sb[2]]

        exprs = [:($(Symbol("tmp_$k2"))[$k1]) for k1 = 1:sa[1], k2 = 1:sb[2]]

        return quote
            @inbounds $(Expr(:block, vect_exprs...))
            $(Expr(:block,
                :(@inbounds return (mul_result_structure(wrapped_a, wrapped_b))(similar_type(a, T, $S)(tuple($(exprs...)))))
            ))
        end
    end

    return quote
        @_inline_meta
        T = promote_op(matprod, Ta, Tb)
        a = mul_parent(wrapped_a)
        b = mul_parent(wrapped_b)
        $retexpr
    end
end

#
