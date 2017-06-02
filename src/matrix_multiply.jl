import Base: *,        Ac_mul_B,  A_mul_Bc,  Ac_mul_Bc,  At_mul_B,  A_mul_Bt,  At_mul_Bt
import Base: A_mul_B!, Ac_mul_B!, A_mul_Bc!, Ac_mul_Bc!, At_mul_B!, A_mul_Bt!, At_mul_Bt!

import Base.LinAlg: BlasFloat, matprod

const StaticVecOrMat{T} = Union{StaticVector{<:Any, T}, StaticMatrix{<:Any, <:Any, T}}

# TODO Potentially a loop version for rather large arrays? Or try and figure out inference problems?

# Deal with A_mul_Bc, etc...
# TODO make faster versions of A*_mul_B*
@inline A_mul_Bc(A::StaticVecOrMat, B::StaticVecOrMat) = A * ctranspose(B)
@inline Ac_mul_Bc(A::StaticVecOrMat, B::StaticVecOrMat) = ctranspose(A) * ctranspose(B)
@inline Ac_mul_B(A::StaticVecOrMat, B::StaticVecOrMat) = ctranspose(A) * B

@inline A_mul_Bt(A::StaticVecOrMat, B::StaticVecOrMat) = A * transpose(B)
@inline At_mul_Bt(A::StaticVecOrMat, B::StaticVecOrMat) = transpose(A) * transpose(B)
@inline At_mul_B(A::StaticVecOrMat, B::StaticVecOrMat) = transpose(A) * B

@inline A_mul_Bc!(dest::StaticVecOrMat, A::StaticVecOrMat, B::StaticVecOrMat) = A_mul_B!(dest, A, ctranspose(B))
@inline Ac_mul_Bc!(dest::StaticVecOrMat, A::StaticVecOrMat, B::StaticVecOrMat) = A_mul_B!(dest, ctranspose(A), ctranspose(B))
@inline Ac_mul_B!(dest::StaticVecOrMat, A::StaticVecOrMat, B::StaticVecOrMat) = A_mul_B!(dest, ctranspose(A), B)

@inline A_mul_Bt!(dest::StaticVecOrMat, A::StaticVecOrMat, B::StaticVecOrMat) = A_mul_B!(dest, A, transpose(B))
@inline At_mul_Bt!(dest::StaticVecOrMat, A::StaticVecOrMat, B::StaticVecOrMat) = A_mul_B!(dest, transpose(A), transpose(B))
@inline At_mul_B!(dest::StaticVecOrMat, A::StaticVecOrMat, B::StaticVecOrMat) = A_mul_B!(dest, transpose(A), B)

# Manage dispatch of * and A_mul_B!
# TODO RowVector? (Inner product?)

@inline *(A::StaticMatrix, B::AbstractVector) = _A_mul_B(Size(A), A, B)
@inline *(A::StaticMatrix, B::StaticVector) = _A_mul_B(Size(A), Size(B), A, B)
@inline *(A::StaticMatrix, B::StaticMatrix) = _A_mul_B(Size(A), Size(B), A, B)
@inline *(A::StaticVector, B::StaticMatrix) = *(reshape(A, Size(Size(A)[1], 1)), B)
@inline *(A::StaticVector, B::RowVector{<:Any, <:StaticVector}) = _A_mul_B(Size(A), Size(B), A, B)
@inline *(A::StaticVector, B::RowVector{<:Any, <:ConjVector{<:Any, <:StaticVector}}) = _A_mul_B(Size(A), Size(B), A, B)

@inline A_mul_B!(dest::StaticVecOrMat, A::StaticMatrix, B::StaticVector) = _A_mul_B!(Size(dest), dest, Size(A), Size(B), A, B)
@inline A_mul_B!(dest::StaticVecOrMat, A::StaticMatrix, B::StaticMatrix) = _A_mul_B!(Size(dest), dest, Size(A), Size(B), A, B)
@inline A_mul_B!(dest::StaticVecOrMat, A::StaticVector, B::StaticMatrix) = A_mul_B!(dest, reshape(A, Size(Size(A)[1], 1)), B)
@inline A_mul_B!(dest::StaticVecOrMat, A::StaticVector, B::RowVector{<:Any, <:StaticVector}) = _A_mul_B!(Size(dest), dest, Size(A), Size(B), A, B)
@inline A_mul_B!(dest::StaticVecOrMat, A::StaticVector, B::RowVector{<:Any, <:ConjVector{<:Any, <:StaticVector}}) = _A_mul_B!(Size(dest), dest, Size(A), Size(B), A, B)

#@inline *{TA<:Base.LinAlg.BlasFloat,Tb}(A::StaticMatrix{TA}, b::StaticVector{Tb})

# Implementations

@generated function _A_mul_B(::Size{sa}, a::StaticMatrix{<:Any, <:Any, Ta}, b::AbstractVector{Tb}) where {sa, Ta, Tb}
    if sa[2] != 0
        exprs = [reduce((ex1,ex2) -> :(+($ex1,$ex2)), [:(a[$(sub2ind(sa, k, j))]*b[$j]) for j = 1:sa[2]]) for k = 1:sa[1]]
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

@generated function _A_mul_B(::Size{sa}, ::Size{sb}, a::StaticMatrix{<:Any, <:Any, Ta}, b::StaticVector{<:Any, Tb}) where {sa, sb, Ta, Tb}
    if sb[1] != sa[2]
        throw(DimensionMismatch("Tried to multiply arrays of size $sa and $sb"))
    end

    if sa[2] != 0
        exprs = [reduce((ex1,ex2) -> :(+($ex1,$ex2)), [:(a[$(sub2ind(sa, k, j))]*b[$j]) for j = 1:sa[2]]) for k = 1:sa[1]]
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
@generated function _A_mul_B(::Size{sa}, ::Size{sb}, a::StaticVector{<: Any, Ta}, b::RowVector{Tb, <:StaticVector}) where {sa, sb, Ta, Tb}
    newsize = (sa[1], sb[2])
    exprs = [:(a[$i]*b[$j]) for i = 1:sa[1], j = 1:sb[2]]

    return quote
        @_inline_meta
        T = promote_op(*, Ta, Tb)
        @inbounds return similar_type(b, T, Size($newsize))(tuple($(exprs...)))
    end
end

# complex outer product
@generated function _A_mul_B(::Size{sa}, ::Size{sb}, a::StaticVector{<: Any, Ta}, b::RowVector{Tb, <:ConjVector{<:Any, <:StaticVector}}) where {sa, sb, Ta, Tb}
    newsize = (sa[1], sb[2])
    exprs = [:(a[$i]*b[$j]) for i = 1:sa[1], j = 1:sb[2]]

    return quote
        @_inline_meta
        T = promote_op(*, Ta, Tb)
        @inbounds return similar_type(b, T, Size($newsize))(tuple($(exprs...)))
    end
end

@generated function _A_mul_B(Sa::Size{sa}, Sb::Size{sb}, a::StaticMatrix{<:Any, <:Any, Ta}, b::StaticMatrix{<:Any, <:Any, Tb}) where {sa, sb, Ta, Tb}
    # Heuristic choice for amount of codegen
    if sa[1]*sa[2]*sb[2] <= 8*8*8
        return quote
            @_inline_meta
            return A_mul_B_unrolled(Sa, Sb, a, b)
        end
    elseif sa[1] <= 14 && sa[2] <= 14 && sb[2] <= 14
        return quote
            @_inline_meta
            return A_mul_B_unrolled_chunks(Sa, Sb, a, b)
        end
    else
        return quote
            @_inline_meta
            return A_mul_B_loop(Sa, Sb, a, b)
        end
    end
end

@generated function _A_mul_B(Sa::Size{sa}, Sb::Size{sb}, a::Union{SizedMatrix{T}, MMatrix{T}, MArray{T}}, b::Union{SizedMatrix{T}, MMatrix{T}, MArray{T}}) where {sa, sb, T <: BlasFloat}
    S = Size(sa[1], sb[2])

    # Heuristic choice between BLAS and explicit unrolling (or chunk-based unrolling)
    if sa[1]*sa[2]*sb[2] >= 14*14*14
        return quote
            @_inline_meta
            C = similar(a, T, $S)
            A_mul_B_blas!($S, C, Sa, Sb, a, b)
            return C
        end
    elseif sa[1]*sa[2]*sb[2] < 8*8*8
        return quote
            @_inline_meta
            return A_mul_B_unrolled(Sa, Sb, a, b)
        end
    elseif sa[1] <= 14 && sa[2] <= 14 && sb[2] <= 14
        return quote
            @_inline_meta
            return similar_type(a, T, $S)(A_mul_B_unrolled_chunks(Sa, Sb, a, b))
        end
    else
        return quote
            @_inline_meta
            return A_mul_B_loop(Sa, Sb, a, b)
        end
    end
end

@generated function A_mul_B_unrolled(::Size{sa}, ::Size{sb}, a::StaticMatrix{<:Any, <:Any, Ta}, b::StaticMatrix{<:Any, <:Any, Tb}) where {sa, sb, Ta, Tb}
    if sb[1] != sa[2]
        throw(DimensionMismatch("Tried to multiply arrays of size $sa and $sb"))
    end

    S = Size(sa[1], sb[2])

    if sa[2] != 0
        exprs = [reduce((ex1,ex2) -> :(+($ex1,$ex2)), [:(a[$(sub2ind(sa, k1, j))]*b[$(sub2ind(sb, j, k2))]) for j = 1:sa[2]]) for k1 = 1:sa[1], k2 = 1:sb[2]]
    else
        exprs = [:(zero(T)) for k1 = 1:sa[1], k2 = 1:sb[2]]
    end

    return quote
        @_inline_meta
        T = promote_op(matprod,Ta,Tb)
        @inbounds return similar_type(a, T, $S)(tuple($(exprs...)))
    end
end


@generated function A_mul_B_loop(::Size{sa}, ::Size{sb}, a::StaticMatrix{<:Any, <:Any, Ta}, b::StaticMatrix{<:Any, <:Any, Tb}) where {sa, sb, Ta, Tb}
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
@generated function A_mul_B_unrolled_chunks(::Size{sa}, ::Size{sb}, a::StaticMatrix{<:Any, <:Any, Ta}, b::StaticMatrix{<:Any, <:Any, Tb}) where {sa, sb, Ta, Tb}
    if sb[1] != sa[2]
        throw(DimensionMismatch("Tried to multiply arrays of size $sa and $sb"))
    end

    S = Size(sa[1], sb[2])

    # Do a custom b[:, k2] to return a SVector (an isbits type) rather than (possibly) a mutable type. Avoids allocation == faster
    tmp_type_in = :(SVector{$(sa[1]), T})
    tmp_type_out = :(SVector{$(sb[1]), T})
    vect_exprs = [:($(Symbol("tmp_$k2"))::$tmp_type_out = partly_unrolled_multiply(Size(a), Size($(sa[1])), a, $(Expr(:call, tmp_type_in, [Expr(:ref, :b, sub2ind(S, i, k2)) for i = 1:sb[1]]...)))::$tmp_type_out) for k2 = 1:sb[2]]

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

@generated function partly_unrolled_multiply(::Size{sa}, ::Size{sb}, a::StaticMatrix{<:Any, <:Any, Ta}, b::StaticArray{<:Any, Tb}) where {sa, sb, Ta, Tb}
    if sa[2] != sb[1]
        throw(DimensionMismatch("Tried to multiply arrays of size $sa and $sb"))
    end

    if sa[2] != 0
        exprs = [reduce((ex1,ex2) -> :(+($ex1,$ex2)), [:(a[$(sub2ind(sa, k, j))]*b[$j]) for j = 1:sa[2]]) for k = 1:sa[1]]
    else
        exprs = [:(zero(promote_op(matprod,Ta,Tb))) for k = 1:sa[1]]
    end

    return quote
        $(Expr(:meta,:noinline))
        @inbounds return SVector(tuple($(exprs...)))
    end
end

# TODO aliasing problems if c === b?
@generated function _A_mul_B!(::Size{sc}, c::StaticVector, ::Size{sa}, ::Size{sb}, a::StaticMatrix, b::StaticVector) where {sa, sb, sc}
    if sb[1] != sa[2] || sc[1] != sa[1]
        throw(DimensionMismatch("Tried to multiply arrays of size $sa and $sb and assign to array of size $sc"))
    end

    if sa[2] != 0
        exprs = [:(c[$k] = $(reduce((ex1,ex2) -> :(+($ex1,$ex2)), [:(a[$(sub2ind(sa, k, j))]*b[$j]) for j = 1:sa[2]]))) for k = 1:sa[1]]
    else
        exprs = [:(c[$k] = zero(eltype(c))) for k = 1:sa[1]]
    end

    return quote
        @_inline_meta
        @inbounds $(Expr(:block, exprs...))
        return c
    end
end

@generated function _A_mul_B!(::Size{sc}, c::StaticMatrix, ::Size{sa}, ::Size{sb}, a::StaticVector, b::RowVector{<:Any, <:StaticVector}) where {sa, sb, sc}
    if sa[1] != sc[1] || sb[2] != sc[2]
        throw(DimensionMismatch("Tried to multiply arrays of size $sa and $sb and assign to array of size $sc"))
    end

    exprs = [:(c[$(sub2ind(sc, i, j))] = a[$i] * b[$j]) for i = 1:sa[1], j = 1:sb[2]]

    return quote
        @_inline_meta
        @inbounds $(Expr(:block, exprs...))
        return c
    end
end

@generated function _A_mul_B!(Sc::Size{sc}, c::StaticMatrix{<:Any, <:Any, Tc}, Sa::Size{sa}, Sb::Size{sb}, a::StaticMatrix{<:Any, <:Any, Ta}, b::StaticMatrix{<:Any, <:Any, Tb}) where {sa, sb, sc, Ta, Tb, Tc}
    can_blas = Tc == Ta && Tc == Tb && Tc <: BlasFloat

    if can_blas
        if sa[1] * sa[2] * sb[2] < 4*4*4
            return quote
                @_inline_meta
                A_mul_B_unrolled!(Sc, c, Sa, Sb, a, b)
                return c
            end
        elseif sa[1] * sa[2] * sb[2] < 14*14*14 # Something seems broken for this one with large matrices (becomes allocating)
            return quote
                @_inline_meta
                A_mul_B_unrolled_chunks!(Sc, c, Sa, Sb, a, b)
                return c
            end
        else
            return quote
                @_inline_meta
                A_mul_B_blas!(Sc, c, Sa, Sb, a, b)
                return c
            end
        end
    else
        if sa[1] * sa[2] * sb[2] < 4*4*4
            return quote
                @_inline_meta
                A_mul_B_unrolled!(Sc, c, Sa, Sb, a, b)
                return c
            end
        else
            return quote
                @_inline_meta
                A_mul_B_unrolled_chunks!(Sc, c, Sa, Sb, a, b)
                return c
            end
        end
    end
end


@generated function A_mul_B_blas!(::Size{s}, c::StaticMatrix{<:Any, <:Any, T}, ::Size{sa}, ::Size{sb}, a::StaticMatrix{<:Any, <:Any, T}, b::StaticMatrix{<:Any, <:Any, T}) where {s,sa,sb, T <: BlasFloat}
    if sb[1] != sa[2] || sa[1] != s[1] || sb[2] != s[2]
        throw(DimensionMismatch("Tried to multiply arrays of size $sa and $sb and assign to array of size $s"))
    end

    if sa[1] > 0 && sa[2] > 0 && sb[2] > 0
        # This code adapted from `gemm!()` in base/linalg/blas.jl

        if T == Float64
            gemm = :dgemm_
        elseif T == Float32
            gemm = :sgemm_
        elseif T == Complex{Float64}
            gemm = :zgemm_
        else # T == Complex{Float32}
            gemm = :cgemm_
        end

        return quote
            alpha = one(T)
            beta = zero(T)
            transA = 'N'
            transB = 'N'
            m = $(sa[1])
            ka = $(sa[2])
            kb = $(sb[1])
            n = $(sb[2])
            strideA = $(sa[1])
            strideB = $(sb[1])
            strideC = $(s[1])

            ccall((Base.BLAS.@blasfunc($gemm), Base.BLAS.libblas), Void,
                (Ptr{UInt8}, Ptr{UInt8}, Ptr{Base.BLAS.BlasInt}, Ptr{Base.BLAS.BlasInt},
                 Ptr{Base.BLAS.BlasInt}, Ptr{$T}, Ptr{$T}, Ptr{Base.BLAS.BlasInt},
                 Ptr{$T}, Ptr{Base.BLAS.BlasInt}, Ptr{$T}, Ptr{$T},
                 Ptr{Base.BLAS.BlasInt}),
                 &transA, &transB, &m, &n,
                 &ka, &alpha, a, &strideA,
                 b, &strideB, &beta, c,
                 &strideC)
            return c
        end
    else
        throw(DimensionMismatch("Cannot call BLAS gemm with zero-dimension arrays, attempted $sa * $sb -> $s."))
    end
end


@generated function A_mul_B_unrolled!(::Size{sc}, c::StaticMatrix, ::Size{sa}, ::Size{sb}, a::StaticMatrix, b::StaticMatrix) where {sa, sb, sc}
    if sb[1] != sa[2] || sa[1] != sc[1] || sb[2] != sc[2]
        throw(DimensionMismatch("Tried to multiply arrays of size $sa and $sb and assign to array of size $sc"))
    end

    if sa[2] != 0
        exprs = [:(c[$(sub2ind(sc, k1, k2))] = $(reduce((ex1,ex2) -> :(+($ex1,$ex2)), [:(a[$(sub2ind(sa, k1, j))]*b[$(sub2ind(sb, j, k2))]) for j = 1:sa[2]]))) for k1 = 1:sa[1], k2 = 1:sb[2]]
    else
        exprs = [:(c[$(sub2ind(sc, k1, k2))] = zero(eltype(c))) for k1 = 1:sa[1], k2 = 1:sb[2]]
    end

    return quote
        @_inline_meta
        @inbounds $(Expr(:block, exprs...))
    end
end

@generated function A_mul_B_unrolled_chunks!(::Size{sc}, c::StaticMatrix, ::Size{sa}, ::Size{sb}, a::StaticMatrix, b::StaticMatrix) where {sa, sb, sc}
    if sb[1] != sa[2] || sa[1] != sc[1] || sb[2] != sc[2]
        throw(DimensionMismatch("Tried to multiply arrays of size $sa and $sb and assign to array of size $sc"))
    end

    #vect_exprs = [:($(Symbol("tmp_$k2")) = partly_unrolled_multiply(A, B[:, $k2])) for k2 = 1:sB[2]]

    # Do a custom b[:, k2] to return a SVector (an isbits type) rather than a mutable type. Avoids allocation == faster
    tmp_type = SVector{sb[1], eltype(c)}
    vect_exprs = [:($(Symbol("tmp_$k2")) = partly_unrolled_multiply($(Size(sa)), $(Size(sb[1])), a, $(Expr(:call, tmp_type, [Expr(:ref, :b, sub2ind(sc, i, k2)) for i = 1:sb[1]]...)))) for k2 = 1:sb[2]]

    exprs = [:(c[$(sub2ind(sc, k1, k2))] = $(Symbol("tmp_$k2"))[$k1]) for k1 = 1:sa[1], k2 = 1:sb[2]]

    return quote
        @_inline_meta
        @inbounds $(Expr(:block, vect_exprs...))
        @inbounds $(Expr(:block, exprs...))
    end
end

#function A_mul_B_blas(a, b, c, A, B)
#q
#end

# The idea here is to get pointers to stack variables and call BLAS.
# This saves an aweful lot of time compared to copying SArray's to Ref{SArray{...}}
# and using BLAS should be fastest for (very) large SArrays

# Here is an LLVM function that gets the pointer to its input, %x
# After this we would make the ccall above.
#
# define i8* @f(i32 %x) #0 {
#     %1 = alloca i32, align 4
#     store i32 %x, i32* %1, align 4
#     ret i32* %1
# }
