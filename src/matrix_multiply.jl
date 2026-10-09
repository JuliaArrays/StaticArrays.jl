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
    mul_result_structure(a::Type, b::Type)

Get a structure wrapper that should be applied to the result of multiplication of matrices
of given types (`a*b`).
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
function mul_result_structure(::UpperTriangular{<:Any, <:StaticMatrix}, ::SDiagonal)
    return UpperTriangular
end
function mul_result_structure(::LowerTriangular{<:Any, <:StaticMatrix}, ::SDiagonal)
    return LowerTriangular
end
function mul_result_structure(::SDiagonal, ::UpperTriangular{<:Any, <:StaticMatrix})
    return UpperTriangular
end
function mul_result_structure(::SDiagonal, ::LowerTriangular{<:Any, <:StaticMatrix})
    return LowerTriangular
end
function mul_result_structure(::UnitUpperTriangular{<:Any, <:StaticMatrix}, ::SDiagonal)
    return UpperTriangular
end
function mul_result_structure(::UnitLowerTriangular{<:Any, <:StaticMatrix}, ::SDiagonal)
    return LowerTriangular
end
function mul_result_structure(::SDiagonal, ::UnitUpperTriangular{<:Any, <:StaticMatrix})
    return UpperTriangular
end
function mul_result_structure(::SDiagonal, ::UnitLowerTriangular{<:Any, <:StaticMatrix})
    return LowerTriangular
end
function mul_result_structure(::SDiagonal, ::SDiagonal)
    return Diagonal
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
            return :(@inbounds return similar_type(b, T, Size(sa[1]))(tuple($(exprs...))))
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
    exprs = [:(a[$i]*b[$j]) for i = 1:sa[1], j = 1:sb[2]]

    return quote
        @_inline_meta
        T = promote_op(*, Ta, Tb)
        @inbounds return similar_type(b, T, Size($newsize))(tuple($(exprs...)))
    end
end

_unstatic_array(::Type{TSA}) where {S, T, N, TSA<:StaticArray{S,T,N}} = AbstractArray{T,N}
for TWR in [Adjoint, Transpose, Symmetric, Hermitian, LowerTriangular, UpperTriangular, UnitUpperTriangular, UnitLowerTriangular, Diagonal]
    @eval _unstatic_array(::Type{$TWR{T,TSA}}) where {S, T, N, TSA<:StaticArray{S,T,N}} = $TWR{T,<:AbstractArray{T,N}}
end

@generated function _mul(Sa::Size{sa}, Sb::Size{sb}, a::StaticMatMulLike{<:Any, <:Any, Ta}, b::StaticMatMulLike{<:Any, <:Any, Tb}) where {sa, sb, Ta, Tb}
    if mul_tileable(a) && mul_tileable(b)
        return quote
            @_inline_meta
            return mul_tiled(Sa, Sb, a, b)
        end
    end

    # Heuristic choice for amount of codegen
    a_tri_mul = a <: LinearAlgebra.AbstractTriangular ? 4 : 1
    b_tri_mul = b <: LinearAlgebra.AbstractTriangular ? 4 : 1
    ab_tri_mul = (a_tri_mul == 4 && b_tri_mul == 4) ? 2 : 1
    if a <: StaticMatrix && b <: StaticMatrix
        # Julia unrolls these loops pretty well
        return quote
            @_inline_meta
            return mul_loop(Sa, Sb, a, b)
        end
    elseif sa[1]*sa[2]*sb[2] <= 4*8*8*8*a_tri_mul*b_tri_mul*ab_tri_mul || a <: Diagonal || b <: Diagonal
        return quote
            @_inline_meta
            return mul_unrolled(Sa, Sb, a, b)
        end
    elseif (sa[1] <= 14 && sa[2] <= 14 && sb[2] <= 14) || !(a <: StaticMatrix) || !(b <: StaticMatrix)
        return quote
            @_inline_meta
            return mul_unrolled_chunks(Sa, Sb, a, b)
        end
    else
        # we don't have any special code for handling this case so let's fall back to
        # the generic implementation of matrix multiplication
        return quote
            @_inline_meta
            return mul_generic(Sa, Sb, a, b)
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

    # optimal for AVX2 with `Float64
    # AVX512 would want something more like 16x14 or 24x9 with `Float64`
    M_r, N_r = 8, 6
    n = 0
    M, K = sa
    N = sb[2]
    q = Expr(:block)
    atemps = [Symbol(:a_, k1) for k1 = 1:M]
    tmps = [Symbol("tmp_$(k1)_$(k2)") for k1 = 1:M, k2 = 1:N]
    while n < N
        nu = min(N, n + N_r)
        nrange = n+1:nu
        m = 0
        while m < M
            mu = min(M, m + M_r)
            mrange = m+1:mu

            atemps_init = [:($(atemps[k1]) = a[$k1]) for k1 = mrange]
            exprs_init = [:($(tmps[k1,k2])  = $(atemps[k1]) * b[$(1 + (k2-1) * sb[1])]) for k1 = mrange, k2 = nrange]
            atemps_loop_init = [:($(atemps[k1]) = a[$(k1-sa[1]) + $(sa[1])*j]) for k1 = mrange]
            exprs_loop = [:($(tmps[k1,k2]) = muladd($(atemps[k1]), b[j + $((k2-1) * sb[1])], $(tmps[k1,k2]))) for k1 = mrange, k2 = nrange]
            qblock = quote
                @inbounds $(Expr(:block, atemps_init...))
                @inbounds $(Expr(:block, exprs_init...))
                for j = 2:$(sa[2])
                    @inbounds $(Expr(:block, atemps_loop_init...))
                    @inbounds $(Expr(:block, exprs_loop...))
                end
            end
            push!(q.args, qblock)
            m = mu
        end
        n = nu
    end
    return quote
        @_inline_meta
        T = promote_op(matprod,Ta,Tb)
        $q
        @inbounds return similar_type(a, T, $S)(tuple($(tmps...)))
    end
end

"""
    MUL_TILE_BUDGET

Maximum size in bytes of the block of accumulators that `mul_tiled!` unrolls, where the size
of an accumulator is approximated by the size of the elements of the factors.

Loops over blocks and over the inner dimension are not unrolled. Hence the amount of
generated arithmetic code, and thereby the compilation time, is bounded independently of
the size of the matrices and of the size of their elements (e.g., `ForwardDiff.Dual`
numbers, see #513). The value corresponds to the 8×6 block of `Float64` values of `mul_loop`.
"""
const MUL_TILE_BUDGET = 384

"""
    mul_tile(M, N, s, budget)

Size `(mr, nr)` of the block of accumulators of `mul_tiled!` for a product of size `M`×`N`
with elements of `s >= 1` bytes.

The block has at most 8 rows and satisfies `mr * nr * s <= budget`, unless a single
element exceeds the budget (then `mr = nr = 1`).

Elements of at least 32 bytes (the size of AVX2 registers) are not vectorized across rows,
so for them the rows are distributed evenly among the blocks: Otherwise the block at the
bottom edge might consist of only a few rows, i.e., few independent accumulators.
"""
function mul_tile(M::Int, N::Int, s::Int, budget::Int)
    mr = min(M, 8, max(1, budget ÷ s))
    nr = min(N, max(1, budget ÷ (s * mr)))
    if s >= 32
        mr = cld(M, cld(M, mr))
    end
    return mr, nr
end

# Names of the accumulators of an `mr`×`nr` block (see `mul_tile_expr`)
mul_tile_accs(mr::Int, nr::Int) = [Symbol(:acc_, r, :_, c) for r in 1:mr, c in 1:nr]

# Compute the `mr`×`nr` block of `a * b` with offsets `i0` and `j0`, where `a` has `M` rows
# and `b` has `K` rows, and store its entries with the expressions `store(i, j, acc)`.
# All indices are within bounds since `i0 + mr <= M`, `j0 + nr <= N`, and `1 <= k <= K`.
# Linear indices are used since, in contrast to `getindex` with two indices, `getindex` of an
# `SMatrix` with a linear index does not involve `checkbounds` (which, even though it is
# removed by `@inbounds`, increases the amount of code to be inferred and inlined).
function mul_tile_expr(store, M::Int, K::Int, mr::Int, nr::Int, @nospecialize(i0), @nospecialize(j0))
    avals = [Symbol(:a_, r) for r in 1:mr]
    bvals = [Symbol(:b_, c) for c in 1:nr]
    accs = mul_tile_accs(mr, nr)
    load_a(k) = [:($(avals[r]) = @inbounds a[$i0 + $r + ($k - 1) * $M]) for r in 1:mr]
    load_b(k) = [:($(bvals[c]) = @inbounds b[$k + ($j0 + $(c - 1)) * $K]) for c in 1:nr]
    return quote
        $(load_a(1)...)
        $(load_b(1)...)
        $([:($(accs[r, c]) = $(avals[r]) * $(bvals[c])) for r in 1:mr, c in 1:nr]...)
        for k in 2:$K
            $(load_a(:k)...)
            $(load_b(:k)...)
            $([:($(accs[r, c]) = muladd($(avals[r]), $(bvals[c]), $(accs[r, c]))) for r in 1:mr, c in 1:nr]...)
        end
        $([store(:($i0 + $r), :($j0 + $c), accs[r, c]) for r in 1:mr, c in 1:nr]...)
    end
end

# Compute columns `j0 + 1`, ..., `j0 + nc` of `a * b` in blocks with at most `mr` rows
function mul_tile_cols_expr(store, M::Int, K::Int, mr::Int, nc::Int, @nospecialize(j0))
    Mf = M - M % mr
    # No loop over a single block: Inference iterates over loops until convergence
    ex = if Mf == mr
        mul_tile_expr(store, M, K, mr, nc, 0, j0)
    else
        :(for i0 in 0:$mr:$(Mf - mr)
            $(mul_tile_expr(store, M, K, mr, nc, :i0, j0))
        end)
    end
    return Mf < M ? Expr(:block, ex, mul_tile_expr(store, M, K, M - Mf, nc, Mf, j0)) : ex
end

# Compute the `M`×`N` product `a * b` with inner dimension `K >= 1` in blocks of size
# `mul_tile(M, N, s, MUL_TILE_BUDGET)`, where `s` is the size of the elements in bytes, and
# store its entries with the expressions `store(i, j, acc)`.
# Only the computation of a single block is unrolled, so the generated code consists of at
# most four variants of the block (including the blocks at the bottom and right edges).
# The products are accumulated in the same order as in `mul_loop`.
function mul_tiled_expr(store, M::Int, K::Int, N::Int, s::Int)
    mr, nr = mul_tile(M, N, s, MUL_TILE_BUDGET)
    Nf = N - N % nr
    # No loop over a single column of blocks (see `mul_tile_cols_expr`)
    ex = if Nf == nr
        mul_tile_cols_expr(store, M, K, mr, nr, 0)
    else
        :(for j0 in 0:$nr:$(Nf - nr)
            $(mul_tile_cols_expr(store, M, K, mr, nr, :j0))
        end)
    end
    return Nf < N ? Expr(:block, ex, mul_tile_cols_expr(store, M, K, mr, N - Nf, Nf)) : ex
end

# Compute `a * b` in blocks with `mul_tiled!`
# For `isbitstype` elements, the temporary `MMatrix` is not allocated since it does not escape.
# If the product consists of a single block, its accumulators are returned directly instead.
# The function is inlined since its code is bounded and small products benefit from inlining.
@generated function mul_tiled(Sa::Size{sa}, Sb::Size{sb}, a::StaticMatrix{<:Any, <:Any, Ta}, b::StaticMatrix{<:Any, <:Any, Tb}) where {sa, sb, Ta, Tb}
    if sb[1] != sa[2]
        throw(DimensionMismatch("Tried to multiply arrays of size $sa and $sb"))
    end

    M, K = sa
    N = sb[2]
    if M > 0 && N > 0 && K > 0 && mul_tile(M, N, max(sizeof(Ta), sizeof(Tb), 1), MUL_TILE_BUDGET) == (M, N)
        return quote
            @_inline_meta
            T = promote_op(matprod, Ta, Tb)
            TC = similar_type(a, T, Size($M, $N))
            # `getindex` of an `MArray` preserves it for every element, unlike an `SMatrix`
            a = SMatrix{$M, $K, Ta}(Tuple(a))
            b = SMatrix{$K, $N, Tb}(Tuple(b))
            $(mul_tile_expr((i, j, acc) -> nothing, M, K, M, N, 0, 0))
            return TC(tuple($(mul_tile_accs(M, N)...)))
        end
    end

    return quote
        @_inline_meta
        T = promote_op(matprod, Ta, Tb)
        C = similar(SMatrix{$M, $N, T})
        mul_tiled!(TSize(C), C, Sa, Sb, a, b, NoMulAdd{T, T}())
        return similar_type(a, T, Size($M, $N))(Tuple(C))
    end
end

# Compute `c = a * b` or `c = c * β + α * (a * b)` (see `_muladd_expr`) in place
# `a` and `b` are copied to an `SMatrix` first, which ensures correct results if `c` aliases
# them. The function is inlined for the same reasons as `mul_tiled`.
# Elements of an `MArray` with `isbitstype` elements are accessed with a pointer within a
# single `GC.@preserve` block: `getindex` and `setindex!` preserve `c` for every element,
# which prevents LLVM from vectorizing the computation of the blocks (the preserved regions
# are only removed after vectorization since `c` is not allocated in the function).
@generated function mul_tiled!(::TSize{sc, :any}, c::StaticMatrix, ::Size{sa}, ::Size{sb}, a::StaticMatrix{<:Any, <:Any, Ta}, b::StaticMatrix{<:Any, <:Any, Tb}, _add::MulAddMul) where {sc, sa, sb, Ta, Tb}
    if !check_dims(Size(sc), Size(sa), Size(sb))
        throw(DimensionMismatch("Tried to multiply arrays of size $sa and $sb and assign to array of size $sc"))
    end

    M, K = sa
    N = sb[2]

    # The closure only captures `Int` and `Bool` values (no types), so its type does not
    # depend on the arguments and `mul_tiled_expr` is compiled only once
    use_pointer = c <: MArray && isbitstype(eltype(c))
    is_muladd = _add <: AlphaBeta
    function store(i, j, acc)
        # Linear index of entry `(i, j)` of `c` (all indices are within bounds, see `mul_tile_expr`)
        ind = :($i + ($j - 1) * $M)
        if use_pointer
            rhs = is_muladd ? :(unsafe_load(p, $ind) * β + α * $acc) : acc
            return :(unsafe_store!(p, $rhs, $ind))
        else
            rhs = is_muladd ? :(c[$ind] * β + α * $acc) : acc
            return :(@inbounds c[$ind] = $rhs)
        end
    end

    if M == 0 || N == 0
        ex = nothing
    elseif K == 0
        ex = :(for j in 1:$N, i in 1:$M
            $(store(:i, :j, :(zero(eltype(c)))))
        end)
    else
        ex = mul_tiled_expr(store, M, K, N, max(sizeof(Ta), sizeof(Tb), 1))
    end
    if use_pointer
        ex = :(GC.@preserve c begin
            p = pointer(c)
            $ex
        end)
    end
    return quote
        @_inline_meta
        α = alpha(_add)
        β = beta(_add)
        a = SMatrix{$M, $K, Ta}(Tuple(a))
        b = SMatrix{$K, $N, Tb}(Tuple(b))
        $ex
        return c
    end
end

@generated function mul_generic(::Size{sa}, ::Size{sb}, wrapped_a::StaticMatMulLike{<:Any, <:Any, Ta}, wrapped_b::StaticMatMulLike{<:Any, <:Any, Tb}) where {sa, sb, Ta, Tb}
    if sb[1] != sa[2]
        throw(DimensionMismatch("Tried to multiply arrays of size $sa and $sb"))
    end

    S = Size(sa[1], sb[2])

    return quote
        @_inline_meta
        T = promote_op(matprod, Ta, Tb)
        a = mul_parent(wrapped_a)
        b = mul_parent(wrapped_b)
        return (mul_result_structure(wrapped_a, wrapped_b))(similar_type(a, T, $S)(invoke(*, Tuple{$_unstatic_array(a),$_unstatic_array(b)}, a, b)))
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
        vect_exprs = [:($(Symbol("tmp_$k2")) = partly_unrolled_multiply($(Size{sa}()), $(Size{(sb[1],)}()),
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

# a special version for plain matrices
@generated function mul_unrolled_chunks(::Size{sa}, ::Size{sb}, a::StaticMatrix{<:Any, <:Any, Ta}, b::StaticMatrix{<:Any, <:Any, Tb}) where {sa, sb, Ta, Tb}
    if sb[1] != sa[2]
        throw(DimensionMismatch("Tried to multiply arrays of size $sa and $sb"))
    end

    S = Size(sa[1], sb[2])

    # optimal for AVX2 with `Float64
    # AVX512 would want something more like 16x14 or 24x9 with `Float64`
    M_r, N_r = 8, 6
    n = 0
    M, K = sa
    N = sb[2]
    q = Expr(:block)
    atemps = [Symbol(:a_, k1) for k1 = 1:M]
    tmps = [Symbol("tmp_$(k1)_$(k2)") for k1 = 1:M, k2 = 1:N]
    while n < N
        nu = min(N, n + N_r)
        nrange = n+1:nu
        m = 0
        while m < M
            mu = min(M, m + M_r)
            mrange = m+1:mu

            atemps_init = [:($(atemps[k1]) = a[$k1]) for k1 = mrange]
            exprs_init = [:($(tmps[k1,k2])  = $(atemps[k1]) * b[$(1 + (k2-1) * sb[1])]) for k1 = mrange, k2 = nrange]
            push!(q.args, :(@inbounds $(Expr(:block, atemps_init...))))
            push!(q.args, :(@inbounds $(Expr(:block, exprs_init...))))

            for j in 2:K
                atemps_loop_init = [:($(atemps[k1]) = a[$(LinearIndices(sa)[k1,j])]) for k1 = mrange]
                exprs_loop = [:($(tmps[k1,k2]) = muladd($(atemps[k1]), b[$(LinearIndices(sb)[j,k2])], $(tmps[k1,k2]))) for k1 = mrange, k2 = nrange]
                push!(q.args, :(@inbounds $(Expr(:block, atemps_loop_init...))))
                push!(q.args, :(@inbounds $(Expr(:block, exprs_loop...))))
            end
            m = mu
        end
        n = nu
    end
    return quote
        @_inline_meta
        T = promote_op(matprod,Ta,Tb)
        $q
        @inbounds return similar_type(a, T, $S)(tuple($(tmps...)))
    end
end

#
