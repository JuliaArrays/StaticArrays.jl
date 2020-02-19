
# 5-argument matrix multiplication
@inline LinearAlgebra.mul!(dest::StaticVecOrMat, A::StaticMatrix, B::StaticVecOrMat, α::Real, β::Real) =
    _mul!(Size(dest), dest, Size(A), Size(B), A, B, α, β)
@inline mul!(dest::StaticVecOrMat, A::StaticVector, B::Transpose{<:Any, <:StaticVector}, α::Real, β::Real) =
    _mul!(Size(dest), dest, Size(A), Size(B), A, B, α, β)
@inline mul!(dest::StaticVecOrMat, A::StaticVector, B::Adjoint{<:Any, <:StaticVector}, α::Real, β::Real) =
    _mul!(Size(dest), dest, Size(A), Size(B), A, B, α, β)

# Matrix-matrix multiplication
@generated function _mul!(Sc::Size{sc}, c::StaticMatrix{<:Any, <:Any, Tc},
        Sa::Size{sa}, Sb::Size{sb},
        a::StaticMatrix{<:Any, <:Any, Ta}, b::StaticMatrix{<:Any, <:Any, Tb},
        α::Real, β::Real) where {sa, sb, sc, Ta, Tb, Tc}
    can_blas = Tc == Ta && Tc == Tb && Tc <: BlasFloat

    if can_blas
        if sa[1] * sa[2] * sb[2] < 4*4*4
            return quote
                @_inline_meta
                muladd_unrolled!(Sc, c, Sa, Sb, a, b, α, β)
                return c
            end
        elseif sa[1] * sa[2] * sb[2] < 14*14*14 # Something seems broken for this one with large matrices (becomes allocating)
            return quote
                @_inline_meta
                muladd_unrolled_chunks!(Sc, c, Sa, Sb, a, b, α, β)
                return c
            end
        else
            return quote
                @_inline_meta
                mul_blas!(Sc, c, Sa, Sb, a, b, α, β)
                return c
            end
        end
    else
        if sa[1] * sa[2] * sb[2] < 4*4*4
            return quote
                @_inline_meta
                mul_unrolled!(Sc, c, Sa, Sb, a, b)
                return c
            end
        else
            return quote
                @_inline_meta
                mul_unrolled_chunks!(Sc, c, Sa, Sb, a, b)
                return c
            end
        end
    end
end

# Matrix-vector multiplication
@generated function _mul!(::Size{sc}, c::StaticVector, ::Size{sa}, ::Size{sb},
        a::StaticMatrix, b::StaticVector, α::Real, β::Real) where {sa, sb, sc}
    if sb[1] != sa[2] || sc[1] != sa[1]
        throw(DimensionMismatch("Tried to multiply arrays of size $sa and $sb and assign to array of size $sc"))
    end

    if sa[2] != 0
        exprs = [:(c[$k] = β * c[$k] + α * $(reduce((ex1,ex2) -> :(+($ex1,$ex2)),
            [:(a[$(LinearIndices(sa)[k, j])]*b[$j]) for j = 1:sa[2]]))) for k = 1:sa[1]]
    else
        exprs = [:(c[$k] = zero(eltype(c))) for k = 1:sa[1]]
    end

    return quote
        @_inline_meta
        @inbounds $(Expr(:block, exprs...))
        return c
    end
end

# Outer product
@generated function _mul!(::Size{sc}, c::StaticMatrix, ::Size{sa}, ::Size{sb}, a::StaticVector,
        b::Union{Transpose{<:Any, <:StaticVector}, Adjoint{<:Any, <:StaticVector}},
        α::Real, β::Real) where {sa, sb, sc}
    if sa[1] != sc[1] || sb[2] != sc[2]
        throw(DimensionMismatch("Tried to multiply arrays of size $sa and $sb and assign to array of size $sc"))
    end

    exprs = [:(c[$(LinearIndices(sc)[i, j])] = β * c[$(LinearIndices(sc)[i, j])] + α *
        a[$i] * b[$j]) for i = 1:sa[1], j = 1:sb[2]]

    return quote
        @_inline_meta
        @inbounds $(Expr(:block, exprs...))
        return c
    end
end

@generated function muladd_unrolled!(::Size{sc}, c::StaticMatrix, ::Size{sa}, ::Size{sb},
        a::StaticMatrix, b::StaticMatrix, α::Real, β::Real) where {sa, sb, sc}
    if sb[1] != sa[2] || sa[1] != sc[1] || sb[2] != sc[2]
        throw(DimensionMismatch("Tried to multiply arrays of size $sa and $sb and assign to array of size $sc"))
    end

    if sa[2] != 0
        exprs = [:(c[$(LinearIndices(sc)[k1, k2])] = β*c[$(LinearIndices(sc)[k1, k2])] + α *
            $(reduce((ex1,ex2) -> :(+($ex1,$ex2)),
                [:(a[$(LinearIndices(sa)[k1, j])]*b[$(LinearIndices(sb)[j, k2])]) for j = 1:sa[2]]
            ))) for k1 = 1:sa[1], k2 = 1:sb[2]]
    else
        exprs = [:(c[$(LinearIndices(sc)[k1, k2])] = zero(eltype(c))) for k1 = 1:sa[1], k2 = 1:sb[2]]
    end

    return quote
        @_inline_meta
        @inbounds $(Expr(:block, exprs...))
    end
end

@generated function muladd_unrolled_chunks!(::Size{sc}, c::StaticMatrix,
        ::Size{sa}, ::Size{sb}, a::StaticMatrix, b::StaticMatrix,
        α::Real, β::Real) where {sa, sb, sc}
    if sb[1] != sa[2] || sa[1] != sc[1] || sb[2] != sc[2]
        throw(DimensionMismatch("Tried to multiply arrays of size $sa and $sb and assign to array of size $sc"))
    end

    #vect_exprs = [:($(Symbol("tmp_$k2")) = partly_unrolled_multiply(A, B[:, $k2])) for k2 = 1:sB[2]]

    # Do a custom b[:, k2] to return a SVector (an isbitstype type) rather than a mutable type. Avoids allocation == faster
    tmp_type = SVector{sb[1], eltype(c)}
    vect_exprs = [:($(Symbol("tmp_$k2")) =
        partly_unrolled_multiply($(Size(sa)), $(Size(sb[1])), a,
        $(Expr(:call, tmp_type,
        [Expr(:ref, :b, LinearIndices(sb)[i, k2]) for i = 1:sb[1]]...)))) for k2 = 1:sb[2]]

    exprs = [:(c[$(LinearIndices(sc)[k1, k2])] = β*c[$(LinearIndices(sc)[k1, k2])] + α *
        $(Symbol("tmp_$k2"))[$k1]) for k1 = 1:sa[1], k2 = 1:sb[2]]

    return quote
        @_inline_meta
        @inbounds $(Expr(:block, vect_exprs...))
        @inbounds $(Expr(:block, exprs...))
    end
end

# Special-case SizedMatrix
@inline mul!(dest::SizedMatrix{<:Any, <:Any, Tc}, A::SizedMatrix{<:Any, <:Any, Ta},
    B::SizedMatrix{<:Any, <:Any, Tb}) where {Ta,Tb,Tc} =
        _mul!(Size(dest), dest, Size(A), Size(B), A, B, one(Ta), zero(Ta))

@generated function _mul!(Sc::Size{sc}, c::SizedMatrix{<:Any, <:Any, Tc},
        Sa::Size{sa}, Sb::Size{sb},
        a::SizedMatrix{<:Any, <:Any, Ta}, b::SizedMatrix{<:Any, <:Any, Tb},
        α::Real, β::Real) where {sa, sb, sc, Ta, Tb, Tc}
    can_blas = Tc == Ta && Tc == Tb && Tc <: BlasFloat

    if can_blas
        if sa[1] * sa[2] * sb[2] < 4*4*4
            return quote
                @_inline_meta
                muladd_unrolled!(Sc, c, Sa, Sb, a, b, α, β)
                return c
            end
        elseif sa[1] * sa[2] * sb[2] < 14*14*14 # Something seems broken for this one with large matrices (becomes allocating)
            return quote
                @_inline_meta
                muladd_unrolled_chunks!(Sc, c, Sa, Sb, a, b, α, β)
                return c
            end
        else
            return quote
                # @_inline_meta
                BLAS.gemm!('N','N', α, a.data, b.data, β, c.data)
                return c
            end
        end
    end
end
