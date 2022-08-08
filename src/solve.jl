@inline (\)(a::StaticMatrix, b::StaticVecOrMat) = _solve(Size(a), Size(b), a, b)

@inline function _solve(::Size{(1,1)}, ::Size{(1,)}, a::StaticMatrix{<:Any, <:Any, Ta}, b::StaticVector{<:Any, Tb}) where {Ta, Tb}
    @inbounds return similar_type(b, typeof(a[1] \ b[1]))(a[1] \ b[1])
end

@inline function _solve(::Size{(2,2)}, ::Size{(2,)}, a::StaticMatrix{<:Any, <:Any, Ta}, b::StaticVector{<:Any, Tb}) where {Ta, Tb}
    d = det(a)
    T = typeof((one(Ta)*zero(Tb) + one(Ta)*zero(Tb))/d)
    @inbounds return similar_type(b, T)((a[2,2]*b[1] - a[1,2]*b[2])/d,
                                        (a[1,1]*b[2] - a[2,1]*b[1])/d)
end

@inline function _solve(::Size{(3,3)}, ::Size{(3,)}, a::StaticMatrix{<:Any, <:Any, Ta}, b::StaticVector{<:Any, Tb}) where {Ta, Tb}
    d = det(a)
    T = typeof((one(Ta)*zero(Tb) + one(Ta)*zero(Tb))/d)
    @inbounds return similar_type(b, T)(
        ((a[2,2]*a[3,3] - a[2,3]*a[3,2])*b[1] +
         (a[1,3]*a[3,2] - a[1,2]*a[3,3])*b[2] +
         (a[1,2]*a[2,3] - a[1,3]*a[2,2])*b[3]) / d,
        ((a[2,3]*a[3,1] - a[2,1]*a[3,3])*b[1] +
         (a[1,1]*a[3,3] - a[1,3]*a[3,1])*b[2] +
         (a[1,3]*a[2,1] - a[1,1]*a[2,3])*b[3]) / d,
        ((a[2,1]*a[3,2] - a[2,2]*a[3,1])*b[1] +
         (a[1,2]*a[3,1] - a[1,1]*a[3,2])*b[2] +
         (a[1,1]*a[2,2] - a[1,2]*a[2,1])*b[3]) / d )
end

for Sa in [(2,2), (3,3)]  # not needed for Sa = (1, 1);
    @eval begin
        @inline function _solve(::Size{$Sa}, ::Size{Sb}, a::StaticMatrix{<:Any, <:Any, Ta}, b::StaticMatrix{<:Any, <:Any, Tb}) where {Sb, Ta, Tb}
            d = det(a)
            T = typeof((one(Ta)*zero(Tb) + one(Ta)*zero(Tb))/d)
            if isbitstype(T)
                # This if block can be removed when https://github.com/JuliaArrays/StaticArrays.jl/pull/749 is merged.
                c = similar(b, T)
                for col in 1:Sb[2]
                    @inbounds c[:, col] = _solve(Size($Sa), Size($Sa[1],), a, b[:, col])
                end
                return similar_type(b, T)(c)
            else
                return _solve_general($(Size(Sa)), Size(Sb), a, b)
            end
        end
    end # @eval
end

@inline function _solve(sa::Size, sb::Size, a::StaticMatrix, b::StaticVecOrMat)
    _solve_general(sa, sb, a, b)
end

@generated function _solve_general(::Size{Sa}, ::Size{Sb}, a::StaticMatrix{<:Any, <:Any, Ta}, b::StaticVecOrMat{Tb}) where {Sa, Sb, Ta, Tb}
    if Sa[1] != Sb[1]
        return quote
            throw(DimensionMismatch("Left and right hand side first dimensions do not match in backdivide (got sizes $Sa and $Sb)"))
        end
    end
    if prod(Sa) ≤ 14*14 && Sa[1] == Sa[2]
        # TODO: Consider triangular special cases as in Base?
        quote
            @_inline_meta
            LUp = lu(a)
            LUp.U \ (LUp.L \ $(length(Sb) > 1 ? :(b[LUp.p,:]) : :(b[LUp.p])))
        end
        # TODO: Could also use static QR here if `a` is nonsquare.
        # Requires that we implement \(::StaticArrays.QR,::StaticVecOrMat)
    else
        # Fall back to LinearAlgebra, but carry across the statically known size.
        outsize = length(Sb) == 1 ? Size(Sa[2]) : Size(Sa[2],Sb[end])
        quote
            @_inline_meta
            T = typeof((one(Ta)*zero(Tb) + one(Ta)*zero(Tb))/one(Ta))
            similar_type(b, T, $outsize)(Matrix(a) \ b)
        end
    end
end
