@inline (\)(a::StaticMatrix, b::StaticVecOrMat) = solve(Size(a), Size(b), a, b)

@inline function solve(::Size{(1,1)}, ::Size{(1,)}, a::StaticMatrix{<:Any, <:Any, Ta}, b::StaticVector{<:Any, Tb}) where {Ta, Tb}
    @inbounds return similar_type(b, typeof(a[1] \ b[1]))(a[1] \ b[1])
end

@inline function solve(::Size{(2,2)}, ::Size{(2,)}, a::StaticMatrix{<:Any, <:Any, Ta}, b::StaticVector{<:Any, Tb}) where {Ta, Tb}
    d = det(a)
    T = typeof((one(Ta)*zero(Tb) + one(Ta)*zero(Tb))/d)
    @inbounds return similar_type(b, T)((a[2,2]*b[1] - a[1,2]*b[2])/d,
                                        (a[1,1]*b[2] - a[2,1]*b[1])/d)
end

@inline function solve(::Size{(3,3)}, ::Size{(3,)}, a::StaticMatrix{<:Any, <:Any, Ta}, b::StaticVector{<:Any, Tb}) where {Ta, Tb}
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

@generated function solve(::Size{Sa}, ::Size{Sb}, a::StaticMatrix{<:Any, <:Any, Ta}, b::StaticVecOrMat{Tb}) where {Sa, Sb, Ta, Tb}
    if Sa[end] != Sb[1]
        throw(DimensionMismatch("right hand side B needs first dimension of size $(Sa[end]), has size $Sb"))
    end
    LinearAlgebra.checksquare(a)
    if prod(Sa) ≤ 14*14
        quote
            @_inline_meta
            LUp = lu(a)
            LUp.U \ (LUp.L \ $(length(Sb) > 1 ? :(b[LUp.p,:]) : :(b[LUp.p])))
        end
    else
        quote
            @_inline_meta
            T = typeof((one(Ta)*zero(Tb) + one(Ta)*zero(Tb))/one(Ta))
            similar_type(b, T)(Matrix(a) \ b)
        end
    end
end
