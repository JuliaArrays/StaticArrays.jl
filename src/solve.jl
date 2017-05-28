@inline (\)(a::StaticMatrix{<:Any, <:Any, T}, b::StaticVector{<:Any, T}) where {T} = solve(Size(a), Size(b), a, b)
@inline (\)(a::Union{UpperTriangular{T, S}, LowerTriangular{T, S}} where {S<:StaticMatrix{<:Any, <:Any, T}}, b::StaticVector{<:Any, T}) where {T} = solve(Size(a.data), Size(b), a, b)
@inline (\)(a::Union{UpperTriangular{T, S}, LowerTriangular{T, S}} where {S<:StaticMatrix{<:Any, <:Any, T}}, b::StaticMatrix{<:Any, <:Any, T}) where {T} = solve(Size(a.data), Size(b), a, b)

# TODO: Ineffecient but requires some infrastructure (e.g. LU or QR) to make efficient so we fall back on inv for now
@inline solve(::Size, ::Size, a, b) = inv(a) * b

@inline function solve(::Size{(1,1)}, ::Size{(1,)}, a::StaticMatrix{<:Any, <:Any, Ta}, b::StaticVector{<:Any, Tb}) where {Ta, Tb}
    @inbounds return similar_type(b, typeof(b[1] \ a[1]))(b[1] \ a[1])
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

@generated function solve(::Size{sa}, ::Size{sb}, a::UpperTriangular{Ta, Sa} where {Sa<:StaticMatrix{<:Any, <:Any, Ta}}, b::StaticVector{<:Any, Tb}) where {sa, sb, Ta, Tb}
    if sa[1] != sb[1]
        throw(DimensionMismatch("right hand side b needs first dimension of size $(sa[1]), has size $(sb[1])"))
    end

    x = [Symbol("x$k") for k = 1:sb[1]]
    expr = [:($(x[i]) = $(reduce((ex1, ex2) -> :(-($ex1,$ex2)), [j == i ? :(b[$j]) : :(a[$i, $j]*$(x[j])) for j = i:sa[1]]))/a[$i, $i]) for i = sb[1]:-1:1]

    quote
      @_inline_meta
      T = typeof((zero(Ta)*zero(Tb) + zero(Ta)*zero(Tb))/one(Ta))
      @inbounds $(Expr(:block, expr...))
      @inbounds return similar_type(b, T)(tuple($(x...)))
    end
end

@generated function solve(::Size{sa}, ::Size{sb}, a::UpperTriangular{Ta, Sa} where {Sa<:StaticMatrix{<:Any, <:Any, Ta}}, b::StaticMatrix{<:Any, <:Any, Tb}) where {sa, sb, Ta, Tb}
    if sa[1] != sb[1]
        throw(DimensionMismatch("right hand side b needs first dimension of size $(sa[1]), has size $(sb[1])"))
    end

    x = [Symbol("x$k1$k2") for k1 = 1:sb[1], k2 = 1:sb[2]]
    expr = [:($(x[k1, k2]) = $(reduce((ex1, ex2) -> :(-($ex1,$ex2)), [j == k1 ? :(b[$j, $k2]) : :(a[$k1, $j]*$(x[j, k2])) for j = k1:sa[1]]))/a[$k1, $k1]) for k1 = sb[1]:-1:1, k2 = 1:sb[2]]

    quote
      @_inline_meta
      T = typeof((zero(Ta)*zero(Tb) + zero(Ta)*zero(Tb))/one(Ta))
      @inbounds $(Expr(:block, expr...))
      @inbounds return similar_type(b, T)(tuple($(x...)))
    end
end

@generated function solve(::Size{sa}, ::Size{sb}, a::LowerTriangular{Ta, Sa} where {Sa<:StaticMatrix{<:Any, <:Any, Ta}}, b::StaticVector{<:Any, Tb}) where {sa, sb, Ta, Tb}
    if sa[1] != sb[1]
        throw(DimensionMismatch("right hand side b needs first dimension of size $(sa[1]), has size $(sb[1])"))
    end

    x = [Symbol("x$k") for k = 1:sb[1]]
    expr = [:($(x[i]) = $(reduce((ex1, ex2) -> :(-($ex1,$ex2)), [j == i ? :(b[$j]) : :(a[$i, $j]*$(x[j])) for j = i:-1:1]))/a[$i, $i]) for i = 1:sb[1]]

    quote
      @_inline_meta
      T = typeof((zero(Ta)*zero(Tb) + zero(Ta)*zero(Tb))/one(Ta))
      @inbounds $(Expr(:block, expr...))
      @inbounds return similar_type(b, T)(tuple($(x...)))
    end
end

@generated function solve(::Size{sa}, ::Size{sb}, a::LowerTriangular{Ta, Sa} where {Sa<:StaticMatrix{<:Any, <:Any, Ta}}, b::StaticMatrix{<:Any, <:Any, Tb}) where {sa, sb, Ta, Tb}
    if sa[1] != sb[1]
        throw(DimensionMismatch("right hand side b needs first dimension of size $(sa[1]), has size $(sb[1])"))
    end

    x = [Symbol("x$k1$k2") for k1 = 1:sb[1], k2 = 1:sb[2]]
    expr = [:($(x[k1, k2]) = $(reduce((ex1, ex2) -> :(-($ex1,$ex2)), [j == k1 ? :(b[$j, $k2]) : :(a[$k1, $j]*$(x[j, k2])) for j = k1:-1:1]))/a[$k1, $k1]) for k1 = 1:sb[1], k2 = 1:sb[2]]

    quote
      @_inline_meta
      T = typeof((zero(Ta)*zero(Tb) + zero(Ta)*zero(Tb))/one(Ta))
      @inbounds $(Expr(:block, expr...))
      @inbounds return similar_type(b, T)(tuple($(x...)))
    end
end
