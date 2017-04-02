@inline (\)(a::StaticMatrix{<:Any, <:Any, T}, b::StaticVector{<:Any, T}) where {T} = solve(Size(a), Size(b), a, b)

# TODO: Ineffective but requires some infrastructure (e.g. LU or QR) to make efficient so we fall back on inv for now
@inline solve(::Size, ::Size, a, b) = inv(a) * b

@inline solve(::Size{(1,1)}, ::Size{(1,)}, a, b) = similar_type(b, typeof(b[1] \ a[1]))(b[1] \ a[1])

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
