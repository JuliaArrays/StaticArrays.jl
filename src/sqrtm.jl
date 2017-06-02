@inline sqrtm(A::StaticMatrix) = _sqrtm(Size(A),A)

@inline function _sqrtm(::Size{(1,1)}, A::SA) where {SA<:StaticArray}
    s = sqrt(A[1,1])
    similar_type(SA,typeof(s))(s)
end

@inline function _sqrtm(::Size{(2,2)}, A::SA) where {SA<:StaticArray}
    a,b,c,d = A
    if a==b==c==d==0
        zero(A)
    else
        s = sqrtm(a*d-b*c)
        t = inv(sqrtm(a+d+2s))
        similar_type(SA,typeof(t))(t*(a+s), t*b, t*c, t*(d+s))
    end
end

@inline _sqrtm(s::Size, A::StaticArray) = s(Base.sqrtm(Array(A)))
