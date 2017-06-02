@inline sqrtm(A::StaticMatrix) = _sqrtm(Size(A),A)

@inline function _sqrtm(::Size{(2,2)}, A::StaticMatrix)
    a,b,c,d = A
    s = sqrtm(a*d-b*c)
    t = inv(sqrtm(a+d+2s))
    similar_type(typeof(A),typeof(t))(t*(a+s), t*b, t*c, t*(d+s))
end

@inline _sqrtm(s::Size, A::StaticArray) = s(Base.sqrtm(Array(A)))
