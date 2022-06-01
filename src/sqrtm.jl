
@inline sqrt(A::StaticMatrix) = _sqrt(Size(A),A)

@inline function _sqrt(::Size{(0,0)}, A::SA) where {SA<:StaticArray}
    similar_type(A,typeof(sqrt(zero(eltype(A)))))()
end

@inline function _sqrt(::Size{(1,1)}, A::SA) where {SA<:StaticArray}
    s = sqrt(A[1,1])
    similar_type(SA,typeof(s))(s)
end

@inline function _sqrt(::Size{(2,2)}, A::SA) where {SA<:StaticArray}
    a,b,c,d = A
    if a==b==c==d==0
        zero(A)
    else
        s = sqrt(a*d-b*c)
        t = inv(sqrt(a+d+2s))
        similar_type(SA,typeof(t))(t*(a+s), t*b, t*c, t*(d+s))
    end
end

@inline _sqrt(::Size{S}, A::StaticArray) where {S} = SizedArray{Tuple{S...}}(sqrt(Array(A)))
