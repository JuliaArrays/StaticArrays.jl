lyap(a::StaticMatrix,  c::StaticMatrix) = _lyap(Size(a), Size(c), a, c)

_lyap(::Size{(1,1)}, ::Size{(1,1)}, a::StaticMatrix,  c::StaticMatrix) = -c/(2a[1,1])

@inline function _lyap(::Size{(2,2)}, ::Size{(2,2)}, a::StaticMatrix,  c::StaticMatrix)
    d = det(a)
    t = trace(a)
     -(d*c  + (a - t*I)*c*(a-t*I)')/(2*d*t) # http://www.nber.org/papers/w8956.pdf
end

@inline _lyap(sa::Size, sc::Size, a::StaticMatrix, c::StaticMatrix) = sc(Base.lyap(Array(a),Array(c)))
 