_thin_must_hold(thin) = 
    thin || throw(ArgumentError("For the sake of type stability, `thin = true` must hold."))
import Base.qr

function qr(A::StaticMatrix, pivot::Type{Val{true}}; thin::Bool=true)
    _thin_must_hold(thin)
    Q0, R0, p0 = Base.qr(Matrix(A), pivot)
    T = arithmetic_closure(eltype(A))
    QT = similar_type(A, T, Size(diagsize(A), diagsize(A)))
    RT = similar_type(A, T)
    PT = similar_type(A, Int, Size(Size(A)[2]))
    QT(Q0), RT(R0), PT(p0)
end
function qr(A::StaticMatrix, pivot::Type{Val{false}}; thin::Bool=true)
    _thin_must_hold(thin)
    Q0, R0 = Base.qr(Matrix(A), pivot)
    T = arithmetic_closure(eltype(A))
    QT = similar_type(A, T, Size(diagsize(A), diagsize(A)))
    RT = similar_type(A, T)
    QT(Q0), RT(R0)
end
