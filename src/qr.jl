_stabilize_type(::Type{T}) where {T} = typeof((one(T)*zero(T) + zero(T))/one(T))
_thin_must_hold(thin) = 
    thin || error("For the sake of type stability, `thin = true` must hold.")
import Base.qr

function qr(A::StaticMatrix, pivot::Type{Val{true}}; thin::Bool=true)
    _thin_must_hold(thin)
    Q0, R0, p0 = Base.qr(Matrix(A), pivot)
    T = _stabilize_type(eltype(A))
    QT = similar_type(A, T, Size(diagsize(A), diagsize(A)))
    RT = similar_type(A, T)
    PT = similar_type(A, Int, Size(Size(A)[2]))
    QT(Q0), RT(R0), PT(p0)
end
function qr(A::StaticMatrix, pivot::Type{Val{false}}; thin::Bool=true)
    _thin_must_hold(thin)
    Q0, R0 = Base.qr(Matrix(A), pivot)
    T = _stabilize_type(eltype(A))
    QT = similar_type(A, T, Size(diagsize(A), diagsize(A)))
    RT = similar_type(A, T)
    QT(Q0), RT(R0)
end
