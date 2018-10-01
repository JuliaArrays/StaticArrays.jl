# adapted from Julia LinearAlgebra

"""
    SSchur

Schur decomposition of a matrix.
"""
struct SSchur{TT<:MMatrix,TZ<:SMatrix,TValues<:SVector}
    "an upper triangular matrix"
    T::TT # LAPACK.gees! mutates its first argument, so this matrix has to be mutable
    "Schur vectors (for transforming `T` back to original form)"
    Z::TZ
    "eigenvalues of a given matrix"
    values::TValues
end
SSchur(T::TT, Z::TZ, values::TValues) where {TT,TZ,TValues} = SSchur{TT,TZ,TValues}(T, Z, values)

function schur(A::SMatrix{N,N,<:BlasFloat}) where {N}
    t,z,values = LinearAlgebra.LAPACK.gees!('V', MMatrix{N,N}(A))
    SSchur(t, SMatrix{N,N}(z), SVector{N}(values))
end

function schur(A::MMatrix{N,N,<:BlasFloat}) where {N}
    t,z,values = LinearAlgebra.LAPACK.gees!('V', copy(A))
    SSchur(t, SMatrix{N,N}(z), SVector{N}(values))
end
