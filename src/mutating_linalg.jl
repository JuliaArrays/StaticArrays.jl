# Mutating ("in-place") factorizations such as `eigen!` and `lu!` are allowed to use their
# input as workspace, but are not required to. Generic code calls them to save an
# allocation, so they must work for static matrices too: forward them to the corresponding
# non-mutating StaticArrays method.

# `eigen!`, `eigvals!`, `svd!`, `svdvals!`: LinearAlgebra has no generic fallback for these
# (only LAPACK-backed methods for strided matrices), so the MethodError hits mutable and
# immutable static matrices alike.
for (f!, f) in ((:eigen!, :eigen), (:eigvals!, :eigvals), (:svd!, :svd), (:svdvals!, :svdvals))
    @eval @inline LinearAlgebra.$f!(A::StaticMatrixLike; kws...) = $f(A; kws...)
end
# resolve ambiguity with LinearAlgebra's `svdvals!(::RealHermSymComplexHerm)`, which
# overwrites the eigenvalues of `A` in place and thus fails for immutable `A`
@inline function LinearAlgebra.svdvals!(
    A::LinearAlgebra.RealHermSymComplexHerm{<:Real,<:StaticMatrix}
)
    svdvals(A)
end

# `lu!`, `qr!`, `bunchkaufman!`: LinearAlgebra's generic fallbacks already work in place on
# mutable static matrices, so we only forward the immutable ones, which they would fail to
# `setindex!` into.
# The `pivot` argument is typed with each pivot type that the non-mutating method accepts,
# rather than left untyped: LinearAlgebra's generic `lu!(::AbstractMatrix, pivot::<type>)`
# and `qr!(::AbstractMatrix, pivot::<type>)` methods are more specific in `pivot`, so an
# untyped `pivot` would be ambiguous with them (the same holds for `lu` and `qr`, see
# lu.jl and qr.jl). `NoPivot` etc. only exist on Julia ≥ 1.7.
const SMatrixOrHermSym = Union{
    SMatrix,
    Symmetric{<:Any,<:SMatrix},
    Hermitian{<:Any,<:SMatrix}
}
@inline LinearAlgebra.lu!(A::SMatrixOrHermSym; kws...) = lu(A; kws...)
@inline LinearAlgebra.qr!(A::SMatrix) = qr(A)
let has_strategies = isdefined(LinearAlgebra, :PivotingStrategy)
    lu_pivots = has_strategies ? (:(Val{true}), :(Val{false}), :NoPivot, :RowMaximum) :
                                 (:(Val{true}), :(Val{false}))
    qr_pivots = has_strategies ? (:(Val{true}), :(Val{false}), :NoPivot, :ColumnNorm) :
                                 (:(Val{true}), :(Val{false}))
    for pv in lu_pivots
        @eval @inline function LinearAlgebra.lu!(A::SMatrixOrHermSym, pivot::$pv; kws...)
            lu(A, pivot; kws...)
        end
    end
    for pv in qr_pivots
        @eval @inline LinearAlgebra.qr!(A::SMatrix, pivot::$pv) = qr(A, pivot)
    end
end

@inline function LinearAlgebra.bunchkaufman!(
    A::SMatrixOrHermSym, rook::Bool = false; kws...
)
    bunchkaufman(A, rook; kws...)
end
