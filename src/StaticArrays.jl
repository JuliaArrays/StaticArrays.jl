module StaticArrays

import Base: @_propagate_inbounds_meta, @propagate_inbounds, @pure

import Base: getindex, setindex!, size, similar, vec, show, length, convert, promote_op,
             promote_rule, map, map!, reduce, mapreduce, foldl, mapfoldl, broadcast,
             broadcast!, conj, hcat, vcat, ones, zeros, one, reshape, fill, fill!, inv,
             iszero, sum, prod, count, any, all, minimum, maximum, extrema,
             copy, read, read!, write, reverse

using Random
import Random: rand, randn, randexp, rand!, randn!, randexp!
using Core.Compiler: return_type
import Base: sqrt, exp, log, float, real
using LinearAlgebra
import LinearAlgebra: transpose, adjoint, dot, eigvals, eigen, lyap, tr,
                      kron, diag, norm, dot, diagm, lu, svd, svdvals, pinv,
                      factorize, ishermitian, issymmetric, isposdef, issuccess, normalize,
                      normalize!, Eigen, det, logdet, logabsdet, cross, diff, qr, \,
                      triu, tril
using LinearAlgebra: checksquare

using PrecompileTools

# StaticArraysCore imports
# there is intentionally no "using StaticArraysCore" to not take all symbols exported
# from StaticArraysCore to make transitioning definitions to StaticArraysCore easier.
using StaticArraysCore: StaticArraysCore, StaticArray, StaticScalar, StaticVector,
                        StaticMatrix, StaticVecOrMat, tuple_length, tuple_prod,
                        tuple_minimum, size_to_tuple
using StaticArraysCore: FieldArray, FieldMatrix, FieldVector
using StaticArraysCore: StaticArrayStyle
using StaticArraysCore: Dynamic, StaticDimension
import StaticArraysCore: SArray, SVector, SMatrix
import StaticArraysCore: MArray, MVector, MMatrix
import StaticArraysCore: SizedArray, SizedVector, SizedMatrix
import StaticArraysCore: check_array_parameters, convert_ntuple
import StaticArraysCore: similar_type, Size

# end of StaticArraysCore imports
# StaticArraysCore exports
export StaticScalar, StaticArray, StaticVector, StaticMatrix
export Scalar, SArray, SVector, SMatrix
export MArray, MVector, MMatrix
export SizedArray, SizedVector, SizedMatrix
# end of StaticArraysCore exports

export SOneTo
export Scalar
export FieldVector, FieldMatrix, FieldArray
export SDiagonal
export SHermitianCompact

export Size, Length

export SA, SA_F32, SA_F64
export @SVector, @SMatrix, @SArray
export @MVector, @MMatrix, @MArray

export similar_type
export push, pop, pushfirst, popfirst, insert, deleteat, setindex
export enumerate_static

export StaticArraysCore

include("SOneTo.jl")

# Being a member of StaticMatrixLike, StaticVecOrMatLike, or StaticArrayLike implies that Size(A)
# returns a static Size instance (none of the dimensions are Dynamic). The converse may not be true.
# These are akin to aliases like StridedArray and in similarly bad taste, but the current approach
# in Base necessitates their existence.
const StaticMatrixLike{T} = Union{
    StaticMatrix{<:Any, <:Any, T},
    Transpose{T, <:StaticVecOrMat{T}},
    Adjoint{T, <:StaticVecOrMat{T}},
    Symmetric{T, <:StaticMatrix{<:Any, <:Any, T}},
    Hermitian{T, <:StaticMatrix{<:Any, <:Any, T}},
    Diagonal{T, <:StaticVector{<:Any, T}},
    # We specifically list *Triangular here rather than using
    # AbstractTriangular to avoid ambiguities in size() etc.
    UpperTriangular{T, <:StaticMatrix{<:Any, <:Any, T}},
    LowerTriangular{T, <:StaticMatrix{<:Any, <:Any, T}},
    UnitUpperTriangular{T, <:StaticMatrix{<:Any, <:Any, T}},
    UnitLowerTriangular{T, <:StaticMatrix{<:Any, <:Any, T}}
}
const StaticVecOrMatLike{T} = Union{StaticVector{<:Any, T}, StaticMatrixLike{T}}
const StaticArrayLike{T} = Union{StaticVecOrMatLike{T}, StaticArray{<:Tuple, T}}

const AbstractScalar{T} = AbstractArray{T, 0} # not exported, but useful none-the-less
const StaticArrayNoEltype{S, N, T} = StaticArray{S, T, N}

include("util.jl")
include("traits.jl")

include("SUnitRange.jl")
include("FieldArray.jl")
include("SArray.jl")
include("SMatrix.jl")
include("SVector.jl")
include("Scalar.jl")
include("MArray.jl")
include("MVector.jl")
include("MMatrix.jl")
include("SizedArray.jl")
include("SDiagonal.jl")
include("SHermitianCompact.jl")

include("initializers.jl")
include("convert.jl")

include("abstractarray.jl")
include("indexing.jl")
include("broadcast.jl")
include("mapreduce.jl")
include("sort.jl")
include("arraymath.jl")
include("linalg.jl")
include("matrix_multiply_add.jl")
include("matrix_multiply.jl")
include("lu.jl")
include("det.jl")
include("inv.jl")
include("solve.jl")
include("eigen.jl")
include("expm.jl")
include("sqrtm.jl")
include("lyap.jl")
include("triangular.jl")
include("cholesky.jl")
include("svd.jl")
include("qr.jl")
include("deque.jl")
include("flatten.jl")
include("io.jl")
include("pinv.jl")

@static if VERSION >= v"1.7"
    include("blas.jl")
end

@static if !isdefined(Base, :get_extension) # VERSION < v"1.9-"
    include("../ext/StaticArraysStatisticsExt.jl")
end

include("precompile.jl")

end # module
