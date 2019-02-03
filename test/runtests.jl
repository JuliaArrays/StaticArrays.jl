using StaticArrays, Test, Random, LinearAlgebra, SpecialFunctions
using InteractiveUtils

# Allow no new ambiguities (see #18), unless you fix some old ones first!
if VERSION >= v"1.0.0"
    @test length(detect_ambiguities(Base, LinearAlgebra, StaticArrays)) <= 7
end

# We generate a lot of matrices using rand(), but unit tests should be
# deterministic. Therefore seed the RNG here (and further down, to avoid test
# file order dependence)
Random.seed!(42)

include("testutil.jl")
include("SVector.jl")
include("MVector.jl")
include("SMatrix.jl")
include("MMatrix.jl")
include("SArray.jl")
include("MArray.jl")
include("FieldVector.jl")
include("Scalar.jl")
include("SUnitRange.jl")
include("SizedArray.jl")
include("SDiagonal.jl")

include("custom_types.jl")
include("convert.jl")
include("core.jl")
include("abstractarray.jl")
include("indexing.jl")
Random.seed!(42); include("mapreduce.jl")
Random.seed!(42); include("arraymath.jl")
include("broadcast.jl")
include("linalg.jl")
Random.seed!(42); include("matrix_multiply.jl")
Random.seed!(42); include("triangular.jl")
include("det.jl")
include("inv.jl")
Random.seed!(42); include("solve.jl")
Random.seed!(44); include("eigen.jl")
include("expm.jl")
include("sqrtm.jl")
include("lyap.jl")
include("lu.jl")
Random.seed!(42); include("qr.jl")
Random.seed!(42); include("chol.jl") # hermitian_type(::Type{Any}) for block algorithm
include("deque.jl")
include("flatten.jl")
include("io.jl")
include("svd.jl")
Random.seed!(42); include("fixed_size_arrays.jl")
