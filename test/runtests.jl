using StaticArrays
using Base.Test

@testset "StaticArrays" begin
    include("SVector.jl")
    include("MVector.jl")
    include("SMatrix.jl")
    include("MMatrix.jl")
    include("SArray.jl")
    include("MArray.jl")
    include("FieldVector.jl")
    include("Scalar.jl")

    include("core.jl")
    include("abstractarray.jl")
    include("indexing.jl")
    include("mapreduce.jl")
    include("arraymath.jl")
    include("linalg.jl")
    include("matrix_multiply.jl")
    include("det.jl")
    include("inv.jl")
    include("solve.jl")
    include("eigen.jl")
    include("deque.jl")
    include("fixed_size_arrays.jl")
end
