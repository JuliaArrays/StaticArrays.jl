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


    include("core.jl")
    include("abstractarray.jl")
    include("indexing.jl")
    include("mapreduce.jl")
    include("arraymath.jl")
    include("linalg.jl")
    include("matrix_multiply.jl")
    include("deque.jl")
end
