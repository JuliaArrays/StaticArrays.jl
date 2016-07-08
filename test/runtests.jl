using StaticArrays
using Base.Test

#=
# @inferred throws an error, which doesn't interract particularly well
# with BaseTestNext
macro test_inferred(ex)
    quote
        @test try
            @inferred($ex)
            true
        catch
            false
        end
    end
end

@testset "StaticArrays" begin
    include("core.jl")
    include("abstractarray.jl")
    include("indexing.jl")
    include("linalg.jl")
end
=#
