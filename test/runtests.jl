using StaticArrays
if VERSION >= v"0.5-"
    using Base.Test
else
    using BaseTestNext
    const Test = BaseTestNext
end

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
