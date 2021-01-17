using StaticArrays, Test

@testset "sort" begin

    @testset "basics" for T in (Int, Float64)
        for N in (0, 1, 2, 3, 10, 20, 30)
            vs = rand(SVector{N,T})
            vm = MVector{N,T}(vs)
            vref = sort(Vector(vs))

            @test @inferred(sort(vs)) isa SVector
            @test @inferred(sort(vs, alg=QuickSort)) isa SVector
            @test @inferred(sort(vm)) isa MVector
            @test vref == sort(vs)
            @test vref == sort(vm)

            # @allocated seems broken since 1.4
            #N <= 20 && @test 0 == @allocated sort(vs)
        end
    end

    @testset "sortperm" for T in (Int, Float64)
        for N in (0, 1, 2, 3, 10, 20, 30)
            v = rand(SVector{N,T})
            vp = sortperm(v)
            @test v[vp] == sort(v)
        end

        # stability
        @test sortperm(SA[1, 1, 1, 0]) == SA[4, 1, 2, 3]
    end

end
