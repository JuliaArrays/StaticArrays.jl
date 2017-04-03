@testset "Map, reduce, mapreduce, broadcast" begin
    @testset "map and map!" begin
        v1 = @SVector [2,4,6,8]
        v2 = @SVector [4,3,2,1]
        mv = MVector{4, Int}()

        normal_v1 = [2,4,6,8]
        normal_v2 = [4,3,2,1]

        @test @inferred(map(-, v1)) === @SVector [-2, -4, -6, -8]
        @test @inferred(map(+, v1, v2)) === @SVector [6, 7, 8, 9]
        @test @inferred(map(+, normal_v1, v2)) === @SVector [6, 7, 8, 9]
        @test @inferred(map(+, v1, normal_v2)) === @SVector [6, 7, 8, 9]

        map!(+, mv, v1, v2)
        @test mv == @MVector [6, 7, 8, 9]
    end

    @testset "reduce" begin
        v1 = @SVector [2,4,6,8]
        @test reduce(+, v1) === 20
        @test reduce(+, 0, v1) === 20
        @test sum(v1) === 20
        @test prod(v1) === 384
    end

    @testset "reduce in dim" begin
        a = @SArray rand(4,3,2)
        @test maximum(a, Val{1}) == maximum(a, 1)
        @test maximum(a, Val{2}) == maximum(a, 2)
        @test maximum(a, Val{3}) == maximum(a, 3)
        @test minimum(a, Val{1}) == minimum(a, 1)
        @test minimum(a, Val{2}) == minimum(a, 2)
        @test minimum(a, Val{3}) == minimum(a, 3)
        @test diff(a) == diff(a, Val{1}) == a[2:end,:,:] - a[1:end-1,:,:]
        @test diff(a, Val{2}) == a[:,2:end,:] - a[:,1:end-1,:]
        @test diff(a, Val{3}) == a[:,:,2:end] - a[:,:,1:end-1]

        a = @SArray rand(4,3)  # as of Julia v0.5, diff() for regular Array is defined only for vectors and matrices
        @test diff(a) == diff(a, Val{1}) == diff(a, 1)
        @test diff(a, Val{2}) == diff(a, 2)
    end

    @testset "mapreduce" begin
        v1 = @SVector [2,4,6,8]
        v2 = @SVector [4,3,2,1]
        @test mapreduce(-, +, v1) === -20
        @test mapreduce(-, +, 0, v1) === -20
        @test mapreduce(*, +, v1, v2) === 40
        @test mapreduce(*, +, 0, v1, v2) === 40
    end

    @testset "broadcast and broadcast!" begin
        c = 2
        v1 = @SVector [2,4,6,8]
        v2 = @SVector [4,3,2,1]
        mv = MVector{4, Int}()
        mm = MMatrix{4, 2, Int}()
        M = @SMatrix [1 2; 3 4; 5 6; 7 8]

        @test @inferred(broadcast(-, v1)) === map(-, v1)

        @test @inferred(broadcast(+, v1, c)) === @SVector [4, 6, 8, 10]
        @test @inferred(broadcast(+, v1, v2)) === map(+, v1, v2)
        @test @inferred(broadcast(+, v1, M)) === @SMatrix [3 4; 7 8; 11 12; 15 16]

        broadcast!(-, mv, v1)
        @test mv == @MVector [-2, -4, -6, -8]

        broadcast!(+, mv, v1, v2)
        @test mv == @MVector [6, 7, 8, 9]

        broadcast!(+, mm, v1, M)
        @test mm == @MMatrix [3 4; 7 8; 11 12; 15 16]
        # issue #103
        @test map(+, M, M) == [2 4; 6 8; 10 12; 14 16]
        
        @test ((@SVector Int64[]) + (@SVector Int64[])) == (@SVector Int64[])
    end
end
