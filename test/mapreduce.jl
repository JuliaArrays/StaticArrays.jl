@testset "Map, reduce, mapreduce, broadcast" begin
    @testset "map and map!" begin
        v1 = @SVector [2,4,6,8]
        v2 = @SVector [4,3,2,1]
        mv = MVector{4, Int}()

        @test map(-, v1) === @SVector [-2, -4, -6, -8]
        @test map(+, v1, v2) === @SVector [6, 7, 8, 9]

        map!(+, mv, v1, v2)
        @test mv == @MVector [6, 7, 8, 9]
    end

    @testset "reduce" begin
        v1 = @SVector [2,4,6,8]
        @test reduce(+, v1) === 20
        @test reduce(+, 0, v1) === 20
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
        M = @SMatrix [1 2; 3 4; 5 6; 7 8]

        @test broadcast(-, v1) === map(-, v1)

        @test broadcast(+, v1, c) === @SVector [4, 6, 8, 10]
        @test broadcast(+, v1, v2) === map(+, v1, v2)
        @test broadcast(+, v1, M) === @SMatrix [3 4; 7 8; 11 12; 15 16]

        broadcast!(-, mv, v1)
        @test mv == @MVector [-2, -4, -6, -8]

        broadcast!(+, mv, v1, v2)
        @test mv == @MVector [6, 7, 8, 9]
    end
end
