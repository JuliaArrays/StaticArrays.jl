@testset "Linear algebra" begin

    @testset "SVector as a (mathematical) vector space" begin
        c = 2
        v1 = @SVector [2,4,6,8]
        v2 = @SVector [4,3,2,1]

        @test v1 + c === @SVector [4,6,8,10]
        @test v1 - c === @SVector [0,2,4,6]
        @test v1 * c === @SVector [4,8,12,16]
        @test v1 / c === @SVector [1.0,2.0,3.0,4.0]

        @test v1 + v2 === @SVector [6, 7, 8, 9]
        @test v1 - v2 === @SVector [-2, 1, 4, 7]
    end

    @testset "diagm()" begin
        @test diagm(SVector(1,2)) === @SMatrix [1 0; 0 2]
    end

    @testset "one()" begin
        @test one(SMatrix{2,2,Int}) === @SMatrix [1 0; 0 1]
        @test one(SMatrix{2,2}) === @SMatrix [1.0 0.0; 0.0 1.0]
        @test one(SMatrix{2}) === @SMatrix [1.0 0.0; 0.0 1.0]

        @test one(MMatrix{2,2,Int})::MMatrix == @MMatrix [1 0; 0 1]
        @test one(MMatrix{2,2})::MMatrix == @MMatrix [1.0 0.0; 0.0 1.0]
        @test one(MMatrix{2})::MMatrix == @MMatrix [1.0 0.0; 0.0 1.0]
    end

    @testset "eye()" begin
        @test eye(SMatrix{2,2,Int}) === @SMatrix [1 0; 0 1]
        @test eye(SMatrix{2,2}) === @SMatrix [1.0 0.0; 0.0 1.0]
        @test eye(SMatrix{2}) === @SMatrix [1.0 0.0; 0.0 1.0]

        @test eye(SMatrix{2,3,Int}) === @SMatrix [1 0 0; 0 1 0]
        @test eye(SMatrix{2,3}) === @SMatrix [1.0 0.0 0.0; 0.0 1.0 0.0]

        @test eye(MMatrix{2,2,Int})::MMatrix == @MMatrix [1 0; 0 1]
        @test eye(MMatrix{2,2})::MMatrix == @MMatrix [1.0 0.0; 0.0 1.0]
        @test eye(MMatrix{2})::MMatrix == @MMatrix [1.0 0.0; 0.0 1.0]
    end

    @testset "cross()" begin
        @test cross(SVector(1,2,3), SVector(4,5,6)) === SVector(-3, 6, -3)
    end

    @testset "transpose() and conj()" begin
        @test conj(SVector(1+im, 2+im)) === SVector(1-im, 2-im)

        @test @SVector([1, 2, 3]).' === @SMatrix([1 2 3])
        @test @SMatrix([1 2; 0 3]).' === @SMatrix([1 0; 2 3])
        @test @SMatrix([1 2 3; 4 5 6]).' === @SMatrix([1 4; 2 5; 3 6])

        @test @SVector([1, 2, 3])' === @SMatrix([1 2 3])
        @test @SMatrix([1 2; 0 3])' === @SMatrix([1 0; 2 3])
        @test @SMatrix([1 2 3; 4 5 6])' === @SMatrix([1 4; 2 5; 3 6])
    end

    @testset "vcat() and hcat()" begin
        @test vcat(SVector(1,2,3), SVector(4,5,6)) === SVector(1,2,3,4,5,6)
        @test hcat(SVector(1,2,3), SVector(4,5,6)) === @SMatrix [1 4; 2 5; 3 6]

        @test vcat(@SMatrix([1;2;3]), SVector(4,5,6)) === @SMatrix([1;2;3;4;5;6])
        @test vcat(SVector(1,2,3), @SMatrix([4;5;6])) === @SMatrix([1;2;3;4;5;6])
        @test hcat(@SMatrix([1;2;3]), SVector(4,5,6)) === @SMatrix [1 4; 2 5; 3 6]
        @test hcat(SVector(1,2,3), @SMatrix([4;5;6])) === @SMatrix [1 4; 2 5; 3 6]

        @test vcat(@SMatrix([1;2;3]), @SMatrix([4;5;6])) === @SMatrix([1;2;3;4;5;6])
        @test hcat(@SMatrix([1;2;3]), @SMatrix([4;5;6])) === @SMatrix [1 4; 2 5; 3 6]
    end

    @testset "normalization" begin
        @test norm(SVector(1.0,2.0,2.0)) ≈ 3.0
        @test norm(SVector(1.0,2.0,2.0),2) ≈ 3.0
        @test norm(SVector(1.0,2.0,2.0),Inf) ≈ 2.0
        @test norm(SVector(1.0,2.0,2.0),1) ≈ 5.0
        @test norm(SVector(1.0,2.0,0.0),0) ≈ 2.0

        @test vecnorm(SVector(1.0,2.0)) ≈ vecnorm([1.0,2.0])
        @test vecnorm(@SMatrix [1 2; 3 4.0+im]) ≈ vecnorm([1 2; 3 4.0+im])

        @test normalize(SVector(1,2,3)) ≈ normalize([1,2,3])
    end
end
