@testset "Array math" begin
    @testset "zeros() and ones()" begin
        @test @inferred(zeros(SVector{3,Float64})) === @SVector [0.0, 0.0, 0.0]
        @test @inferred(zeros(SVector{3,Int})) === @SVector [0, 0, 0]
        @test @inferred(ones(SVector{3,Float64})) === @SVector [1.0, 1.0, 1.0]
        @test @inferred(ones(SVector{3,Int})) === @SVector [1, 1, 1]

        @test @inferred(zeros(SVector{3})) === @SVector [0.0, 0.0, 0.0]
        @test @inferred(zeros(SMatrix{2,2})) === @SMatrix [0.0 0.0; 0.0 0.0]
        @test @inferred(zeros(SArray{Tuple{1,1,1}})) === SArray{Tuple{1,1,1}}((0.0,))
        @test @inferred(zeros(MVector{3}))::MVector == @MVector [0.0, 0.0, 0.0]
        @test @inferred(zeros(MMatrix{2,2}))::MMatrix == @MMatrix [0.0 0.0; 0.0 0.0]
        @test @inferred(zeros(MArray{Tuple{1,1,1}}))::MArray == MArray{Tuple{1,1,1}}((0.0,))

        @test @inferred(ones(SVector{3})) === @SVector [1.0, 1.0, 1.0]
        @test @inferred(ones(SMatrix{2,2})) === @SMatrix [1.0 1.0; 1.0 1.0]
        @test @inferred(ones(SArray{Tuple{1,1,1}})) === SArray{Tuple{1,1,1}}((1.0,))
        @test @inferred(ones(MVector{3}))::MVector == @MVector [1.0, 1.0, 1.0]
        @test @inferred(ones(MMatrix{2,2}))::MMatrix == @MMatrix [1.0 1.0; 1.0 1.0]
        @test @inferred(ones(MArray{Tuple{1,1,1}}))::MArray == MArray{Tuple{1,1,1}}((1.0,))
    end

    @testset "zero()" begin
        @test @inferred(zero(SVector{3, Float64})) === @SVector [0.0, 0.0, 0.0]
        @test @inferred(zero(SVector{3, Int})) === @SVector [0, 0, 0]
    end

    @testset "fill()" begin
        @test all(@inferred(fill(3., SMatrix{4, 16, Float64})) .== 3.)
        @test @allocated(fill(0., SMatrix{1, 16, Float64})) == 0 # #81
    end

    @testset "fill!()" begin
        m = MMatrix{4,16,Float64}()
        fill!(m, 3)
        @test all(m .== 3.)
    end
end
