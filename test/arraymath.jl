@testset "Array math" begin
    @testset "AbstractArray-of-StaticArray with scalar math" begin
        v = SVector{2,Float64}[SVector{2,Float64}(1,1)]
        @test v .* 1.0 == v
        @test typeof(v .* 1.0) == typeof(v)
        @test 1 .- v == v .- v
        @test typeof(1 .- v) == typeof(v)
        v2 = SVector{2,Int}[SVector{2,Int}(1,1)]
        @test v2 .* 1.0 == v
        @test typeof(v2 .* 1.0) == typeof(v)
    end

    @testset "Array-scalar math" begin
        m = @SMatrix [1 2; 3 4]

        @test m .+ 1 === @SMatrix [2 3; 4 5]
        @test 1 .+ m === @SMatrix [2 3; 4 5]
        @test m .* 2 === @SMatrix [2 4; 6 8]
        @test 2 .* m === @SMatrix [2 4; 6 8]
        @test m .- 1 === @SMatrix [0 1; 2 3]
        @test 1 .- m === @SMatrix [0 -1; -2 -3]
        @test m ./ 2 === @SMatrix [0.5 1.0; 1.5 2.0]
        @test 12 ./ m === @SMatrix [12.0 6.0; 4.0 3.0]

    end

    @testset "Elementwise array math" begin
        m1 = @SMatrix [1 2; 3 4]
        m2 = @SMatrix [4 3; 2 1]

        @test .-(m1) === @SMatrix [-1 -2; -3 -4]

        @test m1 .+ m2 === @SMatrix [5 5; 5 5]
        @test m1 .* m2 === @SMatrix [4 6; 6 4]
        @test m1 .- m2 === @SMatrix [-3 -1; 1 3]
        @test m1 ./ m2 === @SMatrix [0.25 2/3; 1.5 4.0]
    end

    @testset "zeros() and ones()" begin
        @test zeros(SVector{3,Float64}) === @SVector [0.0, 0.0, 0.0]
        @test zeros(SVector{3,Int}) === @SVector [0, 0, 0]
        @test ones(SVector{3,Float64}) === @SVector [1.0, 1.0, 1.0]
        @test ones(SVector{3,Int}) === @SVector [1, 1, 1]

        @test zeros(SVector{3}) === @SVector [0.0, 0.0, 0.0]
        @test zeros(SMatrix{2,2}) === @SMatrix [0.0 0.0; 0.0 0.0]
        @test zeros(SArray{(1,1,1)}) === SArray{(1,1,1)}((0.0,))
        @test zeros(MVector{3})::MVector == @MVector [0.0, 0.0, 0.0]
        @test zeros(MMatrix{2,2})::MMatrix == @MMatrix [0.0 0.0; 0.0 0.0]
        @test zeros(MArray{(1,1,1)})::MArray == MArray{(1,1,1)}((0.0,))

        @test ones(SVector{3}) === @SVector [1.0, 1.0, 1.0]
        @test ones(SMatrix{2,2}) === @SMatrix [1.0 1.0; 1.0 1.0]
        @test ones(SArray{(1,1,1)}) === SArray{(1,1,1)}((1.0,))
        @test ones(MVector{3})::MVector == @MVector [1.0, 1.0, 1.0]
        @test ones(MMatrix{2,2})::MMatrix == @MMatrix [1.0 1.0; 1.0 1.0]
        @test ones(MArray{(1,1,1)})::MArray == MArray{(1,1,1)}((1.0,))
    end

    @testset "zero()" begin
        @test zero(SVector{3, Float64}) === @SVector [0.0, 0.0, 0.0]
        @test zero(SVector{3, Int}) === @SVector [0, 0, 0]
    end

    @testset "fill()" begin
        @test all(fill(3., SMatrix{4, 16, Float64}) .== 3.)
        @test @allocated(fill(0., SMatrix{1, 16, Float64})) == 0 # #81
    end
end
