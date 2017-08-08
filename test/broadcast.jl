using StaticArrays, Base.Test

include("testutil.jl")

@testset "Broadcast sizes" begin
    @test @inferred(StaticArrays.broadcast_sizes(1, 1, 1)) === (Size(), Size(), Size())
    for t in (SVector{2}, MVector{2}, SMatrix{2, 2}, MMatrix{2, 2})
        @test @inferred(StaticArrays.broadcast_sizes(ones(t), ones(t), ones(t))) === (Size(t), Size(t), Size(t))
        @test @inferred(StaticArrays.broadcast_sizes(ones(t), 1, ones(t))) === (Size(t), Size(), Size(t))
        @test @inferred(StaticArrays.broadcast_sizes(1, ones(t), ones(t))) === (Size(), Size(t), Size(t))
        @test @inferred(StaticArrays.broadcast_sizes(ones(t), ones(t), 1)) === (Size(t), Size(t), Size())
        @test @inferred(StaticArrays.broadcast_sizes(1, ones(t), 1)) === (Size(), Size(t), Size())
        @test @inferred(StaticArrays.broadcast_sizes(ones(t), 1, 1)) === (Size(t), Size(), Size())
        @test @inferred(StaticArrays.broadcast_sizes(1, 1, ones(t))) === (Size(), Size(), Size(t))
    end
    # test case issue #191
    @test @inferred(broadcast((a, b, c) -> 0, SVector(1, 1), 0, 0)) == SVector(0, 0)
end

@testset "Broadcast" begin
    @testset "AbstractArray-of-StaticArray with scalar math" begin
        v = SVector{2,Float64}[SVector{2,Float64}(1,1)]
        @test @inferred(v .* 1.0)::typeof(v) == v
        @test @inferred(1 .- v)::typeof(v) == v .- v
        v2 = SVector{2,Int}[SVector{2,Int}(1,1)]
        @test @inferred(v2 .* 1.0)::typeof(v) == v
    end

    @testset "2x2 StaticMatrix with StaticVector" begin
        m = @SMatrix [1 2; 3 4]
        v = SVector(1, 4)
        @test @inferred(broadcast(+, m, v)) === @SMatrix [2 3; 7 8]
        @test @inferred(m .+ v) === @SMatrix [2 3; 7 8]
        @test @inferred(v .+ m) === @SMatrix [2 3; 7 8]
        @test @inferred(m .* v) === @SMatrix [1 2; 12 16]
        @test @inferred(v .* m) === @SMatrix [1 2; 12 16]
        @test @inferred(m ./ v) === @SMatrix [1 2; 3/4 1]
        @test @inferred(v ./ m) === @SMatrix [1 1/2; 4/3 1]
        @test @inferred(m .- v) === @SMatrix [0 1; -1 0]
        @test @inferred(v .- m) === @SMatrix [0 -1; 1 0]
        @test @inferred(m .^ v) === @SMatrix [1 2; 81 256]
        @test @inferred(v .^ m) === @SMatrix [1 1; 64 256]
    end

    @testset "2x2 StaticMatrix with 1x2 StaticMatrix" begin
        m1 = @SMatrix [1 2; 3 4]
        m2 = @SMatrix [1 4]
        @test @inferred(broadcast(+, m1, m2)) === @SMatrix [2 6; 4 8] #197
        @test @inferred(m1 .+ m2) === @SMatrix [2 6; 4 8] #197
        @test @inferred(m2 .+ m1) === @SMatrix [2 6; 4 8]
        @test @inferred(m1 .* m2) === @SMatrix [1 8; 3 16] #197
        @test @inferred(m2 .* m1) === @SMatrix [1 8; 3 16]
        @test @inferred(m1 ./ m2) === @SMatrix [1 1/2; 3 1] #197
        @test @inferred(m2 ./ m1) === @SMatrix [1 2; 1/3 1]
        @test @inferred(m1 .- m2) === @SMatrix [0 -2; 2 0] #197
        @test @inferred(m2 .- m1) === @SMatrix [0 2; -2 0]
        @test @inferred(m1 .^ m2) === @SMatrix [1 16; 3 256] #197
    end

    @testset "1x2 StaticMatrix with StaticVector" begin
        m = @SMatrix [1 2]
        v = SVector(1, 4)
        @test @inferred(broadcast(+, m, v)) === @SMatrix [2 3; 5 6]
        @test @inferred(m .+ v) === @SMatrix [2 3; 5 6]
        @test @inferred(v .+ m) === @SMatrix [2 3; 5 6] #197
        @test @inferred(m .* v) === @SMatrix [1 2; 4 8]
        @test @inferred(v .* m) === @SMatrix [1 2; 4 8] #197
        @test @inferred(m ./ v) === @SMatrix [1 2; 1/4 1/2]
        @test @inferred(v ./ m) === @SMatrix [1 1/2; 4 2] #197
        @test @inferred(m .- v) === @SMatrix [0 1; -3 -2]
        @test @inferred(v .- m) === @SMatrix [0 -1; 3 2] #197
        @test @inferred(m .^ v) === @SMatrix [1 2; 1 16]
        @test @inferred(v .^ m) === @SMatrix [1 1; 4 16] #197
    end

    @testset "StaticVector with StaticVector" begin
        v1 = SVector(1, 2)
        v2 = SVector(1, 4)
        @test @inferred(broadcast(+, v1, v2)) === SVector(2, 6)
        @test @inferred(v1 .+ v2) === SVector(2, 6)
        @test @inferred(v2 .+ v1) === SVector(2, 6)
        @test @inferred(v1 .* v2) === SVector(1, 8)
        @test @inferred(v2 .* v1) === SVector(1, 8)
        @test @inferred(v1 ./ v2) === SVector(1, 1/2)
        @test @inferred(v2 ./ v1) === SVector(1, 2/1)
        @test @inferred(v1 .- v2) === SVector(0, -2)
        @test @inferred(v2 .- v1) === SVector(0, 2)
        @test @inferred(v1 .^ v2) === SVector(1, 16)
        @test @inferred(v2 .^ v1) === SVector(1, 16)
        # test case issue #199
        @test @inferred(SVector(1) .+ SVector()) === SVector()
        @test @inferred(SVector() .+ SVector(1)) === SVector()
        # test case issue #200
        @test @inferred(v1 .+ v2') === @SMatrix [2 5; 3 6]
    end

    @testset "StaticVector with Scalar" begin
        v = SVector(1, 2)
        @test @inferred(broadcast(+, v, 2)) === SVector(3, 4)
        @test @inferred(v .+ 2) === SVector(3, 4)
        @test @inferred(2 .+ v) === SVector(3, 4)
        @test @inferred(v .* 2) === SVector(2, 4)
        @test @inferred(2 .* v) === SVector(2, 4)
        @test @inferred(v ./ 2) === SVector(1/2, 1)
        @test @inferred(2 ./ v) === SVector(2, 1/1)
        @test @inferred(v .- 2) === SVector(-1, 0)
        @test @inferred(2 .- v) === SVector(1, 0)
        @test @inferred(v .^ 2) === SVector(1, 4)
        @test @inferred(2 .^ v) === SVector(2, 4)
    end

    @testset "Empty arrays" begin
        @test @inferred(1.0 .+ zeros(SMatrix{2,0})) === zeros(SMatrix{2,0})
        @test @inferred(1.0 .+ zeros(SMatrix{0,2})) === zeros(SMatrix{0,2})
        @test @inferred(1.0 .+ zeros(SArray{Tuple{2,3,0}})) === zeros(SArray{Tuple{2,3,0}})
        @test @inferred(zeros(SVector{0}) .+ zeros(SMatrix{0,2})) === zeros(SMatrix{0,2})
        m = zeros(MMatrix{0,2})
        @test @inferred(broadcast!(+, m, m, zeros(SVector{0}))) == zeros(SMatrix{0,2})
    end

    @testset "Mutating broadcast!" begin
        # No setindex! error
        A = eye(SMatrix{2, 2}); @test_throws ErrorException broadcast!(+, A, A, SVector(1, 4))
        A = eye(MMatrix{2, 2}); @test @inferred(broadcast!(+, A, A, SVector(1, 4))) == @MMatrix [2 1; 4 5]
        A = eye(MMatrix{2, 2}); @test @inferred(broadcast!(+, A, A, @SMatrix([1  4]))) == @MMatrix [2 4; 1 5]
        A = @MMatrix([1 0]); @test_throws DimensionMismatch broadcast!(+, A, A, SVector(1, 4))
        A = @MMatrix([1 0]); @test @inferred(broadcast!(+, A, A, @SMatrix([1 4]))) == @MMatrix [2 4]
        A = @MMatrix([1 0]); @test @inferred(broadcast!(+, A, A, 2)) == @MMatrix [3 2]
    end

    @testset "broadcast! with mixtures of SArray and Array" begin
        a = zeros(MVector{2}); @test @inferred(broadcast!(+, a, [1,2])) == [1,2]
        a = zeros(MMatrix{2,3}); @test @inferred(broadcast!(+, a, [1,2])) == [1 1 1; 2 2 2]
    end

    @testset "eltype after broadcast" begin
        # test cases issue #198
        let a = SVector{4, Number}(2, 2.0, 4//2, 2+0im)
            @test eltype(a + 2) == Number
            @test eltype(a - 2) == Number
            @test eltype(a * 2) == Number
            @test eltype(a / 2) == Number
        end
        let a = SVector{3, Real}(2, 2.0, 4//2)
            @test eltype(a + 2) == Real
            @test eltype(a - 2) == Real
            @test eltype(a * 2) == Real
            @test eltype(a / 2) == Real
        end
        let a = SVector{3, Real}(2, 2.0, 4//2)
            @test eltype(a + 2.0) == Real
            @test eltype(a - 2.0) == Real
            @test eltype(a * 2.0) == Real
            @test eltype(a / 2.0) == Real
        end
        let a = broadcast(Float32, SVector(3, 4, 5))
            @test eltype(a) == Float32
        end
    end

    @testset "broadcast general scalars" begin
        # Issue #239 - broadcast with non-numeric element types
        @eval @enum Axis aX aY aZ
        @testinf (SVector(aX,aY,aZ) .== aX) == SVector(true,false,false)
        mv = MVector(aX,aY,aZ)
        @testinf broadcast!(identity, mv, aX) == MVector(aX,aX,aX)
        @test mv == SVector(aX,aX,aX)
    end
end
