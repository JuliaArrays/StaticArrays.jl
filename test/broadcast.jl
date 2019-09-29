using StaticArrays, Test
struct ScalarTest end
Base.:(+)(x::Number, y::ScalarTest) = x
Broadcast.broadcastable(x::ScalarTest) = Ref(x)

@testset "Scalar Broadcast" begin
    for t in (SVector{2}, MVector{2}, SMatrix{2, 2}, MMatrix{2, 2})
        x = rand(t)
        @test x == @inferred(x .+ ScalarTest())
        @test x .+ 1 == @inferred(x .+ Ref(1))
    end

    @test Scalar(3) == @inferred(Scalar(1) .+ 2)
end

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
        v2 = SVector{2,Int}[SVector{2,Int}(1,1)]
        @test @inferred(v2 .* 1.0)::typeof(v) == v
    end

    @testset "0-dimensional Array broadcast" begin
        x = Array{Int, 0}(undef)
        x .= Scalar(4)
        @test x[] == 4
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
        # Issue #546
        @test @inferred(m ./ (v .* v')) === @SMatrix [1.0 0.5; 0.75 0.25]
        testinf(m, v) = m ./ (v .* v')
        @test @inferred(testinf(m, v)) === @SMatrix [1.0 0.5; 0.75 0.25]
    end

    @testset "2x2 StaticMatrix with 1x2 StaticMatrix" begin
        # Issues #197, #242: broadcast between SArray and row-like SMatrix
        m1 = @SMatrix [1 2; 3 4]
        m2 = @SMatrix [1 4]
        @test @inferred(broadcast(+, m1, m2)) === @SMatrix [2 6; 4 8]
        @test @inferred(m1 .+ m2) === @SMatrix [2 6; 4 8]
        @test @inferred(m2 .+ m1) === @SMatrix [2 6; 4 8]
        @test @inferred(m1 .* m2) === @SMatrix [1 8; 3 16]
        @test @inferred(m2 .* m1) === @SMatrix [1 8; 3 16]
        @test @inferred(m1 ./ m2) === @SMatrix [1 1/2; 3 1]
        @test @inferred(m2 ./ m1) === @SMatrix [1 2; 1/3 1]
        @test @inferred(m1 .- m2) === @SMatrix [0 -2; 2 0]
        @test @inferred(m2 .- m1) === @SMatrix [0 2; -2 0]
        @test @inferred(m1 .^ m2) === @SMatrix [1 16; 3 256]
    end

    @testset "1x2 StaticMatrix with StaticVector" begin
        # Issues #197, #242: broadcast between SVector and row-like SMatrix
        m = @SMatrix [1 2]
        v = SVector(1, 4)
        @test @inferred(broadcast(+, m, v)) === @SMatrix [2 3; 5 6]
        @test @inferred(m .+ v) === @SMatrix [2 3; 5 6]
        @test @inferred(v .+ m) === @SMatrix [2 3; 5 6]
        @test @inferred(m .* v) === @SMatrix [1 2; 4 8]
        @test @inferred(v .* m) === @SMatrix [1 2; 4 8]
        @test @inferred(m ./ v) === @SMatrix [1 2; 1/4 1/2]
        @test @inferred(v ./ m) === @SMatrix [1 1/2; 4 2]
        @test @inferred(m .- v) === @SMatrix [0 1; -3 -2]
        @test @inferred(v .- m) === @SMatrix [0 -1; 3 2]
        @test @inferred(m .^ v) === @SMatrix [1 2; 1 16]
        @test @inferred(v .^ m) === @SMatrix [1 1; 4 16]
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
        # Issue #200: broadcast with Adjoint
        @test @inferred(v1 .+ v2') === @SMatrix [2 5; 3 6]
        @test @inferred(v1 .+ transpose(v2)) === @SMatrix [2 5; 3 6]
        # Issue 382: infinite recursion in Base.Broadcast.broadcast_indices with Adjoint
        @test @inferred(SVector(1,1)' .+ [1, 1]) == [2 2; 2 2]
        @test @inferred(transpose(SVector(1,1)) .+ [1, 1]) == [2 2; 2 2]
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
        # Issue #199: broadcast with empty SArray
        @test @inferred(SVector(1) .+ SVector{0,Int}()) === SVector{0,Int}()
        @test @inferred(SVector{0,Int}() .+ SVector(1.0)) === SVector{0,Float64}()
        # Issue #528
        @test @inferred(isapprox(SMatrix{3,0,Float64}(), SMatrix{3,0,Float64}()))
        @test @inferred(broadcast(length, SVector{0,String}())) === SVector{0,Int}()
        @test @inferred(broadcast(join, SVector{0,String}(), SVector{0,String}(), SVector{0,String}())) === SVector{0,String}()
    end

    @testset "Mutating broadcast!" begin
        # No setindex! error
        A = one(SMatrix{2, 2}); @test_throws ErrorException broadcast!(+, A, A, SVector(1, 4))
        A = one(MMatrix{2, 2}); @test @inferred(broadcast!(+, A, A, SVector(1, 4))) == @MMatrix [2 1; 4 5]
        A = one(MMatrix{2, 2}); @test @inferred(broadcast!(+, A, A, @SMatrix([1  4]))) == @MMatrix [2 4; 1 5]
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
            @test eltype(a .+ 2) == Number
            @test eltype(a .- 2) == Number
            @test eltype(a * 2) == Number
            @test eltype(a / 2) == Number
        end
        let a = SVector{3, Real}(2, 2.0, 4//2)
            @test eltype(a .+ 2) == Real
            @test eltype(a .- 2) == Real
            @test eltype(a * 2) == Real
            @test eltype(a / 2) == Real
        end
        let a = SVector{3, Real}(2, 2.0, 4//2)
            @test eltype(a .+ 2.0) == Float64
            @test eltype(a .- 2.0) == Float64
            @test eltype(a * 2.0) == Float64
            @test eltype(a / 2.0) == Float64
        end
        let a = broadcast(Float32, SVector(3, 4, 5))
            @test eltype(a) == Float32
        end
    end

    @testset "broadcast general scalars" begin
        # Issue #239 - broadcast with non-numeric element types
        @eval @enum Axis aX aY aZ
        @testinf (SVector(aX,aY,aZ) .== Ref(aX)) == SVector(true,false,false)
        mv = MVector(aX,aY,aZ)
        @testinf broadcast!(identity, mv, Ref(aX)) == MVector(aX,aX,aX)
        @test mv == SVector(aX,aX,aX)
    end

    @testset "broadcast! with Array destination" begin
        # Issue #385
        a = zeros(3, 3)
        b = @SMatrix [1 2 3; 4 5 6; 7 8 9]
        a .= b
        @test a == b

        c = SVector(1, 2, 3)
        a .= c
        @test a == [1 1 1; 2 2 2; 3 3 3]

        d = SVector(1, 2, 3, 4)
        @test_throws DimensionMismatch a .= d
    end

    @testset "issue #493" begin
        X = rand(SVector{3,SVector{2,Float64}})
        foo493(X) = normalize.(X)
        @test foo493(X) isa Core.Compiler.return_type(foo493, Tuple{typeof(X)})
    end

    @testset "broadcasting with tuples" begin
        # issue 485
        @test @inferred(SA[1,2,3] .+ (1,))               === SA{Int}[2, 3, 4]
        @test @inferred(SA[1,2,3] .+ (10, 20, 30))       === SA{Int}[11, 22, 33]
        @test @inferred((1,2)     .+ (SA[10 20; 30 40])) === SA{Int}[11 21; 32 42]
        @test @inferred((SA[10 20; 30 40]) .+ (1,2))     === SA{Int}[11 21; 32 42]

        add_bc!(m, v) = m .+= v  # Helper function; @inferred gets confused by .+= syntax
        @test @inferred(add_bc!(MVector((1,2,3)), (10, 20, 30)))   ::MVector{3,Int}   == SA[11, 22, 33]
        @test @inferred(add_bc!(MMatrix(SA[10 20; 30 40]), (1,2))) ::MMatrix{2,2,Int} == SA[11 21; 32 42]

        # Tuples of SA
        @test SA[1,2,3] .* (SA[1,0],) === SVector{3,SVector{2,Int}}(((1,0), (2,0), (3,0)))
        # Unfortunately this case of nested broadcasting is not inferred
        @test_broken @inferred(SA[1,2,3] .* (SA[1,0],))
    end
end
