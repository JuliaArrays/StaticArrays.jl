@testset "SizedArray" begin
    @testset "Inner Constructors" begin
        @test SizedArray{Tuple{2},Int,1,1,Vector{Int}}((3, 4)).data == [3, 4]
        @test SizedArray{Tuple{2},Int,1,1,Vector{Int}}([3, 4]).data == [3, 4]
        @test SizedArray{Tuple{2,2},Int,2,1,Vector{Int}}([3, 4, 5, 6]).data == [3, 4, 5, 6]
        @test SizedArray{Tuple{},Int,0,1,Vector{Int}}((2,)).data == [2]
        @test_throws DimensionMismatch SizedArray{Tuple{2,2,1},Int,3,2,Matrix{Int}}([1 2; 3 4]).data == [1 2; 3 4]

        # Bad input
        @test_throws Exception SArray{Tuple{1},Int,1}([2 3])

        # Bad parameters
        @test_throws Exception SizedArray{Tuple{1},Int,2}(undef)
        @test_throws Exception SArray{Tuple{3, 4},Int,1}(undef)

        # Parameter/input size mismatch
        @test_throws Exception SizedArray{Tuple{1},Int,2}([2; 3])
        @test_throws Exception SizedArray{Tuple{1},Int,2}((2, 3))
    end

    @testset "Outer Constructors" begin
        @test SizedArray{Tuple{2},Int,1}([3, 4]).data == [3, 4]
        @test SizedArray{Tuple{2},Int,1,1}([3, 4]).data == [3, 4]

        # From Array
        @test @inferred(SizedArray{Tuple{2},Float64,1,1}([1,2]))::SizedArray{Tuple{2},Float64,1,1} == [1.0, 2.0]
        @test @inferred(SizedArray{Tuple{2},Float64,1}([1,2]))::SizedArray{Tuple{2},Float64,1,1} == [1.0, 2.0]
        @test @inferred(SizedArray{Tuple{2},Float64}([1,2]))::SizedArray{Tuple{2},Float64,1,1} == [1.0, 2.0]
        @test @inferred(SizedArray{Tuple{2}}([1,2]))::SizedArray{Tuple{2},Int,1,1} == [1,2]
        @test @inferred(SizedArray{Tuple{2,2}}([1 2;3 4]))::SizedArray{Tuple{2,2},Int,2,2} == [1 2; 3 4]
        # From Array, reshaped
        @test @inferred(SizedArray{Tuple{2,2}}([1,2,3,4]))::SizedArray{Tuple{2,2},Int,2,1} == [1 3; 2 4]
        @test_throws DimensionMismatch SizedArray{Tuple{4}}([1 2; 3 4])
        # Uninitialized
        @test @inferred(SizedArray{Tuple{2,2},Int,2,2,Matrix{Int}}(undef)) isa SizedArray{Tuple{2,2},Int,2,2,Matrix{Int}}
        @test @inferred(SizedArray{Tuple{2,2},Int,2,2}(undef)) isa SizedArray{Tuple{2,2},Int,2,2,Matrix{Int}}
        @test @inferred(SizedArray{Tuple{2,2},Int,2,1}(undef)) isa SizedArray{Tuple{2,2},Int,2,1,Vector{Int}}
        @test @inferred(SizedArray{Tuple{2,2},Int,2}(undef)) isa SizedArray{Tuple{2,2},Int,2,2,Matrix{Int}}
        @test @inferred(SizedArray{Tuple{2,2},Int}(undef)) isa SizedArray{Tuple{2,2},Int,2,2,Matrix{Int}}
        @test size(SizedArray{Tuple{4,5},Int,2}(undef).data) == (4, 5)
        @test size(SizedArray{Tuple{4,5},Int}(undef).data) == (4, 5)

        # 0-element constructor
        if VERSION >= v"1.1"
            @test (@inferred SizedArray(MMatrix{0,0,Float64}()))::SizedMatrix{0,0,Float64} == SizedMatrix{0,0,Float64}()
        end

        # From Tuple
        @test @inferred(SizedArray{Tuple{2},Float64,1,1}((1,2)))::SizedArray{Tuple{2},Float64,1,1,Vector{Float64}} == [1.0, 2.0]
        @test @inferred(SizedArray{Tuple{2},Float64,1}((1,2)))::SizedArray{Tuple{2},Float64,1,1,Vector{Float64}} == [1.0, 2.0]
        @test @inferred(SizedArray{Tuple{2},Float64}((1,2)))::SizedArray{Tuple{2},Float64,1,1,Vector{Float64}} == [1.0, 2.0]
        @test @inferred(SizedArray{Tuple{2}}((1,2)))::SizedArray{Tuple{2},Int,1,1,Vector{Int}} == [1,2]
        @test @inferred(SizedArray{Tuple{2,2}}((1,2,3,4)))::SizedArray{Tuple{2,2},Int,2,2,Matrix{Int}} == [1 3; 2 4]
        @test @inferred(SizedArray{Tuple{2,2},Int,2,1}((1,2,3,4)))::SizedArray{Tuple{2,2},Int,2,1,Vector{Int}} == [1 3; 2 4]
        @test SizedArray{Tuple{2},Int,1}((3, 4)).data == [3, 4]

        # Dimension 0
        @test SizedArray{Tuple{},Int,0,1}((2,)).data == [2]
        @test SizedArray{Tuple{},Int,0}((2,)).data == fill(2)
        @test SizedArray{Tuple{},Int}((2,)).data == fill(2)
    end

    @testset "SizedVector and SizedMatrix" begin
        @test @inferred(SizedVector{2}([1,2]))::SizedArray{Tuple{2},Int,1,1,Vector{Int}} == [1,2]
        @test @inferred(SizedVector{2}((1,2)))::SizedArray{Tuple{2},Int,1,1,Vector{Int}} == [1,2]
        @test @inferred(SizedVector{0,Int}(()))::SizedArray{Tuple{0},Int,1,1,Vector{Int}} == []
        # Reshaping
        @test @inferred(SizedVector{2}([1 2]))::SizedArray{Tuple{2},Int,1,1,Vector{Int}} == [1,2]
        # Back to Vector
        @test Vector(SizedVector{2}((1,2))) == [1,2]
        @test convert(Vector, SizedVector{2}((1,2))) == [1,2]

        # 0-element constructor
        @test (@inferred SizedVector(MVector{0,Float64}()))::SizedVector{0,Float64} == SizedVector{0,Float64}()

        @test @inferred(SizedMatrix{2,2}([1 2; 3 4]))::SizedArray{Tuple{2,2},Int,2,2,Matrix{Int}} == [1 2; 3 4]
        # Reshaping
        @test @inferred(SizedMatrix{2,2}([1,2,3,4]))::SizedArray{Tuple{2,2},Int,2,1} == [1 3; 2 4]
        @test @inferred(SizedMatrix{2,2}((1,2,3,4)))::SizedArray{Tuple{2,2},Int,2,2} == [1 3; 2 4]
        # Back to Matrix
        @test Matrix(SizedMatrix{2,2}([1 2;3 4])) == [1 2; 3 4]
        @test convert(Matrix, SizedMatrix{2,2}([1 2;3 4])) == [1 2; 3 4]

        # 0-element constructor
        if VERSION >= v"1.1"
            @test (@inferred SizedMatrix(MMatrix{0,0,Float64}()))::SizedMatrix{0,0,Float64} == SizedMatrix{0,0,Float64}()
        end
    end

    # setindex
    sa = SizedArray{Tuple{2}, Int, 1}([3, 4])
    sa[1] = 2
    @test sa.data == [2, 4]

    # parent
    @test parent(sa) === sa.data

    # pointer
    @test pointer(sa) === pointer(sa.data)

    @testset "vec" begin
        sa2 = SizedArray{Tuple{2, 2}, Int}([1, 2, 3, 4])
        @test (@inferred vec(sa2)) isa SizedVector{4, Int}
        @test vec(sa2).data === vec(sa2.data)
    end

    @testset "aliasing" begin
        a1 = rand(4)
        a2 = copy(a1)
        sa1 = SizedVector{4}(a1)
        sa2 = SizedVector{4}(a2)
        @test Base.mightalias(a1, sa1)
        @test Base.mightalias(sa1, SizedVector{4}(a1))
        @test !Base.mightalias(a2, sa1)
        @test !Base.mightalias(sa1, SizedVector{4}(a2))
        @test Base.mightalias(sa1, view(sa1, 1:2))
        @test Base.mightalias(a1, view(sa1, 1:2))
        @test Base.mightalias(sa1, view(a1, 1:2))
    end

    @testset "back to Array" begin
        @test Array(SizedArray{Tuple{2}, Int, 1}([3, 4])) == [3, 4]
        @test Array{Int}(SizedArray{Tuple{2}, Int, 1}([3, 4])) == [3, 4]
        @test Array{Int, 1}(SizedArray{Tuple{2}, Int, 1}([3, 4])) == [3, 4]
        @test Vector(SizedArray{Tuple{4}, Int, 1}(collect(3:6))) == collect(3:6)
        @test convert(Vector, SizedArray{Tuple{4}, Int, 1}(collect(3:6))) == collect(3:6)
        @test Matrix(SMatrix{2,2}((1,2,3,4))) == [1 3; 2 4]
        @test convert(Matrix, SMatrix{2,2}((1,2,3,4))) == [1 3; 2 4]
        @test convert(Array, SizedArray{Tuple{2,2,2,2}, Int}(ones(2,2,2,2))) == ones(2,2,2,2)
        # Conversion after reshaping
        @test Array(SizedMatrix{2,2}([1,2,3,4])) == [1 3; 2 4]
        # Array(a::Array) makes a copy so this should work similarly
        a = [1 2; 3 4]
        @test Array(SizedMatrix{2,2}(a)) !== a
        @test @inferred(convert(Array, SizedMatrix{2,2}(a))) === a
        @test @inferred(convert(Array{Int}, SizedMatrix{2,2}(a))) === a
        @test @inferred(convert(Matrix{Int}, SizedMatrix{2,2}(a))) === a
        @test @inferred(convert(Matrix{Float64}, SizedMatrix{2,2}(a))) == a
    end

    @testset "promotion" begin
        @test @inferred(promote_type(SizedVector{1,Float64}, SizedVector{1,BigFloat})) == SizedVector{1,BigFloat}
        @test @inferred(promote_type(SizedVector{2,Int}, SizedVector{2,Float64})) === SizedVector{2,Float64}
        @test @inferred(promote_type(SizedMatrix{2,3,Float32,2}, SizedMatrix{2,3,Complex{Float64},2})) === SizedMatrix{2,3,Complex{Float64},2}
        @test @inferred(promote_type(SizedArray{Tuple{2,2},Float64,2,2,Matrix{Float64}}, SizedArray{Tuple{2,2},BigFloat,2,2,Matrix{BigFloat}})) == SizedArray{Tuple{2,2},BigFloat,2,2,Matrix{BigFloat}}
    end

    @testset "reshaping" begin
        y = rand(4,1,2)
        sy = SizedArray{Tuple{size(y)...}}(y)

        @test vec(sy) isa SizedArray{Tuple{8}, Float64}
        @test reshape(sy, Size(2,4)) isa SizedArray{Tuple{2, 4}, Float64}
    end

    @testset "sized views" begin
        x = rand(4,1,2)
        y = SizedMatrix{4,2}(view(x, :, 1, :))

        @test isa(y, SizedArray{Tuple{4,2},Float64,2,2,<:SubArray{Float64,2}})
        @test Array(y) isa Matrix{Float64}
        @test Array(y) == x[:, 1, :]
        @test convert(Array, y) isa Matrix{Float64}
        @test convert(Array, y) == x[:, 1, :]
        y[3, 2] = 18
        @test x[3, 1, 2] == 18

        x2 = rand(10)
        y2 = SizedMatrix{4,2}(view(x2, 1:8))
        @test isa(y2, SizedArray{Tuple{4,2},Float64,2,1,<:SubArray{Float64,1}})
        @test Array(y2) isa Matrix{Float64}
        @test Array(y2) == reshape(x2[1:8], 4, 2)
        @test convert(Array, y2) isa Matrix{Float64}
        @test convert(Array, y2) == reshape(x2[1:8], 4, 2)
    end

    @testset "views of sized arrays" begin
        x = SizedArray{Tuple{4,3,2}}(rand(4, 3, 2))
        x1 = view(x, :, 1, 2)
        @test isa(x1, SizedArray{Tuple{4},Float64,1,1,<:SubArray{Float64,1}})
        @test x1 == view(x.data, :, 1, 2)

        x2 = view(x, :, SA[1], 2)
        @test isa(x2, SizedArray{Tuple{4,1},Float64,2,2,<:SubArray{Float64,2}})
        @test x2 == view(x.data, :, SA[1], 2)

        x3 = view(x, SOneTo(3), 1, SA[2])
        @test isa(x3, SizedArray{Tuple{3,1},Float64,2,2,<:SubArray{Float64,2}})
        @test x3 == view(x.data, SOneTo(3), 1, SA[2])

        x4 = view(x, Base.Slice(SOneTo(4)), 1, SA[2])
        @test isa(x4, SizedArray{Tuple{4,1},Float64,2,2,<:SubArray{Float64,2}})
        @test x4 == view(x.data, Base.Slice(SOneTo(4)), 1, SA[2])

        x5 = view(x, :)
        @test isa(x5, SizedArray{Tuple{24},Float64,1,1,<:SubArray{Float64,1}})
        @test x5 == view(x.data, :)
    end

    @testset "views of MArray" begin
        x = MArray{Tuple{4,3,2}}(rand(4, 3, 2))
        x1 = view(x, :, 1, 2)
        @test isa(x1, SizedArray{Tuple{4},Float64,1,1,<:SubArray{Float64,1}})
        @test x1 == view(Array(x), :, 1, 2)

        x2 = view(x, :, SA[1], 2)
        @test isa(x2, SizedArray{Tuple{4,1},Float64,2,2,<:SubArray{Float64,2}})
        @test x2 == view(Array(x), :, SA[1], 2)

        x3 = view(x, SOneTo(3), 1, SA[2])
        @test isa(x3, SizedArray{Tuple{3,1},Float64,2,2,<:SubArray{Float64,2}})
        @test x3 == view(Array(x), SOneTo(3), 1, SA[2])

        x4 = view(x, Base.Slice(SOneTo(4)), 1, SA[2])
        @test isa(x4, SizedArray{Tuple{4,1},Float64,2,2,<:SubArray{Float64,2}})
        @test x4 == view(Array(x), Base.Slice(SOneTo(4)), 1, SA[2])

        x5 = view(x, :)
        @test isa(x5, SizedArray{Tuple{24},Float64,1,1,<:SubArray{Float64,1}})
        @test x5 == view(Array(x5), :)
    end
end

struct OVector <: AbstractVector{Int} end
Base.length(::OVector) = 10
Base.axes(::OVector) = (0:9,)
@test_throws ArgumentError SizedVector{10}(OVector())
