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
        @test_throws DimensionMismatch SizedArray([3, 4])
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
        @test SizedArray{Tuple{4}}([1 2; 3 4]) == vec([1 2; 3 4])
        @test_throws DimensionMismatch SizedArray{Tuple{1,4}}([1 2; 3 4])
        # Uninitialized
        @test @inferred(SizedArray{Tuple{2,2},Int,2,2,Matrix{Int}}(undef)) isa SizedArray{Tuple{2,2},Int,2,2,Matrix{Int}}
        @test @inferred(SizedArray{Tuple{2,2},Int,2,2}(undef)) isa SizedArray{Tuple{2,2},Int,2,2,Matrix{Int}}
        @test @inferred(SizedArray{Tuple{2,2},Int,2,1}(undef)) isa SizedArray{Tuple{2,2},Int,2,1,Vector{Int}}
        @test @inferred(SizedArray{Tuple{2,2},Int,2}(undef)) isa SizedArray{Tuple{2,2},Int,2,2,Matrix{Int}}
        @test @inferred(SizedArray{Tuple{2,2},Int}(undef)) isa SizedArray{Tuple{2,2},Int,2,2,Matrix{Int}}
        @test size(SizedArray{Tuple{4,5},Int,2}(undef).data) == (4, 5)
        @test size(SizedArray{Tuple{4,5},Int}(undef).data) == (4, 5)

        # 0-element constructor
        @test (@inferred SizedArray(MMatrix{0,0,Float64}()))::SizedMatrix{0,0,Float64} == SizedMatrix{0,0,Float64}()

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
        @test @inferred(SizedMatrix{2,2,Int}(SVector(1,2,3,4)))::SizedMatrix{2,2,Int,2} == [1 3; 2 4]
        @test @inferred(SizedMatrix{2,2,Int,1}(SVector(1,2,3,4)))::SizedMatrix{2,2,Int,1} == [1 3; 2 4]
        @test @inferred(SizedMatrix{2,2,Int,1,SVector{4,Int}}(SVector(1,2,3,4)))::SizedMatrix{2,2,Int,1} == [1 3; 2 4]
        # Back to Matrix
        @test Matrix(SizedMatrix{2,2}([1 2;3 4])) == [1 2; 3 4]
        @test convert(Matrix, SizedMatrix{2,2}([1 2;3 4])) == [1 2; 3 4]

        # 0-element constructor
        @test (@inferred SizedMatrix(MMatrix{0,0,Float64}()))::SizedMatrix{0,0,Float64} == SizedMatrix{0,0,Float64}()
    end

    # setindex
    @testset "setindex" begin
        sa = SizedArray{Tuple{2}, Int, 1}([3, 4])
        sa[1] = 2
        @test sa.data == [2, 4]
        @test setindex!(sa, 2, 1) === sa

        sm = SizedArray{Tuple{4,3}}(rand(4, 3))
        sm[1] = 2
        @test sa.data[1] == 2
        sm[2, 3] = 4
        @test sm.data[2, 3] == 4
        @test setindex!(sm, 0.5, 1) === sm
        @test setindex!(sm, 0.5, 2, 3) === sm
    end

    # parent
    sa = SizedArray{Tuple{2}, Int, 1}([3, 4])
    @test parent(sa) === sa.data

    # pointer
    @testset "pointer" begin
        @test Base.cconvert(Ptr{Int}, sa) === Base.cconvert(Ptr{Int}, sa.data)
        @test pointer(sa) === pointer(sa.data)

        A = MMatrix{32,3,Float64}(undef);
        av1 = view(A, 1, :);
        av2 = view(A, :, 1);
        @test pointer(A) == pointer(av1) == pointer(av2)
        @test pointer(A, LinearIndices(A)[1,2]) == pointer(av1, 2)
        @test pointer(A, LinearIndices(A)[2,1]) == pointer(av2, 2)
        @test pointer(A, LinearIndices(A)[4,3]) == pointer(view(A, :, 3), 4) == pointer(view(A, 4, :), 3)
    end
    
    @testset "vec" begin
        sa2 = SizedArray{Tuple{2, 2}, Int}([1, 2, 3, 4])
        @test (@inferred vec(sa2)) isa SizedVector{4, Int}
        @test vec(sa2).data === vec(sa2.data)

        M = @MMatrix [1 2; 3 4]
        @test copy(vec(M)) isa SizedVector{4, Int}
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

        x6 = view(x, StaticArrays.SUnitRange(2, 3), :, 2)
        @test isa(x6, SizedArray{Tuple{2,3},Float64,2,2,<:SubArray{Float64,2}})
        @test x6 == view(x.data, StaticArrays.SUnitRange(2, 3), :, 2)

        x7 = view(x, :, SA[true, false, true], 1)
        @test x7 == view(x.data, :, SA[true, false, true], 1)
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

    @testset "reverse" begin
        x = SizedArray{Tuple{4,3,2}}(rand(4, 3, 2))
        @test reverse(x) == reverse(reverse(reverse(collect(x), dims = 3), dims = 2), dims = 1)
    end
end

struct OVector <: AbstractVector{Int} end
Base.length(::OVector) = 10
Base.axes(::OVector) = (0:9,)
@test_throws ArgumentError SizedVector{10}(OVector())

@testset "some special case" begin
    @test_throws Exception SizedVector{1}(1, 2)
    @test (@inferred(SizedVector{1}((1, 2)))::SizedVector{1,NTuple{2,Int}}) == [(1, 2)]
    @test (@inferred(SizedVector{2}((1, 2)))::SizedVector{2,Int}) == [1, 2]
    @test (@inferred(SizedVector(1, 2))::SizedVector{2,Int}) == [1, 2]
    @test (@inferred(SizedVector((1, 2)))::SizedVector{2,Int}) == [1, 2]

    @test_throws Exception SizedMatrix{1,1}(1, 2)
    @test (@inferred(SizedMatrix{1,1}((1, 2)))::SizedMatrix{1,1,NTuple{2,Int}}) == fill((1, 2),1,1)
    @test (@inferred(SizedMatrix{1,2}((1, 2)))::SizedMatrix{1,2,Int}) == reshape(1:2, 1, 2)
    @test (@inferred(SizedMatrix{1}((1, 2)))::SizedMatrix{1,2,Int}) == reshape(1:2, 1, 2)
    @test (@inferred(SizedMatrix{1}(1, 2))::SizedMatrix{1,2,Int}) == reshape(1:2, 1, 2)
    @test (@inferred(SizedMatrix{2}((1, 2)))::SizedMatrix{2,1,Int}) == reshape(1:2, 2, 1)
    @test (@inferred(SizedMatrix{2}(1, 2))::SizedMatrix{2,1,Int}) == reshape(1:2, 2, 1)
end
