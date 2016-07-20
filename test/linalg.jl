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

    @testset "eye()" begin
        @test eye(SMatrix{2,2,Int}) === @SMatrix [1 0; 0 1]
        @test eye(SMatrix{2,2}) === @SMatrix [1.0 0.0; 0.0 1.0]
        @test eye(SMatrix{2}) === @SMatrix [1.0 0.0; 0.0 1.0]

        @test eye(MMatrix{2,2,Int})::MMatrix == @MMatrix [1 0; 0 1]
        @test eye(MMatrix{2,2})::MMatrix == @MMatrix [1.0 0.0; 0.0 1.0]
        @test eye(MMatrix{2})::MMatrix == @MMatrix [1.0 0.0; 0.0 1.0]
    end
#=
    @testset "StaticVector and StaticMatrix constructors" begin
        sv = SArray{(3,)}((1,2,3))
        mv = MArray{(3,)}((1,2,3))

        sm = SMatrix{(2,2)}((3,4,5,6))
        mm = MMatrix{(2,2)}((3,4,5,6))

        @test SVector((1,2,3)) === sv
        @test_inferred SVector((1,2,3))
        @test SVector{(3,)}((1,2,3)) === sv
        @test_inferred SVector{(3,)}((1,2,3))

        @test SMatrix{(2,2)}((3,4,5,6)) === sm
        @test_inferred SMatrix{(2,2)}((3,4,5,6))

        @test MVector((1,2,3)) == mv
        @test_inferred SVector((1,2,3))
        @test SVector{(3,)}((1,2,3)) == mv
        @test_inferred SVector{(3,)}((1,2,3))

        @test MMatrix{(2,2)}((3,4,5,6)) == mm
        @test_inferred SMatrix{(2,2)}((3,4,5,6))
    end

    @testset "Conversion with AbstractVector and AbstractMatrix" begin
        sv = SArray{(3,)}((1,2,3))
        mv = MArray{(3,)}((1,2,3))
        v = [1,2,3]

        sm = SMatrix{(2,2)}((3,4,5,6))
        mm = MMatrix{(2,2)}((3,4,5,6))
        m = [3 5; 4 6]

        @test convert(SVector{(3,)}, v) === sv
        @test_inferred convert(SVector{(3,)}, v)
        @test convert(SMatrix{(2,2)}, m) === sm
        @test_inferred convert(SMatrix{(2,2)}, m)

        @test convert(MVector{(3,)}, v) == mv
        @test_inferred convert(MVector{(3,)}, v)
        @test convert(MMatrix{(2,2)}, m) == mm
        @test_inferred convert(MMatrix{(2,2)}, m)

        @test convert(Vector, sv) == v
        @test_inferred convert(Vector,sv)
        @test convert(Matrix, sm) == m
        @test_inferred convert(Matrix,sm)

        @test convert(Vector, mv) == v
        @test_inferred convert(Vector,mv)
        @test convert(Matrix, mm) == m
        @test_inferred convert(Matrix,mm)
    end =#
end
