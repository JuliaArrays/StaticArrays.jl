@testset "Linear algebra" begin
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
    end
end
