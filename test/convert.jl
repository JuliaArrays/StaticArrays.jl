using StaticArrays, Test

@testset "Copy constructors" begin
    M = [1 2; 3 4]
    SizeM = SizedMatrix{2,2}(M)
    @test typeof(SizeM)(SizeM).data === M
end # testset

@testset "Constructors of zero size arrays" begin
    # Issue #520
    @testinf SVector{0}(Int8[]) === SVector{0,Int8}()
    @testinf SMatrix{0,0}(zeros(0,0)) === SMatrix{0,0,Float64}(())

    # Issue #651
    @testinf SVector{0,Float64}(Any[]) === SVector{0,Float64}()
    @testinf SVector{0,Float64}(Int8[]) === SVector{0,Float64}()

    # PR #808
    @test Scalar{Int}[SVector{1,Int}(3), SVector{1,Float64}(2.0)] == [Scalar{Int}(3), Scalar{Int}(2)]
    @test Scalar[SVector{1,Int}(3), SVector{1,Float64}(2.0)] == [Scalar{Int}(3), Scalar{Float64}(2.0)]
end

@testset "`real` and `float` of SArray/MArray" begin
    # Issue 935
    for SAT in (SArray, MArray)
        vInt = SAT(SA[1,2,3])           # S/MVector{3, Int}
        @test real(typeof(vInt)) === typeof(vInt)
        @test float(typeof(vInt)) === typeof(float.(vInt))

        vCInt = vInt + 1im*vInt         # S/MVector{3, Complex{Int}}
        @test real(typeof(vCInt)) === typeof(vInt)
        @test float(typeof(vCInt)) === typeof(float.(vCInt))
        
        vvInt = SAT(SA[vInt, vInt])    # S/MVector{2, S/MVector{3, Int}}
        @test real(typeof(vvInt)) === SAT{Tuple{2}, SAT{Tuple{3}, Int, 1, 3}, 1, 2}
        @test float(typeof(vvInt)) === SAT{Tuple{2}, SAT{Tuple{3}, Float64, 1, 3}, 1, 2}

        vvCInt = SAT(SA[vCInt, vCInt]) # S/MVector{2, S/MVector{3, Complex{Int}}}
        @test real(typeof(vvCInt)) === SAT{Tuple{2}, SAT{Tuple{3}, Int, 1, 3}, 1, 2}
        @test float(typeof(vvCInt)) === SAT{Tuple{2}, SAT{Tuple{3}, Complex{Float64}, 1, 3}, 1, 2}
    end
    mInt = SA[Int16(1) Int16(2) Int16(3); Int16(4) Int16(5) Int16(6)] # SMatrix{3,2,Int16}
    @test float(typeof(mInt)) === SMatrix{2, 3, float(Int16), 6}
end

@testset "convert with missing/wrong size" begin
    @test convert(SVector, MVector(1,2,3)) === SVector(1,2,3)
    @test convert(SMatrix{3}, MVector(1,2,3)) === SMatrix{3,1}(1,2,3)
    @test_throws Exception convert(SVector{1}, MVector(1,2,3)) 
end

struct BugStaticVector <: StaticVector{2,Int} end
@testset "Unknow StaticArray" begin
    # Make sure the new `construct_type` cause no regression.
    @test_throws DimensionMismatch BugStaticVector(1,2,3)
    f(x) = StaticArrays.construct_type(BugStaticVector, StaticArrays.Args(x))
    @test_inlined f((1,2,3,4))
end

using OffsetArrays
@testset "constructor/convert from OffsetArray" begin
    a = OffsetArray([-1 1;0 2], -1, -1)
    b = OffsetArray([-1,0,1,2], -1)
    c = OffsetArray(-1:2, -1)
    d = Base.IdentityUnitRange(-1:2)
    @test SVector{4}(a) === SVector{4}(b) === SVector{4}(c) === SVector{4}(d) == [-1,0,1,2]
    @test SMatrix{2,2}(a) === SMatrix{2,2}(b) === SMatrix{2,2}(c) === SMatrix{2,2}(d) == [-1 1;0 2]
end
