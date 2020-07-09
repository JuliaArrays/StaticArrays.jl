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
