using StaticArrays, Test

@testset "Copy constructors" begin
    M = [1 2; 3 4]
    SizeM = Size(2,2)(M)
    @test typeof(SizeM)(SizeM).data === M
end # testset

@testset "Constructors of zero size arrays" begin
    # Issue #520
    @testinf SVector{0}(Int8[]) === SVector{0,Int8}()
    @testinf SMatrix{0,0}(zeros(0,0)) === SMatrix{0,0,Float64}(())
end
