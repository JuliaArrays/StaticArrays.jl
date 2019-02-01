using StaticArrays, Test

@testset "Copy constructors" begin
    M = [1 2; 3 4]
    SizeM = Size(2,2)(M)
    @test typeof(SizeM)(SizeM).data === M
end # testset
