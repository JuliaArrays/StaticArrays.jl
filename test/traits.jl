using StaticArrays, Base.Test

import StaticArrays: SizeClass, Small, Large

@testset "SizeClass" begin
    @test @inferred(SizeClass(eye(SMatrix{2,3}), Size(3,3))) === Small()
    @test @inferred(SizeClass(eye(SMatrix{3,3}), Size(3,3))) === Small()
    @test @inferred(SizeClass(eye(SMatrix{4,3}), Size(3,3))) === Large()

    @test @inferred(SizeClass(eye(SMatrix{3,3}), Length(9))) === Small()
    @test @inferred(SizeClass(eye(SMatrix{3,3}), Length(8))) === Large()
end
