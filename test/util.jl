using StaticArrays, Test

import StaticArrays: drop_sdims

@testset "util" begin
    @test !(drop_sdims(SVector(1,2)) isa StaticArray)
    @test !(drop_sdims([1,2]) isa StaticArray)
    @test !(drop_sdims(SizedVector{2}([1,2])) isa StaticArray)

    @test drop_sdims(SVector(1,2)) == SVector(1,2)
    @test drop_sdims(@SMatrix [1 2; 3 4]) == [1 2; 3 4]
    # Check this works for SizedArray where it's tempting to *unwrap* rather
    # than add an extra wrapper.  Currently unwrapping doesn't work due to
    # reshaping issues.
    @test drop_sdims(SizedMatrix{2,2}([1 2; 3 4])) == [1 2; 3 4]
    @test drop_sdims(SizedMatrix{2,3}([1 4;
                                       2 5;
                                       3 6])) == [1 3 5;
                                                  2 4 6]
end
