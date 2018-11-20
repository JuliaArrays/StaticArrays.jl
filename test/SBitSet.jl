using StaticArrays, Test

@testset "SBitSet" begin
    v1 = @SVector UInt64[2,4,6,8]
    v2 = @SVector UInt64[4,3,2,1]
    sbs1 = SBitSet(v1)
    sbs2 = SBitSet(v2)
    z = SBitSet(xor(v1,v1))


    @test isempty(sbs1) === false
    @test isempty(z) === true

    @test length(sbs1) === 5
    @test length(z) === 0
    @test in(67,sbs1) == true
    @test in(135, sbs1) == false

    z = SBitSet(@SVector [-Int64(1)%UInt64])
    @test in(64, z)==true
    @test in(1, z)==true
    @test in(0, z)==false
    @test in(65, z)==false
    @test in(-1, z)==false


    @test collect(sbs1)::Vector{Int64} == [2, 64+3, 128+2, 128+3, 192+4]
    @test collect(sbs2) == [3, 64+1, 64+2, 128+2, 192+1]
    @test collect(intersect(sbs1,sbs2)) == [128+2]
    @test collect(union(sbs1,sbs2)) == [2, 3, 64+1, 64+2, 64+3, 128+2, 128+3, 192+1, 192+4]
    @test collect(setdiff(sbs1,sbs2)) == [2, 64+3, 128+3, 192+4]
    @test collect(symdiff(sbs1,sbs2)) == [2, 3, 64+1, 64+2, 64+3, 128+3, 192+1, 192+4]

    @test issubset(sbs1, sbs2) == false
    z = intersect(sbs1,sbs2)
    @test issubset(z,sbs1) == true
    @test issubset(sbs1,z) == false
    @test issubset(sbs1,sbs1) == true

    @inferred collect(sbs1)
end