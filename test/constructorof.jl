using StaticArrays, Test, ConstructionBase

@testset "constructorof" begin
    sa = @SVector [2, 4, 6, 8]
    sa2 = ConstructionBase.constructorof(typeof(sa))((3.0, 5.0, 7.0, 9.0))
    @test sa2 === @SVector [3.0, 5.0, 7.0, 9.0]

    ma = @MMatrix [2.0 4.0; 6.0 8.0]
    ma2 = ConstructionBase.constructorof(typeof(ma))((1, 2, 3, 4))
    @test ma2 isa MArray{Tuple{2,2},Int,2,4}
    @test all(ma2 .=== @MMatrix [1 3; 2 4])

    sz = SizedArray{Tuple{2,2}}([1 2;3 4])
    sz2 = ConstructionBase.constructorof(typeof(sz))([:a :b; :c :d]) 
    @test sz2 == SizedArray{Tuple{2,2}}([:a :b; :c :d])
    @test typeof(sz2) <: SizedArray{Tuple{2,2},Symbol,2,2}
end
