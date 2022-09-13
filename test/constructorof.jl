using StaticArrays
using Test
using ConstructionBase: constructorof

@testset "constructorof" begin
    sa = @SVector [2, 4, 6, 8]
    sa2 = constructorof(typeof(sa))((3.0, 5.0, 7.0, 9.0))
    @test sa2 === @SVector [3.0, 5.0, 7.0, 9.0]

    ma = @MMatrix [2.0 4.0; 6.0 8.0]
    ma2 = constructorof(typeof(ma))((1, 2, 3, 4))
    @test ma2 isa MArray{Tuple{2,2},Int,2,4}
    @test all(ma2 .=== @MMatrix [1 3; 2 4])

    for T in (SVector, MVector)
        @test constructorof(T)((1, 2, 3))::T == T((1, 2, 3))
        @test constructorof(T{3})((1, 2, 3))::T == T((1, 2, 3))
        @test constructorof(T{3})((1, 2))::T == T((1, 2))
        @test constructorof(T{3, Symbol})((1, 2, 3))::T == T((1, 2, 3))
        @test constructorof(T{3, Symbol})((1, 2))::T == T((1, 2))
        @test constructorof(T{3, X} where {X})((1, 2, 3))::T == T((1, 2, 3))
        @test constructorof(T{3, X} where {X})((1, 2))::T == T((1, 2))
        @test constructorof(T{X, Symbol} where {X})((1, 2, 3))::T == T((1, 2, 3))
    end
end
