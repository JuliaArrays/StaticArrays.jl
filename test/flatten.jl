using StaticArrays, Test

@testset "Iterators.flatten" begin
    for x in [SVector(1.0, 2.0), MVector(1.0, 2.0),
            @SMatrix([1.0 2.0; 3.0 4.0]), @MMatrix([1.0 2.0]),
            SizedMatrix{1,2}([1.0 2.0])
            ]
        X = [x,x,x]
        @test length(Iterators.flatten(X)) == length(X)*length(x)
        @test collect(Iterators.flatten(typeof(x)[])) == []
        @test collect(Iterators.flatten(X)) == [x..., x..., x...]
    end
    @test collect(Iterators.flatten([SVector(1,1), SVector(1)])) == [1,1,1]
    @test_throws ArgumentError length(Iterators.flatten([SVector(1,1), SVector(1)]))
end
