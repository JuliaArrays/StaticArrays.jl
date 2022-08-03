@testset "SUnitRange" begin
    @test length(StaticArrays.SUnitRange(1,3)) === 3
    @test length(StaticArrays.SUnitRange(1,-10)) === 0

    @test_throws BoundsError StaticArrays.SUnitRange(2,4)[0]
    @test StaticArrays.SUnitRange(2,4)[1] === 2
    @test StaticArrays.SUnitRange(2,4)[2] === 3
    @test StaticArrays.SUnitRange(2,4)[3] === 4
    @test_throws BoundsError StaticArrays.SUnitRange(2,4)[4]

    @test_throws Exception StaticArrays.SUnitRange{1, -1}()
    @test_throws TypeError StaticArrays.SUnitRange{1, 1.5}()

    let su = StaticArrays.SUnitRange(1, 2),
        str = "SUnitRange(1,2)",
        str2 = "($(str), $(str))"
        @test repr(su) == str
        @test repr("text/plain", su) == str
        @test repr((su, su)) == str2
        @test repr("text/plain", (su, su)) == str2
    end
end
