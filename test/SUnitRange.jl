@testset "SUnitRange" begin
    @test length(SUnitRange(1,3)) === 3
    @test length(SUnitRange(1,-10)) === 0

    @test_throws BoundsError SUnitRange(2,4)[0]
    @test SUnitRange(2,4)[1] === 2
    @test SUnitRange(2,4)[2] === 3
    @test SUnitRange(2,4)[3] === 4
    @test_throws BoundsError SUnitRange(2,4)[4]

    @test_throws Exception SUnitRange{1, -1}()
    @test_throws TypeError SUnitRange{1, 1.5}()

    ur_str = sprint(show, SUnitRange)
    @test ur_str == "SUnitRange"
end
