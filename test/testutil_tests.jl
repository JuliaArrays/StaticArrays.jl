@testset "test utils" begin
    @testset "@testinf" begin
        @testinf [1,2] == [1,2]
        x = [1,2]
        @testinf x == [1,2]
        @testinf (@SVector [1,2]) == (@SVector [1,2])
    end

    @testset "@test_inlined" begin
        @test_inlined should_be_inlined(1)
        @test_inlined should_not_be_inlined(1) false
        ts = @testset ErrorCounterTestSet "" begin
            @test_inlined should_be_inlined(1) false
            @test_inlined should_not_be_inlined(1)
        end
        @test ts.errorcount == 0 && ts.failcount == 2 && ts.passcount == 0
    end
end
