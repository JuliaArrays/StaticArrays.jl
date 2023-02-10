using Unitful

@testset "Unitful" begin
    # issue #1124
    @test norm(SVector(1.0*u"m")) == 1.0*u"m"
end
