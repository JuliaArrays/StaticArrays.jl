using Unitful

@testset "Unitful" begin
    # issue #1124
    @test norm(SVector(1.0*u"m")) == 1.0*u"m"
    # issue $1127
    @test norm(SVector(0.0, 0.0)*u"nm") == 0.0*u"nm"
end
