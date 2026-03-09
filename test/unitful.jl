using Unitful

@testset "Unitful" begin
    # issue #1124
    @test norm(SVector(1.0*u"m")) == 1.0*u"m"
    # issue $1127
    @test norm(SVector(0.0, 0.0)*u"nm") == 0.0*u"nm"

    @test norm(SVector(1.0, 2.0)*u"m", 1) == 3.0*u"m"
    @test norm(SVector(0.0, 0.0)*u"nm", 1) == 0.0*u"nm"

    @test norm([SVector(1.0u"m", 2.0u"m")]) == norm([1.0u"m", 2.0u"m"])
    @test isapprox(SVector(1.0u"m", 2.0u"m"), SVector(1.1u"m", 2.1u"m"); atol=0.2u"m")
end
