@testset "[;;;;;]" begin
    @test_broken ((@SVector [;])::SVector{0}).data === ()
    test_expand_error(:(@SVector [;;]))
    @test_broken ((@MVector [;])::MVector{0}).data === ()
    test_expand_error(:(@MVector [;;]))

    @test ((@SMatrix [;;])::SMatrix{0,0}).data === ()
    test_expand_error(:(@SMatrix [;;;]))
    @test ((@MMatrix [;;])::MMatrix{0,0}).data === ()
    test_expand_error(:(@MMatrix [;;;]))
end
