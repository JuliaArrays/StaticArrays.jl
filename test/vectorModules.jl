using StaticArrays.Float64Vectors

@testset "Float64Vectors" begin
    @test Vector3D == SArray{Tuple{3},Float64,1,3}
    @test Vector2D == SArray{Tuple{2},Float64,1,2}

    v3 = Vector3D(1, 2, 3)
    @test typeof(v3) == SArray{Tuple{3},Float64,1,3}
    @test v3[1] == 1.0
    @test v3[2] == 2.0
    @test v3[3] == 3.0

    v2 = Vector2D(10, 20)
    @test typeof(v2) == SArray{Tuple{2},Float64,1,2}
    @test v2[1] == 10.0
    @test v2[2] == 20.0

    @test Vector3D(1,2,3) == Vector3D(1.0, 2.0, 3.0)
    @test Vector2D(10,20) == Vector2D(10.0, 20.0)        
end