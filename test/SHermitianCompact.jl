module SHermitianCompactTest

using StaticArrays
using LinearAlgebra
using Random
using Test

macro test_noalloc(ex)
    esc(quote
        $ex
        @test(@allocated($ex) == 0)
    end)
end

function check_lower_triangle(a::SHermitianCompact, triangle::SVector)
    N = size(a, 1)
    i = 0
    for col = 1 : N, row = 1 : N
        if row >= col
            i += 1
            @test a[row, col] == triangle[i]
        else
            @test a[row, col] == adjoint(a[col, row])
        end
    end
end

function check_lower_triangle(a::SHermitianCompact{3}, triangle::SVector)
    # because the generic check_lower_triangle uses almost the same code as what's being tested
    @test a[1, 1] == triangle[1]
    @test a[2, 1] == triangle[2]
    @test a[3, 1] == triangle[3]
    @test a[1, 2] == adjoint(a[2, 1])
    @test a[2, 2] == triangle[4]
    @test a[3, 2] == triangle[5]
    @test a[1, 3] == adjoint(a[3, 1])
    @test a[2, 3] == adjoint(a[3, 2])
    @test a[3, 3] == triangle[6]
end

fill3(x) = fill(3, x)

@testset "SHermitianCompact" begin
    @testset "Inner Constructor" begin
        for (N, L) in ((3, 6), (4, 10), (6, 21))
            for T in (Int32, Int64)
                @eval begin
                    let lowertriangle = rand(SVector{$L, Int32})
                        a = SHermitianCompact{$N, $T, $L}(lowertriangle)
                        check_lower_triangle(a, lowertriangle)
                        @test a isa SHermitianCompact{$N, $T, $L}
                        @test(@allocated(SHermitianCompact{$N, $T, $L}(lowertriangle)) == 0)
                    end
                end
            end
        end
        @test_throws ArgumentError SHermitianCompact{3, Int64, 7}(rand(SVector{7, Int32}))
    end

    @testset "Outer Constructors" begin
        @test @inferred(SHermitianCompact(MVector(1,2,3))) === SHermitianCompact(SVector(1,2,3))
        @test_throws Exception SHermitianCompact(1,2,3)
        @test_throws Exception SHermitianCompact(1,2,3,4,5)
        @test @inferred(SHermitianCompact(1,2,3,4)) === SHermitianCompact(SVector(1,2,4))

        for (N, L) in ((3, 6), (4, 10), (6, 21))
            for T in (Int32, Int64)
                @eval begin
                    let lowertriangle = rand(SVector{$L, Int32})
                        a1 = SHermitianCompact{$N, $T}(lowertriangle)
                        check_lower_triangle(a1, $T.(lowertriangle))
                        @test a1 isa SHermitianCompact{$N, $T, $L}
                        @test_noalloc SHermitianCompact{$N, $T}(lowertriangle)

                        a2 = SHermitianCompact{$N}(lowertriangle)
                        check_lower_triangle(a2, lowertriangle)
                        @test a2 isa SHermitianCompact{$N, Int32, $L}
                        @test_noalloc SHermitianCompact{$N}(lowertriangle)

                        a3 = SHermitianCompact(lowertriangle)
                        check_lower_triangle(a3, lowertriangle)
                        @test a3 isa SHermitianCompact{$N, Int32, $L}
                        @test_noalloc SHermitianCompact(lowertriangle)

                        a4 = SHermitianCompact{$N, $T}(a3)
                        @test a4 === a1
                        @test_noalloc SHermitianCompact{$N, $T}(a3)
                    end
                    let a = rand(SMatrix{$N, $N, Int32})
                        @test SHermitianCompact{$N, $T, $L}(a) isa SHermitianCompact{$N, $T, $L}
                        @test_noalloc SHermitianCompact{$N, $T, $L}(a)
                        @test SHermitianCompact{$N, $T, $L}(a) == Hermitian($T.(a), :L)

                        @test SHermitianCompact{$N, $T}(a) isa SHermitianCompact{$N, $T, $L}
                        @test_noalloc SHermitianCompact{$N, $T}(a)
                        @test SHermitianCompact{$N, $T}(a) == Hermitian($T.(a), :L)

                        @test SHermitianCompact{$N}(a) isa SHermitianCompact{$N, Int32, $L}
                        @test_noalloc SHermitianCompact{$N}(a)
                        @test SHermitianCompact{$N}(a) == Hermitian(a, :L)

                        @test SHermitianCompact(a) isa SHermitianCompact{$N, Int32, $L}
                        @test_noalloc SHermitianCompact(a)
                        @test SHermitianCompact(a) == Hermitian(a, :L)
                    end
                end
            end
        end
    end

    @testset "convert" begin
        a = SHermitianCompact(SMatrix{3, 3}(1 : 9))
        @test convert(typeof(a), a) === a
        @test convert(SHermitianCompact{3, Float64}, a) isa SHermitianCompact{3, Float64, 6}
        @test convert(SHermitianCompact{3, Float64}, a) == Float64.(Array(a))
        @test convert(SHermitianCompact{3, Float64, 6}, a) isa SHermitianCompact{3, Float64, 6}
    end

    @testset "setindex" begin
        for (N, L) in ((3, 6), (4, 10), (6, 21))
            @eval begin
                let lowertriangle = zero(SVector{$L, Int32})
                    a = SHermitianCompact(lowertriangle)
                    @test_noalloc setindex(a, 2., 1, 1)
                    for col = 1 : $N, row = 1 : $N
                        x = Float64(rand(Int32))
                        a2 = setindex(a, x, row, col)
                        @test a2[row, col] === Int32(x)
                        @test setindex(a2, 0, row, col) === a
                    end
                end
            end
        end
    end

    @testset "ishermitian / issymmetric" begin
        a = rand(SHermitianCompact{5, Float64})
        @test ishermitian(a)
        @test issymmetric(a)

        b = rand(SHermitianCompact{5, ComplexF64})
        @test !ishermitian(b)
        @test !issymmetric(b)

        c = b + conj(b)
        @test ishermitian(c)
        @test issymmetric(c)
        @test_noalloc ishermitian(c)

        d = b + b'
        @test ishermitian(d)
        @test !issymmetric(d)

        e = rand(SHermitianCompact{5, Float64}) + im*I
        @test !ishermitian(e)
        @test issymmetric(e)
    end

    @testset "==" begin
        a = SHermitianCompact(SVector{6}(-5 : 0))
        b = setindex(a, 5, 3)
        @test a == a
        @test !(a == b)
        @test_noalloc a == a
    end

    @testset "Arithmetic" begin
        a = SHermitianCompact(SVector{6, Int}(1 : 6))
        b = SHermitianCompact(SVector(-4, 5, 4, 8, -10, 11))
        c = SMatrix{3, 3}(-9 : -1)

        let a = a
            @test -a == -SMatrix(a)
            @test -a isa SHermitianCompact{3, Int, 6}
        end
        for (x, y) in ((a, b), (a, c), (c, a))
            @eval begin
                let x = $x, y = $y
                    @test x + y == SMatrix(x) + SMatrix(y)
                    @test_noalloc x + y

                    @test x - y == SMatrix(x) - SMatrix(y)
                    @test_noalloc x - y

                    if x isa SHermitianCompact && y isa SHermitianCompact
                        @test x + y isa SHermitianCompact{3, Int, 6}
                        @test x - y isa SHermitianCompact{3, Int, 6}
                    end
                end
            end
        end
    end

    @testset "Scalar-array" begin
        x = SHermitianCompact(SVector{6, Int}(1 : 6))
        for y = (-5, 1.1+4.3im)
            for op in (:*, :/, :\, :(Base.FastMath.mul_fast))
                if op != :\
                    @eval begin
                        @test $op($x, $y) == $op(SMatrix($x), $y)
                        @test_noalloc $op($x, $y)
                    end
                end

                if op != :/
                    @eval begin
                        @test $op($y, $x) == $op($y, SMatrix($x))
                        @test_noalloc $op($y, $x)
                    end
                end
            end
            @eval begin
                @test muladd($y, $x, $x) == muladd($y, SMatrix($x), $x)
                @test_noalloc muladd($y, $x, $x)
                @test muladd($x, $y, $x) == muladd(SMatrix($x), $y, $x)
                @test_noalloc muladd($x, $y, $x)
            end
        end
    end

    @testset "UniformScaling" begin
        let a = SHermitianCompact(SVector{21, Int}(1 : 21))
            @test a + 3I == SMatrix(a) + 3I
            @test a + 3I isa typeof(a)

            @test a - 4I == SMatrix(a) - 4I
            @test a - 4I isa typeof(a)

            @test a * 3I === a * 3
            @test 3I * a === 3 * a
            @test 3I \ a == 3 \ a
            @test a / 3I == a / 3
        end
    end

    @testset "transpose/adjoint" begin
        a = Hermitian([[rand(Complex{Int}) for i = 1 : 2, j = 1 : 2] for row = 1 : 3, col = 1 : 3])
        @test transpose(SHermitianCompact{3}(a)) == transpose(a)
        @test adjoint(SHermitianCompact{3}(a)) == adjoint(a)

        b = Hermitian([rand(Complex{Int}) for i = 1 : 3, j = 1 : 3])
        @test adjoint(SHermitianCompact{3}(b)) == adjoint(b)
    end

    @testset "one/ones/zeros/fill" begin
        for N = 3 : 5, f in (:one, :ones, :zeros, :fill3)
            @eval begin
                @test $f(SHermitianCompact{$N, Int}) == $f(SMatrix{$N, $N, Int})
                @test $f(SHermitianCompact{$N, Int}) isa SHermitianCompact{$N, Int}
                @test_noalloc $f(SHermitianCompact{$N, Int})

                @test $f(SHermitianCompact{$N}) == $f(SMatrix{$N, $N})
                @test $f(SHermitianCompact{$N}) isa SHermitianCompact{$N, eltype($f(SMatrix{$N, $N}))}
                @test_noalloc $f(SHermitianCompact{$N})
            end
        end
    end

    @testset "rand" begin
        for N = 3 : 5, f in (:rand, :randn, :randexp)
            @eval begin
                @test_noalloc $f(SHermitianCompact{$N, Float32})
                @test $f(SHermitianCompact{$N, Float32}) isa SHermitianCompact{$N, Float32}

                @test_noalloc $f(SHermitianCompact{$N})
                @test $f(SHermitianCompact{$N}) isa SHermitianCompact{$N, Float64}
            end
        end
    end
end

end # module
