module SSymmetricCompactTest

using Compat
using StaticArrays
using Base.Test

macro test_noalloc(ex)
    esc(quote
        $ex
        @test(@allocated($ex) == 0)
    end)
end

function check_lower_triangle(a::SSymmetricCompact, triangle::SVector)
    N = size(a, 1)
    i = 0
    for col = 1 : N, row = 1 : N
        if row >= col
            i += 1
            @test a[row, col] == triangle[i]
        else
            @test a[row, col] == a[col, row]
        end
    end
end

function check_lower_triangle(a::SSymmetricCompact{3}, triangle::SVector)
    # because the generic check_lower_triangle uses almost the same code as what's being tested
    @test a[1, 1] == triangle[1]
    @test a[2, 1] == triangle[2]
    @test a[3, 1] == triangle[3]
    @test a[1, 2] == a[2, 1]
    @test a[2, 2] == triangle[4]
    @test a[3, 2] == triangle[5]
    @test a[1, 3] == a[3, 1]
    @test a[2, 3] == a[3, 2]
    @test a[3, 3] == triangle[6]
end

fill3(x) = fill(3, x)

@testset "SSymmetricCompact" begin
    @testset "Inner Constructor" begin
        for (N, L) in ((3, 6), (4, 10), (6, 21))
            for T in (Int32, Int64)
                @eval begin
                    let lowertriangle = rand(SVector{$L, Int32})
                        a = SSymmetricCompact{$N, $T, $L}(lowertriangle)
                        check_lower_triangle(a, lowertriangle)
                        @test a isa SSymmetricCompact{$N, $T, $L}
                        @test(@allocated(SSymmetricCompact{$N, $T, $L}(lowertriangle)) == 0)
                    end
                end
            end
        end
        @test_throws ArgumentError SSymmetricCompact{3, Int64, 7}(rand(SVector{7, Int32}))
    end

    @testset "Outer Constructors" begin
        for (N, L) in ((3, 6), (4, 10), (6, 21))
            for T in (Int32, Int64)
                @eval begin
                    let lowertriangle = rand(SVector{$L, Int32})
                        a1 = SSymmetricCompact{$N, $T}(lowertriangle)
                        check_lower_triangle(a1, $T.(lowertriangle))
                        @test a1 isa SSymmetricCompact{$N, $T, $L}
                        @test_noalloc SSymmetricCompact{$N, $T}(lowertriangle)

                        a2 = SSymmetricCompact{$N}(lowertriangle)
                        check_lower_triangle(a2, lowertriangle)
                        @test a2 isa SSymmetricCompact{$N, Int32, $L}
                        @test_noalloc SSymmetricCompact{$N}(lowertriangle)

                        a3 = SSymmetricCompact(lowertriangle)
                        check_lower_triangle(a3, lowertriangle)
                        @test a3 isa SSymmetricCompact{$N, Int32, $L}
                        @test_noalloc SSymmetricCompact(lowertriangle)

                        a4 = SSymmetricCompact{$N, $T}(a3)
                        @test a4 === a1
                        @test_noalloc SSymmetricCompact{$N, $T}(a3)
                    end
                    let a = rand(SMatrix{$N, $N, Int32})
                        @test SSymmetricCompact{$N, $T, $L}(a) isa SSymmetricCompact{$N, $T, $L}
                        @test_noalloc SSymmetricCompact{$N, $T, $L}(a)
                        @test SSymmetricCompact{$N, $T, $L}(a) == Symmetric($T.(a), :L)

                        @test SSymmetricCompact{$N, $T}(a) isa SSymmetricCompact{$N, $T, $L}
                        @test_noalloc SSymmetricCompact{$N, $T}(a)
                        @test SSymmetricCompact{$N, $T}(a) == Symmetric($T.(a), :L)

                        @test SSymmetricCompact{$N}(a) isa SSymmetricCompact{$N, Int32, $L}
                        @test_noalloc SSymmetricCompact{$N}(a)
                        @test SSymmetricCompact{$N}(a) == Symmetric(a, :L)

                        @test SSymmetricCompact(a) isa SSymmetricCompact{$N, Int32, $L}
                        @test_noalloc SSymmetricCompact(a)
                        @test SSymmetricCompact(a) == Symmetric(a, :L)
                    end
                end
            end
        end
    end

    @testset "convert" begin
        a = SSymmetricCompact(SMatrix{3, 3}(1 : 9))
        @test convert(typeof(a), a) === a
        @test convert(SSymmetricCompact{3, Float64}, a) isa SSymmetricCompact{3, Float64, 6}
        @test convert(SSymmetricCompact{3, Float64}, a) == Float64.(Array(a))
        @test convert(SSymmetricCompact{3, Float64, 6}, a) isa SSymmetricCompact{3, Float64, 6}
    end

    @testset "setindex" begin
        for (N, L) in ((3, 6), (4, 10), (6, 21))
            @eval begin
                let lowertriangle = zero(SVector{$L, Int32})
                    a = SSymmetricCompact(lowertriangle)
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
        a = rand(SSymmetricCompact{5, Float64})
        @test ishermitian(a)
        @test issymmetric(a)

        b = rand(SSymmetricCompact{5, Complex128})
        @test !ishermitian(b)
        @test issymmetric(b)

        c = b + conj(b)
        @test ishermitian(c)
        @test issymmetric(c)
        @test_noalloc ishermitian(c)
    end

    @testset "==" begin
        a = SSymmetricCompact(SVector{6}(-5 : 0))
        b = setindex(a, 5, 3)
        @test a == a
        @test !(a == b)
        @test_noalloc a == a
    end

    @testset "Arithmetic" begin
        a = SSymmetricCompact(SVector{6, Int}(1 : 6))
        b = SSymmetricCompact(SVector(-4, 5, 4, 8, -10, 11))
        c = SMatrix{3, 3}(-9 : -1)

        let a = a
            @test -a == -SMatrix(a)
            @test -a isa SSymmetricCompact{3, Int, 6}
            @test_noalloc -a
        end
        for (x, y) in ((a, b), (a, c), (c, a))
            @eval begin
                let x = $x, y = $y
                    @test x + y == SMatrix(x) + SMatrix(y)
                    @test_noalloc x + y

                    @test x - y == SMatrix(x) - SMatrix(y)
                    @test_noalloc x - y

                    if x isa SSymmetricCompact && y isa SSymmetricCompact
                        @test x + y isa SSymmetricCompact{3, Int, 6}
                        @test x - y isa SSymmetricCompact{3, Int, 6}
                    end
                end
            end
        end
    end

    @testset "Scalar-array" begin
        x = SSymmetricCompact(SVector{6, Int}(1 : 6))
        y = -5
        for op in (:-, :+, :*, :/, :\)
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
    end

    @testset "UniformScaling" begin
        let a = SSymmetricCompact(SVector{21, Int}(1 : 21))
            @test a + 3I == SMatrix(a) + 3I
            @test a + 3I isa typeof(a)
            @test_noalloc a + 3I

            @test a - 4I == SMatrix(a) - 4I
            @test a - 4I isa typeof(a)
            @test_noalloc a - 4I

            @test a * 3I === a * 3
            @test 3I * a === 3 * a
            @test 3I \ a == 3 \ a
            @test a / 3I == a / 3
        end
    end

    @testset "transpose/adjoint" begin
        a = Symmetric([[rand(Complex{Int}) for i = 1 : 2] for row = 1 : 3, col = 1 : 3])
        @test transpose(SSymmetricCompact{3}(a)) == transpose(a)
        # @test adjoint(SSymmetricCompact{3}(a)) == adjoint(a) # doesn't even work for a...

        b = Symmetric([rand(Complex{Int}) for i = 1 : 3, j = 1 : 3])
        @test adjoint(SSymmetricCompact{3}(b)) == adjoint(b)
    end

    @testset "one/eye/ones/zeros/fill" begin
        for N = 3 : 5, f in (:one, :eye, :ones, :zeros, :fill3)
            @eval begin
                @test $f(SSymmetricCompact{$N, Int}) == $f(SMatrix{$N, $N, Int})
                @test $f(SSymmetricCompact{$N, Int}) isa SSymmetricCompact{$N, Int}
                @test_noalloc $f(SSymmetricCompact{$N, Int})

                @test $f(SSymmetricCompact{$N}) == $f(SMatrix{$N, $N})
                @test $f(SSymmetricCompact{$N}) isa SSymmetricCompact{$N, eltype($f(SMatrix{$N, $N}))}
                @test_noalloc $f(SSymmetricCompact{$N})
            end
        end
    end

    @testset "rand" begin
        for N = 3 : 5, f in (:rand, :randn, :randexp)
            @eval begin
                @test_noalloc $f(SSymmetricCompact{$N, Float32})
                @test $f(SSymmetricCompact{$N, Float32}) isa SSymmetricCompact{$N, Float32}

                @test_noalloc $f(SSymmetricCompact{$N})
                @test $f(SSymmetricCompact{$N}) isa SSymmetricCompact{$N, Float64}
            end
        end
    end
end

end # module
