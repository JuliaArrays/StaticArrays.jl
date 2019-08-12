using StaticArrays, Test, Random
@testset "SBitSet" begin 
    m = MBitSet{4}([10, 44, 62, 144, 159, 190, 226])
    @test length(m)==7
    s = SBitSet((0x2000080000000200, 0x0000000000000000, 0x2000000040008000, 0x0000000200000000))
    @test s==m
    m[17] = true
    m[160] = true
    @test collect(m) == [10, 17, 44, 62, 144, 159, 160, 190, 226]


    @test issubset(m & SBitSet{4}(12:159), m)
    m .= m & SBitSet{4}(12:159)
    @test m == MBitSet{4}([17, 44, 62, 144, 159])

    @test m | SBitSet{4}(189, 11) == SBitSet{4}([11, 17, 44, 62, 144, 159, 189])
    @test xor(SBitSet(s), SBitSet{4}(44, 43)) == SBitSet{4}([10, 43, 62, 144, 159, 190, 226])

    @test m !== MBitSet{4}([17, 44, 62, 144, 159])


    @test !(-4 in m)
    @test !(257 in m)
    @test_throws BoundsError m[257]
    @test_throws BoundsError m[0]
    @test_throws BoundsError m[-1]

    r=rand(SBitSet{3}, 10_000);
    r.&= rand(SBitSet{3}, 10_000);
    r.&= rand(SBitSet{3}, 10_000);
    @test sortperm(r)==sortperm(collect.(r))

    s1 = SBitSet((0xcc4e1835c795f9a3, 0x30f595715840b150))
    s2 = SBitSet((0x155fa2af6ce14a59, 0x0b03b4f0509cc36c))
    s3 = SBitSet((0xbdcdfbf000fc7043, 0x650815944c280196))

    m = MBitSet(s1)
    mr1 = reduce(intersect, [s1,s2,s3])
    mr2 = reduce(&, [s1, s2, s3])
    intersect!(m, s2, s3)
    @test mr1 == mr2 == m
    @test m == MBitSet{2}(shuffle([1, 15, 24, 38, 51, 52, 55, 59, 73, 95, 101, 107, 109]))

    m = MBitSet(s1)
    mr1 = reduce(setdiff, [s1,s2,s3])
    mr2 = s1 & ~s2 & ~s3
    setdiff!(m, s2, s3)
    @test mr1 == mr2 == m
    @test m ==  SBitSet{2}(shuffle([6, 8, 9, 16, 25, 26, 32, 63, 77, 78, 87, 97, 115, 117, 118, 119, 120, 125]))

    m = MBitSet(s1&s2)
    mr1 = reduce(|, [m, SBitSet{2}(1), SBitSet{2}(128)])
    mr2 = m | SBitSet{2}(1) | SBitSet{2}(128)
    union!(m, SBitSet{2}(1), SBitSet{2}(128))
    @test mr1 == mr2 == m
    @test m == MBitSet{2}(shuffle([1, 12, 15, 17, 24, 27, 31, 33, 35, 38, 50, 51, 52, 55, 59, 71, 73, 80, 93, 95,
        101, 102, 103, 107, 109, 112, 113, 128]))
    
    m = MBitSet(s1)
    mr1 = reduce(xor, [s1, s2, s3])
    mr2 = xor(xor(s1, s2), s3)
    symdiff!(m, s2, s3)
    @test m == mr1 == mr2
    @test m == SBitSet((0x64dc416aab88c3b9, 0x5efe341544f473aa))

    m[] = s2
    @test m == s2
    @test m[] === s2


    d = Dict()
    d[SBitSet{1}(1,5,63)]=1
    d[SBitSet{1}(9,62, 45)]=2
    d[SBitSet{1}(1,5,63)]=4;
    @test d == Dict{SBitSet{1},Int64}(
  SBitSet{1}([9, 45, 62]) => 2,
  SBitSet{1}([1, 5, 63])  => 4)

  r=[]
  @simd for i in s1
    push!(r, i)
  end
  @test r==collect(s1)

  s1ba = convert(BitVector, s1)
  @test collect(s1) == [i for i=1:length(s1ba) if s1ba[i]]
  @test convert(SBitSet{2}, s1ba) == s1

end