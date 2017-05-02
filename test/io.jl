# Serialize `xs` as type `T` to an IOBuffer one by one using Base.write.
# Return the buffer positioned at the start, ready for reading
write_buf{T}(::Type{T}, xs...) = write_buf(map(T, xs)...)

function write_buf(xs...)
    io = IOBuffer()
    foreach(x->write(io, x), xs)
    seek(io, 0)
    io
end

@testset "Binary IO" begin
    @testset "read" begin
        # Read static arrays from a stream which was serialized elementwise
        @test read(write_buf(UInt8, 1,2,3), SVector{3,UInt8})         === SVector{3,UInt8}(1,2,3)
        @test read(write_buf(Int32, -1,2,3), SVector{3,Int32})        === SVector{3,Int32}(-1,2,3)
        @test read(write_buf(Float64, 1,2,3), SVector{3,Float64})     === SVector{3,Float64}(1,2,3)
        @test read(write_buf(Float64, 1,2,3,4), SMatrix{2,2,Float64}) === @SMatrix [1.0 3.0; 2.0 4.0]
    end

    @testset "write" begin
        # Compare serialized bytes
        @test take!(write_buf(UInt8, 1,2,3))     == take!(write_buf(SVector{3,UInt8}(1,2,3)))
        @test take!(write_buf(Int32, -1,2,3))    == take!(write_buf(SVector{3,Int32}(-1,2,3)))
        @test take!(write_buf(Float64, 1,2,3))   == take!(write_buf(SVector{3,Float64}(1,2,3)))
        @test take!(write_buf(Float64, 1,2,3,4)) == take!(write_buf(@SMatrix [1.0 3.0; 2.0 4.0]))
    end

    @testset "read!" begin
        # Read static arrays from a stream which was serialized elementwise
        @test read!(write_buf(UInt8, 1,2,3),     zeros(MVector{3,UInt8}))     == MVector{3,UInt8}(1,2,3)
        @test read!(write_buf(Int32, -1,2,3),    zeros(MVector{3,Int32}))     == MVector{3,Int32}(-1,2,3)
        @test read!(write_buf(Float64, 1,2,3),   zeros(MVector{3,Float64}))   == MVector{3,Float64}(1,2,3)
        @test read!(write_buf(Float64, 1,2,3,4), zeros(MMatrix{2,2,Float64})) == @MMatrix [1.0 3.0; 2.0 4.0]
    end
end

