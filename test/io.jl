using StaticArrays, Test
# Serialize `xs` as type `T` to an IOBuffer one by one using Base.write.
# Return the buffer positioned at the start, ready for reading
write_buf(::Type{T}, xs...) where {T} = write_buf(map(T, xs)...)

function write_buf(xs...)
    io = IOBuffer()
    foreach(x->write(io, x), xs)
    seek(io, 0)
    io
end

@testset "Binary IO" begin
    for (T, data,Shape) in [
                   (UInt8, (1,2,3), Tuple{3}),
                   (Int32, (-1,2,3), Tuple{3}),
                   (Float64, (1,2,3,4), Tuple{2,2})]
        # Read static arrays from a stream which was serialized elementwise
        io = write_buf(T, data...)
        @test read(io, SArray{Shape,T}) === SArray{Shape,T}(data...)

        io = write_buf(T, data...)
        out = MArray{Shape,T}(zeros(length(data))...)
        expected = MArray{Shape,T}(data...)
        @test read!(io, out) == expected
        @test out == expected
        @test typeof(out) == typeof(expected)

        # Compare serialized bytes
        io = write_buf(T, data...)
        arr = SArray{Shape, T}(data...)
        @test take!(io) == take!(write_buf(arr))
    end
end
