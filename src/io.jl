
@inline function read{SA<:StaticArray}(io::IO, ::Type{SA})
    # Copy Base implementation of `read` for primitive types.  This is less
    # efficient in 0.6 that we'd like because creating the Ref allocates.
    elements = Ref{NTuple{length(SA),eltype(SA)}}()
    read(io, elements)
    SA(elements[])
end

@inline function read!{SA<:StaticArray}(io::IO, a::SA)
    unsafe_read(io, Base.unsafe_convert(Ptr{eltype(SA)}, a), sizeof(a))
    a
end

@inline function write{SA<:StaticArray}(io::IO, a::SA)
    write(io, Ref(Tuple(a)))
end

