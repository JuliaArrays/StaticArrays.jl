
@inline function read(io::IO, ::Type{SA}) where {SA<:StaticArray}
    elements = Ref{NTuple{length(SA),eltype(SA)}}()
    read!(io, elements)
    SA(elements[])
end

@inline function read!(io::IO, a::SA) where {SA<:StaticArray}
    unsafe_read(io, Base.unsafe_convert(Ptr{eltype(SA)}, a), sizeof(a))
    a
end

@inline function write(io::IO, a::SA) where {SA<:StaticArray}
    write(io, Ref(Tuple(a)))
end

