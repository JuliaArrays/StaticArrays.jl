@inline function _read_data(io::IO, ::Type{SA}) where {SA <: StaticArray}
    data = Ref{NTuple{length(SA),eltype(SA)}}()
    read!(io, data)
    data[]
end

@inline function read(io::IO, ::Type{SA}) where {SA<:StaticArray}
    data = _read_data(io, SA)
    SA(data)
end

@inline function read!(io::IO, a::SA) where {SA<:StaticArray}
    data = _read_data(io, SA)
    a.data = data
    a
end

@deprecate read!(io::IO, SA::Type{<:StaticArray}) read(io, SA)

@inline function write(io::IO, a::SA) where {SA<:StaticArray}
    write(io, Ref(Tuple(a)))
end
