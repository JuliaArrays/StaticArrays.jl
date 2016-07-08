immutable SArray{Size, T, N, L} <: StaticArray{T, N}
    data::NTuple{T,L}

    function SArray(x::NTuple{L})
        check_sarray_parameters(Val{Size}, T, Val{N}, Val{L})
        new(x)
    end
end

@generated function check_sarray_parameters{Size,T,N,L}(::Type{Val{Size}}, ::Type{T}, ::Type{Val{N}}, ::Type{Val{L}})
    if !(isa(Size, Tuple{Vararg, Int}))
        error("SArray parameter Size must be a tuple of Ints (e.g. `SArray{(3,3)}`)")
    end

    if L != prod(Size) || L < 0 || minimum(Size) < 0 || length(Size) != N
        error("Size mismatch")
    end

    return nothing
end

# TODO define convenience constructors


#####################
## SMatrix methods ##
#####################

@pure size{Size}(::Union{SMatrix{Size},Type{SMatrix{Size}}}) = Size
@pure size{Size,T}(::Type{SMatrix{Size,T}}) = Size
@pure size{Size,T,N}(::Type{SMatrix{Size,T,N}}) = Size
@pure size{Size,T,N,L}(::Type{SMatrix{Size,T,N,L}}) = Size

function getindex(v::SMatrix, i::Integer)
    Base.@_inline_meta
    v.data[i]
end

@inline Tuple(v::SMatrix) = v.data
