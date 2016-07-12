immutable SArray{Size, T, N, L} <: StaticArray{T, N}
    data::NTuple{L,T}

    function SArray(x)
        check_sarray_parameters(Val{Size}, T, Val{N}, Val{L})
        new(x)
    end
end

@generated function check_sarray_parameters{Size,T,N,L}(::Type{Val{Size}}, ::Type{T}, ::Type{Val{N}}, ::Type{Val{L}})
    if !(isa(Size, Tuple{Vararg{Int}}))
        error("SArray parameter Size must be a tuple of Ints (e.g. `SArray{(3,3)}`)")
    end

    if L != prod(Size) || L < 0 || minimum(Size) < 0 || length(Size) != N
        error("Size mismatch")
    end

    return nothing
end

@generated function (::Type{SArray{Size,T,N}}){Size,T,N}(x)
    return quote
        $(Expr(:meta, :inline))
        SArray{Size,T,N,$(prod(Size))}(x)
    end
end

@generated function (::Type{SArray{Size,T}}){Size,T}(x)
    return quote
        $(Expr(:meta, :inline))
        SArray{Size,T,$(length(Size)),$(prod(Size))}(x)
    end
end

@generated function (::Type{SArray{Size}}){Size}(x)
    return quote
        $(Expr(:meta, :inline))
        SArray{Size,$(promote_tuple_eltype(x)),$(length(Size)),$(prod(Size))}(x)
    end
end

####################
## SArray methods ##
####################

@pure size{Size}(::Union{SArray{Size},Type{SArray{Size}}}) = Size
@pure size{Size,T}(::Type{SArray{Size,T}}) = Size
@pure size{Size,T,N}(::Type{SArray{Size,T,N}}) = Size
@pure size{Size,T,N,L}(::Type{SArray{Size,T,N,L}}) = Size

function getindex(v::SArray, i::Integer)
    Base.@_inline_meta
    v.data[i]
end

@inline Tuple(v::SArray) = v.data
