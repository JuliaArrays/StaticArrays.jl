@pure length{T<:StaticArray}(a::Union{T,Type{T}}) = prod(size(a))

@pure function size{T<:StaticArray}(a::Union{T,Type{T}}, d::Integer)
    s = size(a)
    return (d <= length(s) ? s[d] : 1)
end

# This seems to confuse Julia a bit in certain circumstances (specifically for trailing 1's)
function Base.isassigned(a::StaticArray, i::Int...)
    ii = sub2ind(size(a), i...)
    1 <= ii <= length(a) ? true : false
end

Base.linearindexing{T<:StaticArray}(::Union{T,Type{T}}) = Base.LinearFast()

# Default type search for similar_type
@pure similar_type{SA<:StaticArray}(::Union{SA,Type{SA}}) = SA
@pure function similar_type{SA<:StaticArray,N,T}(::Union{SA,Type{SA}}, ::Type{T}, sizes::NTuple{N,Int})
    similar_type(similar_type(SA, T), sizes)
end
@pure function similar_type{SA<:StaticArray,T}(::Union{SA,Type{SA}}, ::Type{T}, size::Int)
    similar_type(similar_type(SA, T), size)
end
@generated function similar_type{SA<:StaticArray,T}(::Union{SA,Type{SA}}, ::Type{T})
    # This function has a strange error (on tests) regarding double-inference, if it is marked @pure
    if T == eltype(SA)
        return SA
    end

    primary_type = (SA.name.primary)
    eltype_param_number = super_eltype_param(primary_type)
    if isnull(eltype_param_number)
        if ndims(SA) == 1
            return SVector{length(SA), T}
        elseif ndims(SA) == 2
            sizes = size(SA)
            return SMatrix{sizes[1], sizes[2], T, prod(sizes)}
        else
            sizes = size(SA)
            return SArray{sizes, T, length(sizes), prod(sizes)}
        end
    end

    T_out = primary_type
    for i = 1:length(T_out.parameters)
        if i == get(eltype_param_number)
            T_out = T_out{T}
        else
            T_out = T_out{SA.parameters[i]}
        end
    end
    return T_out
end

@pure function super_eltype_param(T)
    T_super = supertype(T)
    if T_super == Any
        error("Unknown error")
    end

    if T_super.name.name == :StaticArray
        if isa(T_super.parameters[1], TypeVar)
            tv = T_super.parameters[1]
            for i = 1:length(T.parameters)
                if tv == T.parameters[i]
                    return Nullable{Int}(i)
                end
            end
            error("Unknown error")
        end
        return Nullable{Int}()
    else
        param_number = super_eltype_param(T_super)
        if isnull(param_number)
            return Nullable{Int}()
        else
            tv = T_super.parameters[get(param_number)]
            for i = 1:length(T.parameters)
                if tv == T.parameters[i]
                    return Nullable{Int}(i)
                end
            end
            error("Unknown error")
        end
    end
end

# Some fallbacks

@pure similar_type{SA<:StaticArray}(::Union{SA,Type{SA}}, size::Int) = SVector{size, eltype(SA)}
@pure similar_type{SA<:StaticArray}(::Union{SA,Type{SA}}, sizes::Tuple{Int}) = SVector{sizes[1], eltype(SA)}

@pure similar_type{SA<:StaticArray}(::Union{SA,Type{SA}}, sizes::Tuple{Int,Int}) = SMatrix{sizes[1], sizes[2], eltype(SA), sizes[1]*sizes[2]}

@pure similar_type{SA<:StaticArray,N}(::Union{SA,Type{SA}}, sizes::Tuple{Vararg{Int,N}}) = SArray{sizes, eltype(SA), N, prod(sizes)}

# Some specializations for the mutable case
@pure similar_type{MA<:Union{MVector,MMatrix,MArray}}(::Union{MA,Type{MA}}, size::Int) = MVector{size, eltype(MA)}
@pure similar_type{MA<:Union{MVector,MMatrix,MArray}}(::Union{MA,Type{MA}}, sizes::Tuple{Int}) = MVector{sizes[1], eltype(MA)}

@pure similar_type{MA<:Union{MVector,MMatrix,MArray}}(::Union{MA,Type{MA}}, sizes::Tuple{Int,Int}) = MMatrix{sizes[1], sizes[2], eltype(MA), sizes[1]*sizes[2]}

@pure similar_type{MA<:Union{MVector,MMatrix,MArray},N}(::Union{MA,Type{MA}}, sizes::Tuple{Vararg{Int,N}}) = MArray{sizes, eltype(MA), N, prod(sizes)}

# And also similar() returning mutable StaticArrays
@inline similar{SV <: StaticVector}(::SV) = MVector{length(SV),eltype(SV)}()
@inline similar{SV <: StaticVector, T}(::SV, ::Type{T}) = MVector{length(SV),T}()
@inline similar{SA <: StaticArray}(::SA, sizes::Tuple{Int}) = MVector{sizes[1], eltype(SA)}()
@inline similar{SA <: StaticArray}(::SA, size::Int) = MVector{size, eltype(SA)}()
@inline similar{T}(::StaticArray, ::Type{T}, sizes::Tuple{Int}) = MVector{sizes[1],T}()
@inline similar{T}(::StaticArray, ::Type{T}, size::Int) = MVector{size,T}()

@inline similar{SM <: StaticMatrix}(m::SM) = MMatrix{size(SM,1),size(SM,2),eltype(SM),length(SM)}()
@inline similar{SM <: StaticMatrix, T}(::SM, ::Type{T}) = MMatrix{size(SM,1),size(SM,2),T,length(SM)}()
@inline similar{SA <: StaticArray}(::SA, sizes::Tuple{Int,Int}) = MMatrix{sizes[1], sizes[2], eltype(SA), sizes[1]*sizes[2]}()
@inline similar(a::StaticArray, T::Type, sizes::Tuple{Int,Int}) = MMatrix{sizes[1], sizes[2], T, sizes[1]*sizes[2]}()

@inline similar{SA <: StaticArray}(m::SA) = MArray{size(SA),eltype(SA),ndims(SA),length(SA)}()
@inline similar{SA <: StaticArray,T}(m::SA, ::Type{T}) = MArray{size(SA),T,ndims(SA),length(SA)}()
@inline similar{SA <: StaticArray,N}(m::SA, sizes::NTuple{N, Int}) = MArray{sizes,eltype(SA),N,prod(sizes)}()
@inline similar{SA <: StaticArray,N,T}(m::SA, ::Type{T}, sizes::NTuple{N, Int}) = MArray{sizes,T,N,prod(sizes)}()

# This is used in Base.LinAlg quite a lot, and it impacts type stability
# since some functions like expm() branch on a check for Hermitian or Symmetric
# TODO much more work on type stability. Base functions are using similar() with
# size, which poses difficulties!!
@generated function Base.full{T,SM<:StaticMatrix}(sym::Symmetric{T,SM})
    exprs_up = [i <= j ? :(m[$(sub2ind(size(SM), i, j))]) : :(m[$(sub2ind(size(SM), j, i))]) for i = 1:size(SM,1), j=1:size(SM,2)]
    exprs_lo = [i >= j ? :(m[$(sub2ind(size(SM), i, j))]) : :(m[$(sub2ind(size(SM), j, i))]) for i = 1:size(SM,1), j=1:size(SM,2)]

    return quote
        $(Expr(:meta, :inline))
        m = sym.data
        if sym.uplo == 'U'
            @inbounds return SM($(Expr(:call, SM, Expr(:tuple, exprs_up...))))
        else
            @inbounds return SM($(Expr(:call, SM, Expr(:tuple, exprs_lo...))))
        end
    end
end

@generated function Base.full{T,SM<:StaticMatrix}(sym::Hermitian{T,SM})
    exprs_up = [i <= j ? :(m[$(sub2ind(size(SM), i, j))]) : :(conj(m[$(sub2ind(size(SM), j, i))])) for i = 1:size(SM,1), j=1:size(SM,2)]
    exprs_lo = [i >= j ? :(m[$(sub2ind(size(SM), i, j))]) : :(conj(m[$(sub2ind(size(SM), j, i))])) for i = 1:size(SM,1), j=1:size(SM,2)]

    return quote
        $(Expr(:meta, :inline))
        m = sym.data
        if sym.uplo == 'U'
            @inbounds return SM($(Expr(:call, SM, Expr(:tuple, exprs_up...))))
        else
            @inbounds return SM($(Expr(:call, SM, Expr(:tuple, exprs_lo...))))
        end
    end
end

# Reshape uses types to specify size (and also conveniently, the output type)
@generated function reshape{SA<:StaticArray}(a::StaticArray, ::Type{SA})
    if !(SA <: StaticVector) && length(a) != length(SA)
        error("Static array of size $(size(a)) cannot be reshaped to size $(size(SA))")
    end

    return quote
        $(Expr(:meta, :inline))
        return SA(tuple(a))
    end
end

function reshape{SA<:StaticArray}(a::AbstractArray, ::Type{SA})
    if !(SA <: StaticVector) && length(a) != length(SA)
        error("Static array of size $(size(a)) cannot be reshaped to size $(size(SA))")
    end

    return SA((a...))
end
