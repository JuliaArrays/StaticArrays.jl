typealias StaticScalar{T} StaticArray{T,0}

@pure length{T<:StaticArray}(a::Union{T,Type{T}}) = prod(size(a))
@pure length{T<:StaticScalar}(a::Union{T,Type{T}}) = 1

@pure function size{T<:StaticArray}(a::Union{T,Type{T}}, d::Integer)
    s = size(a)
    return (d <= length(s) ? s[d] : 1)
end

# This seems to confuse Julia a bit in certain circumstances (specifically for trailing 1's)
@inline function Base.isassigned(a::StaticArray, i::Int...)
    ii = sub2ind(size(a), i...)
    1 <= ii <= length(a) ? true : false
end

Base.linearindexing{T<:StaticArray}(::Union{T,Type{T}}) = Base.LinearFast()

# Default type search for similar_type
"""
    similar_type(static_array)
    similar_type(static_array, T)
    similar_type(static_array, Size)
    similar_type(static_array, T, Size)

Returns a constructor for a statically-sized array similar to the input array
(or type) `static_array`, optionally with different element type `T` or size
`Size`.

This differs from `similar()` in that the resulting array type may not be
mutable (or define `setindex()`)  and therefore the returned type may need to
be *constructed* with its data.

Note that `Size` will need to be a compile-time constant in order for the result
to be inferrable by the compiler.
"""
@pure similar_type{SA<:StaticArray}(::Union{SA,Type{SA}}) = SA
@pure function similar_type{SA<:StaticArray,N,T}(::Union{SA,Type{SA}}, ::Type{T}, sizes::NTuple{N,Int})
    similar_type(similar_type(SA, T), sizes)
end
@pure function similar_type{SA<:StaticArray,T}(::Union{SA,Type{SA}}, ::Type{T}, size::Int)
    similar_type(similar_type(SA, T), size)
end
@pure function similar_type{SA<:StaticArray,T,S}(::Union{SA,Type{SA}}, ::Type{T}, size::Size{S})
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

@pure similar_type{SA<:StaticArray}(::Union{SA,Type{SA}}, size::Tuple{}) = Scalar{eltype(SA)} # No mutable fallback here...
@pure similar_type{SA<:StaticArray}(::Union{SA,Type{SA}}, size::Int) = SVector{size, eltype(SA)}
@pure similar_type{SA<:StaticArray}(::Union{SA,Type{SA}}, sizes::Tuple{Int}) = SVector{sizes[1], eltype(SA)}

@pure similar_type{SA<:StaticArray}(::Union{SA,Type{SA}}, sizes::Tuple{Int,Int}) = SMatrix{sizes[1], sizes[2], eltype(SA), sizes[1]*sizes[2]}

@pure similar_type{SA<:StaticArray,N}(::Union{SA,Type{SA}}, sizes::Tuple{Vararg{Int,N}}) = SArray{sizes, eltype(SA), N, prod(sizes)}

@generated function similar_type{SA <: StaticArray,S}(::Union{SA,Type{SA}}, ::Size{S})
    if length(S) == 1
        return quote
            $(Expr(:meta, :inline))
            SVector{$(S[1]), $(eltype(SA))}
        end
    elseif length(S) == 2
        return quote
            $(Expr(:meta, :inline))
            SMatrix{$(S[1]), $(S[2]), $(eltype(SA))}
        end
    else
        return quote
            $(Expr(:meta, :inline))
            SArray{S, $(eltype(SA)), $(length(S)), $(prod(S))}
        end
    end
end

# Some specializations for the mutable case
@pure similar_type{MA<:Union{MVector,MMatrix,MArray,SizedArray}}(::Union{MA,Type{MA}}, size::Int) = MVector{size, eltype(MA)}
@pure similar_type{MA<:Union{MVector,MMatrix,MArray,SizedArray}}(::Union{MA,Type{MA}}, sizes::Tuple{Int}) = MVector{sizes[1], eltype(MA)}

@pure similar_type{MA<:Union{MVector,MMatrix,MArray,SizedArray}}(::Union{MA,Type{MA}}, sizes::Tuple{Int,Int}) = MMatrix{sizes[1], sizes[2], eltype(MA), sizes[1]*sizes[2]}

@pure similar_type{MA<:Union{MVector,MMatrix,MArray,SizedArray},N}(::Union{MA,Type{MA}}, sizes::Tuple{Vararg{Int,N}}) = MArray{sizes, eltype(MA), N, prod(sizes)}

@generated function similar_type{MA<:Union{MVector,MMatrix,MArray,SizedArray},S}(::Union{MA,Type{MA}}, ::Size{S})
    if length(S) == 1
        return quote
            $(Expr(:meta, :inline))
            MVector{$(S[1]), $(eltype(MA))}
        end
    elseif length(S) == 2
        return quote
            $(Expr(:meta, :inline))
            MMatrix{$(S[1]), $(S[2]), $(eltype(MA))}
        end
    else
        return quote
            $(Expr(:meta, :inline))
            MArray{S, $(eltype(MA)), $(length(S)), $(prod(S))}
        end
    end
end

# And also similar() returning mutable StaticArrays
@inline similar{SV <: StaticVector}(::SV) = MVector{length(SV),eltype(SV)}()
@inline similar{SV <: StaticVector, T}(::SV, ::Type{T}) = MVector{length(SV),T}()

@inline similar{SM <: StaticMatrix}(m::SM) = MMatrix{size(SM,1),size(SM,2),eltype(SM),length(SM)}()
@inline similar{SM <: StaticMatrix, T}(::SM, ::Type{T}) = MMatrix{size(SM,1),size(SM,2),T,length(SM)}()

@inline similar{SA <: StaticArray}(m::SA) = MArray{size(SA),eltype(SA),ndims(SA),length(SA)}()
@inline similar{SA <: StaticArray,T}(m::SA, ::Type{T}) = MArray{size(SA),T,ndims(SA),length(SA)}()

@generated function similar{SA <: StaticArray,S}(::SA, ::Size{S})
    if length(S) == 1
        return quote
            $(Expr(:meta, :inline))
            MVector{$(S[1]), $(eltype(SA))}()
        end
    elseif length(S) == 2
        return quote
            $(Expr(:meta, :inline))
            MMatrix{$(S[1]), $(S[2]), $(eltype(SA))}()
        end
    else
        return quote
            $(Expr(:meta, :inline))
            MArray{S, $(eltype(SA))}()
        end
    end
end

@generated function similar{SA <: StaticArray, T, S}(::SA, ::Type{T}, ::Size{S})
    if length(S) == 1
        return quote
            $(Expr(:meta, :inline))
            MVector{$(S[1]), T}()
        end
    elseif length(S) == 2
        return quote
            $(Expr(:meta, :inline))
            MMatrix{$(S[1]), $(S[2]), T}()
        end
    else
        return quote
            $(Expr(:meta, :inline))
            MArray{S, T}()
        end
    end
end


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

# Reshape used types to specify size (and also conveniently, the output type)
@generated function reshape{SA<:StaticArray}(a::StaticArray, ::Type{SA})
    if !(SA <: StaticVector) && length(a) != length(SA)
        error("Static array of size $(size(a)) cannot be reshaped to size $(size(SA))")
    end

    Base.depwarn("Use reshape(array, Size(dims...)) rather than reshape(array, StaticArrayType)", :reshape)

    return quote
        $(Expr(:meta, :inline))
        return SA(Tuple(a))
    end
end

function reshape{SA<:StaticArray}(a::AbstractArray, ::Type{SA})
    if !(SA <: StaticVector) && length(a) != length(SA)
        error("Static array of size $(size(a)) cannot be reshaped to size $(size(SA))")
    end

    Base.depwarn("Use reshape(array, Size(dims...)) rather than reshape(array, StaticArrayType)", :reshape)

    return SA((a...))
end


# Versions using Size{}
@generated function reshape{S}(a::StaticArray, ::Size{S})
    if length(a) != prod(S)
        error("Static array of size $(size(a)) cannot be reshaped to size $S")
    end

    newtype = similar_type(a, S)

    return quote
        $(Expr(:meta, :inline))
        return $newtype(a)
    end
end

@generated function reshape{S}(a::Array, ::Size{S})
    newtype = SizedArray{S, eltype(a), length(S)}

    return quote
        $(Expr(:meta, :inline))
        return $newtype(a)
    end
end
