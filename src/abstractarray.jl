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


#=
# Methods necessary to fulfil the AbstractArray interface (plus some extensions, like size() on types)

Base.size{Sizes}(::StaticArray{Sizes}) = Sizes
Base.size{Sizes,T,N,D}(::Type{SArray{Sizes,T,N,D}}) = Sizes
Base.size{Sizes,T,N,D}(::Type{MArray{Sizes,T,N,D}}) = Sizes
Base.size{Sizes,T,N}(::StaticArray{Sizes,T,N}, i::Integer) = i <= N ? Sizes[i] : 1 # This form is *not* a compile-time constant
Base.size{Sizes,T,N,D}(::Type{SArray{Sizes,T,N,D}}, i::Integer) = i <= N ? Sizes[i] : 1 # This form is *not* a compile-time constant
Base.size{Sizes,T,N,D}(::Type{MArray{Sizes,T,N,D}}, i::Integer) = i <= N ? Sizes[i] : 1 # This form is *not* a compile-time constant
@generated function Base.size{I,Sizes,N}(::StaticArray{Sizes,N}, ::Type{Val{I}})
    I::Integer
    return (I <= N ? Sizes[I] : 1)
end
@generated function Base.size{I,Sizes,T,N,D}(::Type{SArray{Sizes,T,N,D}}, ::Type{Val{I}})
    I::Integer
    return (I <= N ? Sizes[I] : 1)
end
@generated function Base.size{I,Sizes,T,N,D}(::Type{MArray{Sizes,T,N,D}}, ::Type{Val{I}})
    I::Integer
    return (I <= N ? Sizes[I] : 1)
end

@generated Base.length{Sizes}(::StaticArray{Sizes}) = prod(Sizes)
@generated Base.length{Sizes,T,N,D}(::Type{SArray{Sizes,T,N,D}}) = prod(Sizes)
@generated Base.length{Sizes,T,N,D}(::Type{MArray{Sizes,T,N,D}}) = prod(Sizes)


# This seems to confuse Julia a bit in certain circumstances (specifically for trailing 1's)
function Base.isassigned(a::StaticArray, i::Int...)
    ii = sub2ind(size(a), i...)
    1 <= ii <= length(a) ? true : false
end


# Makes loops behave simplest, indexing itself is overridden
Base.linearindexing(::StaticArray) = Base.LinearFast()
Base.linearindexing{T<:StaticArray}(::Type{T}) = Base.LinearFast()


# TODO: decide whether this should fall back to MArray... at least it would "work"
Base.similar(::SArray, I...) = error("The similar() function is not defined for immutable SArrays. Use similar_type(), or an MArray, instead.")

Base.similar{Sizes,T}(a::MArray{Sizes,T}) = MArray{Sizes,T}()
Base.similar{Sizes,T}(a::MArray{Sizes,T}, I::Int...) = I == Sizes ? MArray{Sizes,T}() : error("similar(m_array, I...) cannot change size of MArray. Old size was $(size(a)) while new size was $I. Use similar(m_array, Val{Sizes}) instead.")
Base.similar{Sizes,T}(a::MArray{Sizes,T}, I::TupleN{Int}) = I == Sizes ? MArray{Sizes,T}() : error("similar(m_array, I::Tuple) cannot change size of MArray. Old size was $(size(a)) while new size was $I. Use similar(m_array, Val{Sizes}) instead.")
Base.similar{Sizes,T}(a::MArray{Sizes}, ::Type{T}) = MArray{Sizes,T}()
Base.similar{Sizes,T}(a::MArray{Sizes}, ::Type{T}, I::Int...) = I == Sizes ? MArray{Sizes,T}() : error("similar(m_array, ::Type{T}, I...) cannot change size of MArray. Old size was $(size(a)) while new size was $I. Use similar(m_array, Val{Sizes}) instead.")
Base.similar{Sizes,T}(a::MArray{Sizes}, ::Type{T}, I::TupleN{Int}) = I == Sizes ? MArray{Sizes,T}() : error("similar(m_array, ::Type{T}, I::Tuple) cannot change size of MArray. Old size was $(size(a)) while new size was $I. Use similar(m_array, Val{Sizes}) instead.")
@generated function Base.similar{Sizes,T,NewSizes}(a::MArray{Sizes,T}, ::Type{Val{NewSizes}})
    NewN = length(NewSizes)
    NewM = prod(NewSizes)

    return :($(MArray{NewSizes,T,NewN,NTuple{NewM,T}})())
end
@generated function Base.similar{Sizes,T,NewT,NewSizes}(a::MArray{Sizes,T}, ::Type{NewT}, ::Type{Val{NewSizes}})
    NewN = length(NewSizes)
    NewM = prod(NewSizes)

    return :($(MArray{NewSizes,NewT,NewN,NTuple{NewM,NewT}})())
end


"""
    similar_type(staticarray, [element_type], [Val{dims}])

Create a static array type with the same (or optionally modified) element type and size.
"""
similar_type(A::StaticArray) = error("Similar type not defined for $(typeof(A))")
similar_type{SA<:StaticArray}(::Type{SA}) = error("Similar type not defined for $(typeof(A))")
similar_type{Sizes,T,N,D}(::Union{SArray{Sizes,T,N,D},Type{SArray{Sizes,T,N,D}}}) = SArray{Sizes,T,N,D}
@generated function similar_type{Sizes,T,N,D,NewSizes}(::Union{SArray{Sizes,T,N,D},Type{SArray{Sizes,T,N,D}}}, ::Type{Val{NewSizes}})
    if isa(NewSizes, TupleN{Int})
        NewN = length(NewSizes)
        numel = prod(NewSizes)
        out = SArray{NewSizes,T,NewN,NTuple{numel,T}}
        return :($out)
    else
        str = "Sizes must be a tuple of integers, got $NewSizes"
        return :(error($str))
    end
end
@generated function similar_type{Sizes,T,N,D,NewT}(::Union{SArray{Sizes,T,N,D},Type{SArray{Sizes,T,N,D}}}, ::Type{NewT})
    M = prod(Sizes)

    return :(SArray{Sizes,NewT,$N,NTuple{$M,$NewT}})
end
@generated function similar_type{Sizes,T,N,D,NewT,NewSizes}(::Union{SArray{Sizes,T,N,D},Type{SArray{Sizes,T,N,D}}}, ::Type{NewT}, ::Type{Val{NewSizes}})
    NewN = length(NewSizes)
    NewM = prod(NewSizes)

    return :($(SArray{NewSizes,NewT,NewN,NTuple{NewM,NewT}}))
end

similar_type{Sizes,T,N,D}(::Union{MArray{Sizes,T,N,D},Type{MArray{Sizes,T,N,D}}}) = MArray{Sizes,T,N,D}
@generated function similar_type{Sizes,T,N,D,NewSizes}(::Union{MArray{Sizes,T,N,D},Type{MArray{Sizes,T,N,D}}}, ::Type{Val{NewSizes}})
    if isa(NewSizes, TupleN{Int})
        NewN = length(NewSizes)
        numel = prod(NewSizes)
        out = MArray{NewSizes,T,NewN,NTuple{numel,T}}
        return :($out)
    else
        str = "Sizes must be a tuple of integers, got $NewSizes"
        return :(error($str))
    end
end
@generated function similar_type{Sizes,T,N,D,NewT}(::Union{MArray{Sizes,T,N,D},Type{MArray{Sizes,T,N,D}}}, ::Type{NewT})
    M = prod(Sizes)

    return :(MArray{Sizes,NewT,$N,NTuple{$M,$NewT}})
end
@generated function similar_type{Sizes,T,N,D,NewT,NewSizes}(::Union{MArray{Sizes,T,N,D},Type{MArray{Sizes,T,N,D}}}, ::Type{NewT}, ::Type{Val{NewSizes}})
    NewN = length(NewSizes)
    NewM = prod(NewSizes)

    return :($(MArray{NewSizes,NewT,NewN,NTuple{NewM,NewT}}))
end

# reshape()
Base.reshape(::StaticArray, ::TupleN{Int}) = error("Need reshape size as a type-paramter. Use reshape(staticarray,Val{(s₁,...)}).")
@generated function Base.reshape{Sizes,T,N,D,NewSizes}(a::SArray{Sizes,T,N,D},::Type{Val{NewSizes}})
    if length(a) != prod(NewSizes)
        str = "new dimensions $NewSizes must be consistent with array length $(length(a))"
        return :(throw(DimensionMismatch($str)))
    end

    NewN = length(NewSizes)
    :($(SArray{NewSizes,T,NewN,D})(a.data))
end
@generated function Base.reshape{Sizes,T,N,D,NewSizes}(a::MArray{Sizes,T,N,D},::Type{Val{NewSizes}})
    if length(a) != prod(NewSizes)
        str = "new dimensions $NewSizes must be consistent with array length $(length(a))"
        return :(throw(DimensionMismatch($str)))
    end

    NewN = length(NewSizes)
    :($(MArray{NewSizes,T,NewN,D})(a.data))
end

# TODO: permutedims() (and transpose/ctranspose in linalg)
=#
