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

# Some fallbacks
@pure similar_type{SA<:StaticArray}(::Union{SA,Type{SA}}) = SA

@pure similar_type{SV<:StaticVector, T}(::Union{SV,Type{SV}}, ::Type{T}) = SVector{length(SV),T}
@pure similar_type{SA<:StaticArray}(::Union{SA,Type{SA}}, size::Integer) = SVector{size, eltype(SA)}
@pure similar_type{SA<:StaticArray}(::Union{SA,Type{SA}}, sizes::Tuple{Integer}) = SVector{sizes[1], eltype(SA)}
@pure similar_type{SA<:StaticArray,T}(::Union{SA,Type{SA}}, ::Type{T}, size::Integer) = SVector{size, T}
@pure similar_type{SA<:StaticArray,T}(::Union{SA,Type{SA}}, ::Type{T}, sizes::Tuple{Integer}) = SVector{sizes[1], T}

@pure similar_type{SM<:StaticMatrix, T}(::Union{SM,Type{SM}}, ::Type{T}) = SMatrix{size(SM,1),size(SM,2),T}
@pure similar_type{SA<:StaticArray}(::Union{SA,Type{SA}}, sizes::Tuple{Integer,Integer}) = SMatrix{sizes[1], sizes[2], eltype(SA)}
@pure similar_type{SA<:StaticArray,T}(::Union{SA,Type{SA}}, ::Type{T}, sizes::Tuple{Integer,Integer}) = SMatrix{sizes[1], sizes[2], T}

@pure similar_type{SA<:StaticArray,T}(::Union{SA,Type{SA}}, ::Type{T}) = SArray{size(SA),T}
@pure similar_type{SA<:StaticArray,N}(::Union{SA,Type{SA}}, sizes::Tuple{Vararg{Integer,N}}) = SArray{sizes, eltype(SA)}
@pure similar_type{SA<:StaticArray,T,N}(::Union{SA,Type{SA}}, ::Type{T}, sizes::Tuple{Vararg{Integer,N}}) = SArray{sizes, T}

@inline similar{SV <: StaticVector}(::SV) = MVector{length(SV),eltype(SV)}()
@inline similar{SV <: StaticVector, T}(::SV, ::Type{T}) = MVector{length(SV),T}()
@inline similar{SA <: StaticArray}(::SA, sizes::Tuple{Integer}) = MVector{sizes[1], eltype{SV}}()
@inline similar{SA <: StaticArray}(::SA, size::Integer) = MVector{size, eltype{SV}}()
@inline similar{SA <: StaticArray, T}(::SA, ::Type{T}, sizes::Tuple{Integer}) = MVector{sizes[1],T}()
@inline similar{SA <: StaticArray, T}(::SA, ::Type{T}, size::Integer) = MVector{size,T}()

# Some specializations for the mutable case
@pure similar_type{MV<:MVector, T}(::Union{MV,Type{MV}}, ::Type{T}) = MVector{length(MV),T}
@pure similar_type{MA<:Union{MVector,MMatrix,MArray}}(::Union{MA,Type{MA}}, size::Integer) = MVector{size, eltype(MA)}
@pure similar_type{MA<:Union{MVector,MMatrix,MArray}}(::Union{MA,Type{MA}}, sizes::Tuple{Integer}) = MVector{sizes[1], eltype(MA)}
@pure similar_type{MA<:Union{MVector,MMatrix,MArray},T}(::Union{MA,Type{MA}}, ::Type{T}, size::Integer) = MVector{size, T}
@pure similar_type{MA<:Union{MVector,MMatrix,MArray},T}(::Union{MA,Type{MA}}, ::Type{T}, sizes::Tuple{Integer}) = MVector{sizes[1], T}

@pure similar_type{MM<:MMatrix, T}(::Union{MM,Type{MM}}, ::Type{T}) = MMatrix{size(MM,1),size(MM,2),T}
@pure similar_type{MA<:Union{MVector,MMatrix,MArray}}(::Union{MA,Type{MA}}, sizes::Tuple{Integer,Integer}) = MMatrix{sizes[1], sizes[2], eltype(MA)}
@pure similar_type{MA<:Union{MVector,MMatrix,MArray},T}(::Union{MA,Type{MA}}, ::Type{T}, sizes::Tuple{Integer,Integer}) = MMatrix{sizes[1], sizes[2], T}

@pure similar_type{MA<:Union{MVector,MMatrix,MArray},T}(::Union{MA,Type{MA}}, ::Type{T}) = MArray{size(MA),T}
@pure similar_type{MA<:Union{MVector,MMatrix,MArray},N}(::Union{MA,Type{MA}}, sizes::Tuple{Vararg{Integer,N}}) = MArray{sizes, eltype(MA)}
@pure similar_type{MA<:Union{MVector,MMatrix,MArray},T,N}(::Union{MA,Type{MA}}, ::Type{T}, sizes::Tuple{Vararg{Integer,N}}) = MArray{sizes, T}


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
