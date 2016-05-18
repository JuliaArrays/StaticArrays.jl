# Methods necessary to fulfil the AbstractArray interface

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



"""
    similar_type(staticarray, [element_type], [Val{dims}])

Create a static array type with the same (or optionally modified) element type and size.
"""
similar_type(A::StaticArray) = error("Similar type not defined for $(typeof(A))")
similar_type{SA<:StaticArray}(::Type{SA}) = error("Similar type not defined for $(typeof(A))")
similar_type{Sizes,T,N,D}(::Union{SArray{Sizes,T,N,D},Type{SArray{Sizes,N,T,D}}}) = SArray{Sizes,N,T,D}
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
similar_type{Sizes,T,N,D,NewT}(::Union{SArray{Sizes,T,N,D},Type{SArray{Sizes,T,N,D}}}, ::Type{NewT}) = SArray{Sizes,NewT}
similar_type{Sizes,T,N,D,NewT,NewSizes}(::Union{SArray{Sizes,N,T,D},Type{SArray{Sizes,T,N,D}}}, ::Type{NewT}, ::Type{Val{NewSizes}}) = SArray{NewSizes,NewT}

similar_type{Sizes,T,N,D}(::Union{MArray{Sizes,T,N,D},Type{MArray{Sizes,N,T,D}}}) = MArray{Sizes,N,T,D}
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
similar_type{Sizes,T,N,D,NewT}(::Union{MArray{Sizes,T,N,D},Type{MArray{Sizes,T,N,D}}}, ::Type{NewT}) = MArray{Sizes,NewT}
similar_type{Sizes,T,N,D,NewT,NewSizes}(::Union{MArray{Sizes,N,T,D},Type{MArray{Sizes,T,N,D}}}, ::Type{NewT}, ::Type{Val{NewSizes}}) = MArray{NewSizes,NewT}

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
