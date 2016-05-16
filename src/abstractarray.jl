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



# Can index linearly with a scalar, a tuple, or colon
Base.getindex(a::StaticArray, i::Int) = a.data[i]
@generated function Base.getindex{N}(a::StaticArray, i::NTuple{N,Int})
    newtype = similar_type(a, Val{(N,)})
    exprs = ntuple(n -> :(a[i[$n]]), N)
    return :($newtype($(Expr(:tuple, exprs...))))
end
Base.getindex(a::StaticArray, ::Colon) = a

# Multidimensional index generalizes the above
# Scalar
@generated function Base.getindex{Sizes,T,N}(a::StaticArray{Sizes,T,N}, i::Int...)
    if length(i) == 0
        return :(a.data[])
    elseif length(i) == 1
        return :(a.data[i])
    else
        return quote
            return a.data[sub2ind(Sizes,i...)]
        end
    end
end
# Other cases...
@generated function Base.getindex{Sizes,T,N}(a::StaticArray{Sizes,T,N}, i...)
    if length(i) == 0
        return :(a.data[])
    elseif length(i) == 1
        return :(a.data[i])
    else
        return quote
            return a.data[sub2ind(Sizes,i...)]
        end
    end
end

function Base.setindex!{Sizes,T}(a::MArray{Sizes,T}, v, i::Int)
    if i < 0 || i > prod(Sizes)
        throw(BoundsError(a,i))
    end

    Base.unsafe_setindex!(a, v, i)
end

function Base.unsafe_setindex!{Sizes,T}(a::MArray{Sizes,T}, v, i::Int)
    # Get a pointer to the object
    p = Base.data_pointer_from_objref(a)
    p = Base.unsafe_convert(Ptr{T}, p)

    # Store the value
    Base.unsafe_store!(p,convert(T,v),i)
end



#=
@generated function Base._getindex(::Base.LinearFast, A::StaticArray, I::Union{Real, AbstractArray, Colon}...)
    S = size(A)
    N = length(I)
    D = zeros(N)
    for i = 1:N
        if I[i] == Colon()
            D[i] = size(A, i)
        else
            D[i] = length(I[i])
        end
    end
    newsizes = (D...)
    newtype = similar_type(A, Val{newsizes})

    numel = prod(newsizes)
    inds = zeros(numel)
    j = ones(N)
    for i = 1:numel
        tmp = ntuple(jj -> Int(I[i][jj]), N)

        tmp2 = 1
        for k = 1:N
            inds[i] += tmp[k]*tmp2
            tmp2 *= S[k]
        end

        j[1] += 1
        for k = 1:N
            if j[k] > D[k]
                j[k] = 1
                j[k+1] += 1
            else
                break
            end
        end
    end

    exprs = map(i -> :(A.data[i]), inds)
    tuple_expr = Expr(:tuple, exprs...)

    return :($(newtype)($(tuple_expr)))
end=#

# TODO: decide whether this should fall back to MArray... at least it "works"
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
