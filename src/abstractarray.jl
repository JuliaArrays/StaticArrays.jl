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
Base.getindex(a::StaticArray) = a.data[1]
Base.getindex(a::StaticArray, i::Int) = a.data[i]
@generated function Base.getindex{N}(a::StaticArray, i::NTuple{N,Int})
    newtype = similar_type(a, Val{(N,)})
    exprs = ntuple(n -> :(a[i[$n]]), N)
    return :($newtype($(Expr(:tuple, exprs...))))
end
Base.getindex(a::SArray, ::Colon) = a
Base.getindex{Sizes,T,N,D}(a::MArray{Sizes,T,N,D}, ::Colon) = MArray{Sizes,T,N,D}(a.data) # make a copy...

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

@generated function Base.getindex{Sizes,T,N}(a::SArray{Sizes,T,N}, i...)
    # Striding lengths
    strides = [1, cumprod(collect(Sizes)[1:end-1])...]

    # Get the parameters of the new matrix
    apl_slicing = VERSION >= v"0.5-"
    NewN = 0
    NewSizes = Vector{Int}()
    OldSizes = Vector{Int}() # Same as NewSizes but includes singleton 1's
    at_end = true
    is_singleton = trues(N)
    for j = length(i):-1:1
        if i[j] == Int
            unshift!(OldSizes,1)
            if !apl_slicing
                if !at_end
                    NewN += 1
                    unshift!(NewSizes,1)
                    is_singleton[j] = false
                end
            end
        elseif i[j] <: TupleN{Int}
            NewN += 1
            unshift!(NewSizes,length(i[j].parameters))
            unshift!(OldSizes,length(i[j].parameters))
            is_singleton[j] = false
            at_end = false
        elseif i[j] == Colon
            NewN += 1
            unshift!(NewSizes,Sizes[j])
            unshift!(OldSizes,Sizes[j])
            is_singleton[j] = false
            at_end = false
        else
            str = "Cannot index dimension $j of $a with a $(i[j])"
            return :(error($str))
        end
    end
    NewSizes = (NewSizes...)
    NewM = prod(NewSizes)

    # Bail early if possible
    if NewN == 0
        return :(a.data[sub2ind(Sizes,i...)])
    end

    NewType = MArray{NewSizes,T,NewN,NTuple{NewM,T}}

    # Now we build an expression for each new element
    exprs = Vector{Expr}()
    inds_old = ones(Int,N)
    for j = 1:NewM
        sum_exprs = ntuple(p -> :($(strides[p]) * (i[$p][$(inds_old[p])] - 1)), N)
        push!(exprs, :(a.data[$(Expr(:call, :+, 1, sum_exprs...))]))

        if j < NewM
            inds_old[1] += 1
        end
        for k = 1:N
            if inds_old[k] > OldSizes[k]
                inds_old[k] = 1
                inds_old[k+1] += 1
            else
                break
            end
        end
    end

    return :(SArray{$NewSizes,$T,$NewN,NTuple{$NewM,$T}}($(Expr(:tuple, exprs...))))
end

@generated function Base.getindex{Sizes,T,N}(a::MArray{Sizes,T,N}, i...)
    # Striding lengths
    strides = [1, cumprod(collect(Sizes)[1:end-1])...]

    # Get the parameters of the new matrix
    apl_slicing = VERSION >= v"0.5-"
    NewN = 0
    NewSizes = Vector{Int}()
    OldSizes = Vector{Int}() # Same as NewSizes but includes singleton 1's
    at_end = true
    is_singleton = trues(N)
    for j = length(i):-1:1
        if i[j] == Int
            unshift!(OldSizes,1)
            if !apl_slicing
                if !at_end
                    NewN += 1
                    unshift!(NewSizes,1)
                    is_singleton[j] = false
                end
            end
        elseif i[j] <: TupleN{Int}
            NewN += 1
            unshift!(NewSizes,length(i[j].parameters))
            unshift!(OldSizes,length(i[j].parameters))
            is_singleton[j] = false
            at_end = false
        elseif i[j] == Colon
            NewN += 1
            unshift!(NewSizes,Sizes[j])
            unshift!(OldSizes,Sizes[j])
            is_singleton[j] = false
            at_end = false
        else
            str = "Cannot index dimension $j of $a with a $(i[j])"
            return :(error($str))
        end
    end
    NewSizes = (NewSizes...)
    NewM = prod(NewSizes)

    # Bail early if possible
    if NewN == 0
        return :(a.data[sub2ind(Sizes,i...)])
    end

    NewType = MArray{NewSizes,T,NewN,NTuple{NewM,T}}

    # Now we build an expression for each new element
    exprs = Vector{Expr}()
    inds_old = ones(Int,N)
    for j = 1:NewM
        sum_exprs = ntuple(p -> :($(strides[p]) * (i[$p][$(inds_old[p])] - 1)), N)
        push!(exprs, :(a.data[$(Expr(:call, :+, 1, sum_exprs...))]))

        if j < NewM
            inds_old[1] += 1
        end
        for k = 1:N
            if inds_old[k] > OldSizes[k]
                inds_old[k] = 1
                inds_old[k+1] += 1
            else
                break
            end
        end
    end

    return :(MArray{$NewSizes,$T,$NewN,NTuple{$NewM,$T}}($(Expr(:tuple, exprs...))))
end

#=
# Other cases...
@generated function Base.getindex{Sizes,T,N}(a::MArray{Sizes,T,N}, i...)
    ind_l_exprs = Vector{Exprs}()
    ind_r_exprs = Vector{Exprs}()
    k0 = 1
    apl_slicing = VERSION < v"0.5-"
    NewN = 0
    NewSizes = Vector{Int}()
    i_sizes = Vector{Int}()
    at_end = true
    for j = length(i):-1:1
        sj = Symbol("i_$j")
        if i.parameters[j] == Int
            unshift!(ind_exprs, :($k0 * (i[$j]-1)))
            unshift!(i_sizes, 1)

            if !apl_slicing
                if !at_end
                    NewN += 1
                    unshift!(NewSizes,1)

                end
            end
        elseif i[j] <: TupleN{Int}
            unshift!(ind_exprs, :($k0 * (i[j][$sj]-1)))
            unshift!(i_sizes, length(i[j].parameters))

            NewN += 1
            unshift!(NewSizes,length(i[j].parameters))
            at_end = false
        elseif i[j] == Colon
            unshift!(ind_exprs, :($k0 * ($sj-1)))
            unshift!(i_sizes, Size[j])

            NewN += 1
            unshift!(NewSizes,Size[j])
            at_end = false
        else
            str = "Cannot index dimension $j of $a with a $(i[j])"
            return :(error($str))
        end
        k0 = k0 * Sizes[j]
    end
    NewM = prod(NewSizes)

    return quote
        out = MArray{$NewSizes,T,NTuple{$NewM,T}}()

        $(NewM > 0 ? :(for i_1 = 1:$(i_sizes[1])) : nothing)
        $(Size > 1 ? :(for i_1 = 1:$(i_sizes[2])) : nothing)
        $(Size > 2 ? :(for i_1 = 1:$(i_sizes[3])) : nothing)

        out[i_1, i_2] = a[$(Expr(:+,ind_exprs...))]

        $(Size > 0 ? :(end) : nothing)
        $(Size > 1 ? :(end) : nothing)
        $(Size > 2 ? :(end) : nothing)

        return out
    end
end =#

# setindex! (linear)
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

# reshape()
Base.reshape(::StaticArray, ::TupleN{Int}) = error("Need reshape size as a type-paramter. Use reshape(staticarray,Val{(s₁,...)}).")
@generated function Base.reshape{Sizes,T,N,D,NewSizes}(a::SArray{Sizes,T,N,D},::Type{Val{NewSizes}})
    NewN = length(NewSizes)
    :($(SArray{NewSizes,T,NewN,D})(a.data))
end
@generated function Base.reshape{Sizes,T,N,D,NewSizes}(a::MArray{Sizes,T,N,D},::Type{Val{NewSizes}})
    NewN = length(NewSizes)
    :($(MArray{NewSizes,T,NewN,D})(a.data))
end

# TODO: permutedims() (and transpose/ctranspose in linalg)
