##############
## Indexing ##
##############

# Indexing with no components
Base.getindex(a::StaticArray) = a.data[1]

# Can index linearly with a scalar, a tuple, or colon
Base.getindex(a::StaticArray, i::Int) = a.data[i]
@generated function Base.getindex{N}(a::StaticArray, i::NTuple{N,Int})
    newtype = similar_type(a, Val{(N,)})
    exprs = ntuple(n -> :(a[i[$n]]), N)
    return :($newtype($(Expr(:tuple, exprs...))))
end
Base.getindex(a::SArray, ::Colon) = a
Base.getindex{Sizes,T,N,D}(a::MArray{Sizes,T,N,D}, ::Colon) = MArray{Sizes,T,N,D}(a.data) # make a copy...

# Multidimensional index generalizes the above
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
            str = "Cannot index dimension $j of $a with a $(i[j]). Use an Int, NTuple{N,Int} or Colon."
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
            str = "Cannot index dimension $j of $a with a $(i[j]). Use an Int, NTuple{N,Int} or Colon."
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

# setindex! (no index)
Base.setindex!(a::MArray, v) = setindex!(a, v, 1)
Base.unsafe_setindex!{Sizes,T}(a::MArray{Sizes,T}, v) = Base.unsafe_setindex!(a, v, 1)


# setindex! (linear scalar)
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


# setindex! arbitray set of indices
function Base.setindex!{Sizes,T}(a::MArray{Sizes,T}, v, i)
    M = prod(Sizes)
    for ind in i
        if ind < 0 || ind > M
            throw(BoundsError(a,i))
        end
    end

    Base.unsafe_setindex!(a, v, i)
end

function Base.unsafe_setindex!{Sizes,T}(a::MArray{Sizes,T}, v, i)
    # Check if v is OK in size
    if length(v) != N
        throw(DimensionMismatch("tried to assign $(prod(size(v))) elements to $N destinations"))
    end

    # Get a pointer to the object
    p = Base.data_pointer_from_objref(a)
    p = Base.unsafe_convert(Ptr{T}, p)

    # Store the value
    for j in i
        Base.unsafe_store!(p,convert(T,v[j]),i[j])
    end
end


# setindex! (linear tuple optimization)
@generated function Base.unsafe_setindex!{Sizes,T,N}(a::MArray{Sizes,T}, v::StaticArray{Sizes}, i::NTuple{N,Int})
    # Compile-time check if v is OK in size
    if length(v) != N
        str = "tried to assign $(length(v)) elements to $N destinations"
        return :(throw(DimensionMismatch($str)))
    end

    quote
        # Get a pointer to the object
        p = Base.data_pointer_from_objref(a)
        p = Base.unsafe_convert(Ptr{T}, p)

        # Store the value
        for j in i
            Base.unsafe_store!(p,convert(T,v[j]),i[j])
        end
    end
end

# setindex! linear with : (Colon)
function Base.setindex!{Sizes,T}(a::MArray{Sizes,T}, v, i::Colon)
    Base.unsafe_setindex!(a, v, i)
end

function Base.unsafe_setindex!{Sizes,T}(a::MArray{Sizes,T}, v, ::Colon)
    # Check if v is OK in size
    if length(v) != length(a)
        throw(DimensionMismatch("tried to assign $(prod(size(v))) elements to $(length(a)) destinations"))
    end

    # Get a pointer to the object
    p = Base.data_pointer_from_objref(a)
    p = Base.unsafe_convert(Ptr{T}, p)

    # Store the value
    for j = 1:length(a)
        Base.unsafe_store!(p,convert(T,v[j]),j)
    end
end

# setindex(m_array, static_array, :) - optimization for know sizes
@generated function Base.unsafe_setindex!{Sizes1,Sizes2,T}(a::MArray{Sizes1,T}, v::StaticArray{Sizes2}, i::Colon)
    # Compile-time check if v is OK in size
    if length(v) != length(a)
        str = "tried to assign $(length(v)) elements to $(length(a)) destinations"
        return :(throw(DimensionMismatch($str)))
    end

    quote
        # Get a pointer to the object
        p = Base.data_pointer_from_objref(a)
        p = Base.unsafe_convert(Ptr{T}, p)

        # Store the value
        for j = 1:$(length(a))
            Base.unsafe_store!(p,convert(T,v[j]),j)
        end
    end
end


# setindex! multi-dimensional general case
function Base.setindex!{Sizes,T}(a::MArray{Sizes,T}, v, i...)
    # check that all i > 0
    for ii in i
        if isa(ii, Colon)
            continue
        elseif any(j -> j<1, ii)
            error("Indices must be positive integers: got $i")
        end
    end

    Base.unsafe_setindex!(a, v, i...)
end

function Base.unsafe_setindex!{Sizes,T}(a::MArray{Sizes,T}, v, i...)
    # First check v and i have correct sizes (ignoring singleton dimensions, mimicking Base.Array)
    i_sizes = ntuple(j -> i[j] == Colon ? Sizes1[j] : length(i[j]), length(i) )

    # "Flatten" out any singleton dimensions on input v and indices i
    Sizes2 = size(v)
    Sizes2_flattened = Vector{Int}()
    for j = 1:length(Sizes2)
        if Sizes2[j] > 1
            push!(Sizes2_flattened, Sizes2[j])
        end
    end

    i_sizes_flattened = Vector{Int}()
    for j = 1:length(i_sizes)
        if i_sizes[j] > 1
            push!(i_sizes_flattened, i_sizes[j])
        end
    end

    # If they aren't consistent, then we can't find a sensible way to assign the data
    if Sizes2_flattened != i_sizes_flattened
        println(Sizes2_flattened)
        println(i_sizes_flattened)
        throw(DimensionMismatch("tried to assign $(Sizes2)-dimensional data to $(i_sizes)-dimensional destination"))
    end

    M = prod(Sizes2)

    # Get a pointer to the object
    p = Base.data_pointer_from_objref(a)
    p = Base.unsafe_convert(Ptr{T}, p)

    # Store the value
    for ind_v = 1:M
        sub_i = ind2sub(i_sizes, ind_v)
        sub_a = ntuple(k -> i[k][sub_i[k]], length(i_sizes))
        ind_a = sub2ind(Sizes, sub_a...)
        Base.unsafe_store!(p, convert(T, v[ind_v]), ind_a)
    end
end

# setindex! multi-dimensional run-time-optimized case
@generated function Base.unsafe_setindex!{Sizes1,Sizes2,T}(a::MArray{Sizes1,T}, v::StaticArray{Sizes2}, i::Union{Int,TupleN{Int},Colon}...)
    # First check v and i have correct sizes (ignoring singleton dimensions, mimicking Base.Array)
    i_sizes = ntuple(j -> i[j] == Colon ? Sizes1[j] : length(i[j].parameters), length(i) )

    # "Flatten" out any singleton dimensions on input v and indices i
    Sizes2_flattened = Vector{Int}()
    for j = 1:length(Sizes2)
        if Sizes2[j] > 1
            push!(Sizes2_flattened, Sizes2[j])
        end
    end

    i_sizes_flattened = Vector{Int}()
    for j = 1:length(i_sizes)
        if i_sizes[j] > 1
            push!(i_sizes_flattened, i_sizes[j])
        end
    end

    # If they aren't consistent, then we can't find a sensible way to assign the data
    if Sizes2_flattened != i_sizes_flattened
        str = "tried to assign $(Sizes2)-dimensional data to $(i_sizes)-dimensional destination"
        return :(throw(DimensionMismatch($str)))
    end

    M = prod(Sizes2)

    return quote
        # Get a pointer to the object
        p = Base.data_pointer_from_objref(a)
        p = Base.unsafe_convert(Ptr{T}, p)

        # Store the value
        for ind_v = 1:$M
            sub_i = ind2sub($(i_sizes), ind_v)
            sub_a = ntuple(k -> i[k][sub_i[k]], $(length(i_sizes)))
            ind_a = sub2ind($(Sizes1), sub_a...)
            Base.unsafe_store!(p, convert(T, v[ind_v]), ind_a)
        end
    end
end
