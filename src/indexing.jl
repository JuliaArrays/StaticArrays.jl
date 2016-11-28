######################
## Linear Indexing  ##
######################

# What to do about @boundscheck and @inbounds? It's worse sometimes than @inline, for tuples...
@generated function getindex{SA<:StaticArray, S}(a::SA, inds::NTuple{S,Integer})
    newtype = similar_type(SA, (S,))
    exprs = [:(a[inds[$i]]) for i = 1:S]

    return quote
        $(Expr(:meta, :inline, :propagate_inbounds))
        return $(Expr(:call, newtype, Expr(:tuple, exprs...)))
    end
end

# Tuple indexing into AbstractArray. TODO, move into base
@generated function getindex{S}(a::AbstractArray, inds::NTuple{S,Integer})
    exprs = [:(a[inds[$i]]) for i = 1:S]
    return quote
        $(Expr(:meta, :inline, :propagate_inbounds))
        return $(Expr(:tuple, exprs...))
    end
end
# Convert to StaticArrays using tuples
# TODO think about bounds checks here.
@generated function getindex{S,T}(m::AbstractArray{T}, inds1::NTuple{S, Integer}, i2::Integer)
    exprs = [:(m[inds1[$j], i2]) for j = 1:S]
    return Expr(:tuple, exprs...)
end

@generated function getindex{S,T}(m::AbstractArray{T}, i1::Integer, inds2::NTuple{S, Integer})
    exprs = [:(m[i1, inds2[$j]]) for j = 1:S]
    return Expr(:tuple, exprs...)
end

@generated function getindex{S1,S2,T}(m::AbstractArray{T}, inds1::NTuple{S1, Integer}, inds2::NTuple{S2, Integer})
    exprs = [:(m[inds1[$j1], inds2[$j2]]) for j1 = 1:S1, j2 = 1:S2]
    return Expr(:call, SMatrix{S1,S2,T}, Expr(:tuple, exprs...)) # TODO decision: return tuple? Leave it?
end

# Static Vector indexing into AbstractArrays
@generated function getindex{T, I <: Integer}(
        a::AbstractArray{T}, inds::StaticVector{I}
    )
    S = length(inds)
    newtype = similar_type(inds, T, (S,))
    exprs = [:(a[inds[$i]]) for i = 1:S]
    return quote
        $(Expr(:meta, :inline, :propagate_inbounds))
        return $(Expr(:call, newtype, Expr(:tuple, exprs...)))
    end
end
# Convert to StaticArrays using tuples
# TODO think about bounds checks here.
@generated function getindex{T, I <: Integer}(
        m::AbstractArray{T}, inds1::StaticVector{I}, i2::Integer
    )
    S = length(inds1)
    newtype = similar_type(inds1, T, (S,)) # drop singular dimension like in base
    exprs = [:(m[inds1[$j], i2]) for j = 1:S]
    return Expr(:call, newtype, Expr(:tuple, exprs...))
end

@generated function getindex{T, I <: Integer}(
        m::AbstractArray{T}, i1::Integer, inds2::StaticVector{I}
    )
    S = length(inds)
    newtype = similar_type(inds2, T, (S,))
    exprs = [:(m[i1, inds2[$j]]) for j = 1:S]
    return Expr(:call, newtype, Expr(:tuple, exprs...))
end

@generated function getindex{I1 <: Integer,I2 <: Integer, T}(
        m::AbstractArray{T}, inds1::StaticVector{I1}, inds2::StaticVector{I2}
    )
    S1, S2 = length(inds1), lengths(inds2)
    exprs = [:(m[inds1[$j1], inds2[$j2]]) for j1 = 1:S1, j2 = 1:S2]
    return Expr(:call, SMatrix{S1, S2, T}, Expr(:tuple, exprs...)) # I guess SMatrix should be fine?
end


@generated function getindex{SA<:StaticArray}(a::SA, ::Colon)
    if SA <: StaticVector && a == similar_type(SA)
        return quote
            $(Expr(:meta, :inline))
            a
        end
    else
        l = length(SA)
        inds = 1:l
        return quote
            $(Expr(:meta, :inline, :propagate_inbounds))
            $(Expr(:call, :getindex, :a, Expr(:tuple, inds...)))
        end
    end
end

# MAYBE: fixed-size indexing with bools? They would have to be Val's...

# Size-indeterminate linear indexing seems to be provided by AbstractArray,
# returning a `Vector`.

# We seem to get an error from Base's implementation with UnitRange
#=
function Base.getindex(v::StaticVector, r::UnitRange)
    l = length(r)
    out = similar(v, (l,))
    for i in r
        out[i] = v[i]
    end
    return out
end
=#


# Same for setindex!
@generated function setindex!{SA <: StaticArray, I <: Integer}(
        a::SA, vals, inds::Union{Tuple{Vararg{I}}, StaticVector{I}}
    )
    S = inds <: Tuple ? length(inds.parameters) : length(inds)
    exprs = [:(a[inds[$i]] = vals[$i]) for i = 1:S]
    if vals <: StaticArray
        if length(vals) != S # We should be able to annotate that the RHS of the above exprs is @inbounds, but not the LHS...
            error("Dimension mismatch")
        end
        return quote
            $(Expr(:meta, :inline, :propagate_inbounds))
            $(Expr(:block, exprs...))
            return :vals
        end
    else
        return quote
            $(Expr(:meta, :inline, :propagate_inbounds))
            @boundscheck if length(vals) != $S
                error("Dimension mismatch")
            end
            $(Expr(:block, exprs...))
            return :vals
        end
    end
end

# this one for consistency
@generated function setindex!{I <: Integer}(
        a::AbstractArray, vals, inds::Union{Tuple{Vararg{I}}, StaticVector{I}}
    )
    S = inds <: Tuple ? length(inds.parameters) : length(inds)
    exprs = [:(a[inds[$i]] = vals[$i]) for i = 1:S]
    if vals <: StaticArray
        if length(vals) != S
            error("Dimension mismatch")
        end
        return quote
            $(Expr(:meta, :inline, :propagate_inbounds))
            $(Expr(:block, exprs...))
            return :vals
        end
    else
        return quote
            $(Expr(:meta, :inline, :propagate_inbounds))
            @boundscheck if length(vals) != $S
                error("Dimension mismatch")
            end
            $(Expr(:block, exprs...))
            return :vals
        end
    end
end

@generated function setindex!{SA<:StaticArray}(a::SA, vals, ::Colon)
    exprs = [:(a[$i] = vals[$i]) for i = 1:length(SA)]
    if vals <: StaticArray
        if length(vals) != length(SA)
            error("Dimension mismatch")
        end
        return quote
            $(Expr(:meta, :inline, :propagate_inbounds))
            $(Expr(:block, exprs...))
            return :vals
        end
    else
        return quote
            $(Expr(:meta, :inline, :propagate_inbounds))
            @boundscheck if length(vals) != $(length(SA))
                error("Dimension mismatch")
            end
            $(Expr(:block, exprs...))
            return :vals
        end
    end
end

###############################
## Two-dimensional Indexing  ##
###############################
# Special 2D case to begin with (and possibly good for avoiding splatting penalties?)
# Furthermore, avoids stupidity regarding two-dimensional indexing on 3+ dimensional arrays!
@generated function getindex{SM<:StaticMatrix}(m::SM, i1::Integer, i2::Integer)
    return quote
        $(Expr(:meta, :inline, :propagate_inbounds))
        @boundscheck if (i1 < 1 || i1 > $(size(SM,1)) || i2 < 1 || i2 > $(size(SM,2)))
            throw(BoundsError(m, (i1,i2)))
        end

        @inbounds return m[i1 + $(size(SM,1))*(i2-1)]
    end
end

# TODO put bounds checks here, as they should have less overhead here
@generated function getindex{SM<:StaticMatrix, S1, S2}(m::SM, inds1::NTuple{S1,Integer}, inds2::NTuple{S2,Integer})
    newtype = similar_type(SM, (S1, S2))
    exprs = [:(m[inds1[$i1], inds2[$i2]]) for i1 = 1:S1, i2 = 1:S2]

    return quote
        $(Expr(:meta, :inline, :propagate_inbounds))
        return $(Expr(:call, newtype, Expr(:tuple, exprs...)))
    end
end

@generated function getindex{SM<:StaticMatrix, S2}(m::SM, i1::Integer, inds2::NTuple{S2,Integer})
    newtype = similar_type(SM, (S2,))
    exprs = [:(m[i1, inds2[$i2]]) for i2 = 1:S2]

    return quote
        $(Expr(:meta, :inline, :propagate_inbounds))
        return $(Expr(:call, newtype, Expr(:tuple, exprs...)))
    end
end

@generated function getindex{SM<:StaticMatrix, S1}(m::SM, inds1::NTuple{S1,Integer}, i2::Integer)
    newtype = similar_type(SM, (S1,))
    exprs = [:(m[inds1[$i1], i2]) for i1 = 1:S1]

    return quote
        $(Expr(:meta, :inline, :propagate_inbounds))
        return $(Expr(:call, newtype, Expr(:tuple, exprs...)))
    end
end







# TODO: bounds checking?
@generated function setindex!{SM<:StaticMatrix, S1, S2}(m::SM, val, inds1::NTuple{S1,Integer}, inds2::NTuple{S2,Integer})
    exprs = [:(m[inds1[$i1], inds2[$i2]] = val[$i1,$i2]) for i1 = 1:S1, i2 = 1:S2]

    return quote
        $(Expr(:meta, :inline, :propagate_inbounds))
        $(Expr(:block, exprs...))
        return val
    end
end
# TODO put bounds checks here, as they should have less overhead here
@generated function getindex{SM<:StaticMatrix}(m::SM, ::Colon, inds2::Union{Integer, Tuple{Vararg{Integer}}})
    inds1 = ntuple(identity, size(SM,1))
    quote
        $(Expr(:meta, :inline, :propagate_inbounds))
        m[$inds1, inds2]
    end
end

# TODO put bounds checks here, as they should have less overhead here
@generated function getindex{SM<:StaticMatrix}(m::SM, inds1::Union{Integer, Tuple{Vararg{Integer}}}, ::Colon)
    inds2 = ntuple(identity, size(SM,2))
    quote
        $(Expr(:meta, :inline, :propagate_inbounds))
        m[inds1, $inds2]
    end
end


@generated function getindex{SM<:StaticMatrix}(m::SM, ::Colon, ::Colon)
    inds1 = ntuple(identity, size(SM,1))
    inds2 = ntuple(identity, size(SM,2))
    quote
        $(Expr(:meta, :inline))
        @inbounds return m[$inds1, $inds2]
    end
end


@generated function setindex!{SM<:StaticMatrix, S2}(m::SM, val, i1::Integer, inds2::NTuple{S2,Integer})
    newtype = similar_type(SM, (S2,))
    exprs = [:(m[i1, inds2[$i2]] = val[$i2]) for i2 = 1:S2]

    return quote
        $(Expr(:meta, :inline, :propagate_inbounds))
        return $(Expr(:block, exprs...))
    end
end
# TODO expand out tuples to vectors in size-indeterminate cases

# 2D setindex!

@generated function setindex!{SM<:StaticMatrix}(m::SM, val, i1::Integer, i2::Integer)
    return quote
        $(Expr(:meta, :inline))
        @boundscheck if (i1 < 1 || i1 > $(size(SM,1)) || i2 < 1 || i2 > $(size(SM,2)))
            throw(BoundsError(m, (i1,i2)))
        end

        @inbounds return m[i1 + $(size(SM,1))*(i2-1)] = val
    end
end



@generated function setindex!{SM<:StaticMatrix, S1}(m::SM, val, inds1::NTuple{S1,Integer}, i2::Integer)
    newtype = similar_type(SM, (S1,))
    exprs = [:(m[inds1[$i1], i2] = val[$i1]) for i1 = 1:S1]

    return quote
        $(Expr(:meta, :inline, :propagate_inbounds))
        return $(Expr(:block, exprs...))
    end
end

# TODO put bounds checks here, as they should have less overhead here
@generated function setindex!{SM<:StaticMatrix}(m::SM, val, ::Colon, inds2::Union{Integer, Tuple{Vararg{Integer}}})
    inds1 = ntuple(identity, size(SM,1))
    quote
        $(Expr(:meta, :inline, :propagate_inbounds))
        m[$inds1, inds2] = val
    end
end

# TODO put bounds checks here, as they should have less overhead here
@generated function setindex!{SM<:StaticMatrix}(m::SM, val, inds1::Union{Integer, Tuple{Vararg{Integer}}}, ::Colon)
    inds2 = ntuple(identity, size(SM,2))
    quote
        $(Expr(:meta, :inline, :propagate_inbounds))
        m[inds1, $inds2] = val
    end
end

# Check bounds for val here?
@generated function setindex!{SM<:StaticMatrix}(m::SM, val, ::Colon, ::Colon)
    inds1 = ntuple(identity, size(SM,1))
    inds2 = ntuple(identity, size(SM,2))
    quote
        $(Expr(:meta, :inline))
        return m[$inds1, $inds2] = val
    end
end

#################################
## Multi-dimensional Indexing  ##
#################################

# Scalar indexing:

@generated function getindex(a::StaticArray, i1::Integer, i2::Integer, i3::Integer)
    @assert ndims(a) <= 3

    strides = [1, size(a,1), size(a,1)*size(a,2)]

    N = 3
    ind_exprs = [:i1, :i2, :i3]
    exprs = [j == 1 ? ind_exprs[1] : :( $(strides[j]) * ($(ind_exprs[j])-1) ) for j = 1:N]
    ind_expr = Expr(:call, :+, exprs...)

    bounds_expr = :(i1 < 1 || i1 > $(size(a,1)) || i2 < 1 || i2 > $(size(a,2)) || i3 < 1 || i3 > $(size(a,3)))

    return quote
        $(Expr(:meta, :inline))
        @boundscheck if $bounds_expr
            throw(BoundsError(a, (i1,i2,i3)))
        end

        @inbounds return a[$ind_expr]
    end
end

# TODO check speed (splatting penalty)
@generated function getindex(a::StaticArray, i1::Integer, i2::Integer, i_n::Integer...)
    @assert ndims(a) <= 2 + length(i_n)

    N = 2 + length(i_n)

    strides = ones(Int, N)
    for j = 2:N
        for k = 1:j - 1
            strides[j] *= size(a, k)
        end
    end

    ind_exprs = [:i1, :i2, [:(i_n[$j]) for j = 1:length(i_n)]...]
    exprs = [j == 1 ? ind_exprs[1] : :( $(strides[j]) * ($(ind_exprs[j])-1) ) for j = 1:N]
    ind_expr = Expr(:call, :+, exprs...)

    bounds_expr = :(i1 < 1 || i1 > $(size(a,1)) || i2 < 1 || i2 > $(size(a,2)))
    for j = 1:N-2
        bound_expr = :($bounds_expr || i_n[$j] < 1 || i_n[$j] > $(size(a,2+j)))
    end

    return quote
        $(Expr(:meta, :inline))
        @boundscheck if $bounds_expr
            throw(BoundsError(a, (i1,i2,i_n...)))
        end

        @inbounds return a[$ind_expr]
    end
end

@generated function setindex!(a::StaticArray, val, i1::Integer, i2::Integer, i3::Integer)
    @assert ndims(a) <= 3

    strides = [1, size(a,1), size(a,1)*size(a,2)]

    N = 3
    ind_exprs = [:i1, :i2, :i3]
    exprs = [j == 1 ? ind_exprs[1] : :($(strides[j]) * ($(ind_exprs[j])-1) ) for j = 1:N]
    ind_expr = Expr(:call, :+, exprs...)

    bounds_expr = :(i1 < 1 || i1 > $(size(a,1)) || i2 < 1 || i2 > $(size(a,2)) || i3 < 1 || i3 > $(size(a,3)))

    return quote
        $(Expr(:meta, :inline))
        @boundscheck if $bounds_expr
            throw(BoundsError(a, (i1,i2,3)))
        end

        @inbounds return a[$ind_expr] = val
    end
end

# TODO check speed (splatting penalty)
@generated function setindex!(a::StaticArray, val, i1::Integer, i2::Integer, i_n::Integer...)
    @assert ndims(a) <= 2 + length(i_n)

    N = 2 + length(i_n)

    strides = ones(Int, N)
    for j = 2:N
        for k = 1:j - 1
            strides[j] *= size(a, k)
        end
    end

    ind_exprs = [:i1, :i2, [:(i_n[$j]) for j = 1:length(i_n)]...]
    exprs = [j == 1 ? ind_exprs[1] : :( $(strides[j]) * ($(ind_exprs[j])-1) ) for j = 1:N]
    ind_expr = Expr(:call, :+, exprs...)

    bounds_expr = :(i1 < 1 || i1 > $(size(a,1)) || i2 < 1 || i2 > $(size(a,2)))
    for j = 1:N-2
        bound_expr = :($bounds_expr || i_n[$j] < 1 || i_n[$j] > $(size(a,2+j)))
    end

    return quote
        $(Expr(:meta, :inline))
        @boundscheck if $bounds_expr
            throw(BoundsError(a, (i1,i2,i_n...)))
        end

        @inbounds return a[$ind_expr] = val
    end
end

# TODO higher dimensional versions with tuples and :

#=
##############
## Indexing ##
##############

# Indexing with no components
Base.getindex(a::StaticArray) = a.data[1]

# Can index linearly with a scalar, a tuple, or colon (overspecified to ovoid ambiguity problems in julia 0.5)
Base.getindex{Sizes,T,N,D}(a::SArray{Sizes,T,N,D}, i::Int) = a.data[i]
Base.getindex{Sizes,T,N,D}(a::MArray{Sizes,T,N,D}, i::Int) = a.data[i]
@generated function Base.getindex{N}(a::SArray, i::NTuple{N,Int})
    newtype = similar_type(a, Val{(N,)})
    exprs = ntuple(n -> :(a[i[$n]]), N)
    return :($newtype($(Expr(:tuple, exprs...))))
end
@generated function Base.getindex{N}(a::MArray, i::NTuple{N,Int})
    newtype = similar_type(a, Val{(N,)})
    exprs = ntuple(n -> :(a[i[$n]]), N)
    return :($newtype($(Expr(:tuple, exprs...))))
end
Base.getindex(a::SArray, ::Colon) = SVector(a.data)
Base.getindex(a::MArray, ::Colon) = MVector(a.data) # Makes a copy?

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
    N = length(i)

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
    for ii in 1:length(i)
        if isa(i[ii], Colon)
            continue
        elseif any(j -> j<1 || j>Sizes[ii], i[ii])
            error("Indices must be positive integers and in-range: got $i for $Sizes-dimensional array")
        end
    end

    Base.unsafe_setindex!(a, v, i...)
end

function Base.unsafe_setindex!{Sizes,T}(a::MArray{Sizes,T}, v, i...)
    # First check v and i have correct sizes (ignoring singleton dimensions, mimicking Base.Array)
    i_sizes = ntuple(j -> i[j] == Colon() ? Sizes[j] : length(i[j]), length(i) )

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
=#
