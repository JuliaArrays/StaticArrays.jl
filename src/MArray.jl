type MArray{Size, T, N, L} <: StaticArray{T, N}
    data::NTuple{L,T}

    function MArray(x::NTuple{L,T})
        check_marray_parameters(Val{Size}, T, Val{N}, Val{L})
        new(x)
    end

    function MArray(x::NTuple{L})
        check_marray_parameters(Val{Size}, T, Val{N}, Val{L})
        new(convert_ntuple(T, x))
    end

    function MArray()
        check_marray_parameters(Val{Size}, T, Val{N}, Val{L})
        new()
    end
end

@generated function check_marray_parameters{Size,T,N,L}(::Type{Val{Size}}, ::Type{T}, ::Type{Val{N}}, ::Type{Val{L}})
    if !(isa(Size, Tuple{Vararg{Int}}))
        error("MArray parameter Size must be a tuple of Ints (e.g. `MArray{(3,3)}`)")
    end

    if L != prod(Size) || L < 0 || minimum(Size) < 0 || length(Size) != N
        error("Size mismatch in MArray parameters. Got size $Size, dimension $N and length $L.")
    end

    return nothing
end

@generated function (::Type{MArray{Size,T,N}}){Size,T,N}(x)
    return quote
        $(Expr(:meta, :inline))
        MArray{Size,T,N,$(prod(Size))}(x)
    end
end

@generated function (::Type{MArray{Size,T}}){Size,T}(x)
    return quote
        $(Expr(:meta, :inline))
        MArray{Size,T,$(length(Size)),$(prod(Size))}(x)
    end
end

@generated function (::Type{MArray{Size}}){Size}(x)
    return quote
        $(Expr(:meta, :inline))
        MArray{Size,$(promote_tuple_eltype(x)),$(length(Size)),$(prod(Size))}(x)
    end
end

@generated function (::Type{MArray{Size,T,N}}){Size,T,N}()
    return quote
        $(Expr(:meta, :inline))
        MArray{Size, T, N, $(prod(Size))}()
    end
end

@generated function (::Type{MArray{Size,T}}){Size,T}()
    return quote
        $(Expr(:meta, :inline))
        MArray{Size, T, $(length(Size)), $(prod(Size))}()
    end
end

# Some more advanced constructor-like functions
@inline zeros{Size}(::Type{MArray{Size}}) = zeros(MArray{Size,Float64})
@inline ones{Size}(::Type{MArray{Size}}) = ones(MArray{Size,Float64})


####################
## MArray methods ##
####################

@pure size{Size}(::Union{MArray{Size},Type{MArray{Size}}}) = Size
@pure size{Size,T}(::Type{MArray{Size,T}}) = Size
@pure size{Size,T,N}(::Type{MArray{Size,T,N}}) = Size
@pure size{Size,T,N,L}(::Type{MArray{Size,T,N,L}}) = Size

function getindex(v::MArray, i::Integer)
    Base.@_inline_meta
    v.data[i]
end

@propagate_inbounds setindex!{S,T}(v::MArray{S,T}, val, i::Integer) = setindex!(v, convert(T, val), i)
@inline function setindex!{S,T}(v::MArray{S,T}, val::T, i::Integer)
    @boundscheck if i < 1 || i > length(v)
        throw(BoundsError())
    end

    if isbits(T)
        unsafe_store!(Base.unsafe_convert(Ptr{T}, Base.data_pointer_from_objref(v)), val, i)
    else # TODO check that this isn't crazy. Also, check it doesn't cause problems with GC...
        unsafe_store!(Base.unsafe_convert(Ptr{Ptr{Void}}, Base.data_pointer_from_objref(v.data)), Base.data_pointer_from_objref(val), i)
    end

    return val
end

@inline Tuple(v::MArray) = v.data

#=
macro MArray(ex)
    @assert isa(ex, Expr)
    if ex.head == :vect # Vector
        return esc(Expr(:call, MArray{(length(ex.args),)}, Expr(:tuple, ex.args...)))
    elseif isa(ex, Expr) && ex.head == :ref
        return esc(Expr(:call, Expr(:curly, :MArray, (length(ex.args[2:end]),), ex.args[1]), Expr(:tuple, ex.args[2:end]...)))
    elseif ex.head == :hcat # 1 x n
        s1 = 1
        s2 = length(ex.args)
        return Expr(:call, MArray{(s1, s2)}, Expr(:tuple, ex.args...))
    elseif ex.head == :vcat
        if isa(ex.args[1], Expr) && ex.args[1].head == :row # n x m
            # Validate
            s1 = length(ex.args)
            s2s = map(i -> ((isa(ex.args[i], Expr) && ex.args[i].head == :row) ? length(ex.args[i].args) : 1), 1:s1)
            s2 = minimum(s2s)
            if maximum(s2s) != s2
                error("Rows must be of matching lengths")
            end

            exprs = [ex.args[i].args[j] for i = 1:s1, j = 1:s2]
            return Expr(:call, MArray{(s1, s2)}, Expr(:tuple, exprs...))
        else # n x 1
            return Expr(:call, MArray{(length(ex.args), 1)}, Expr(:tuple, ex.args...))
        end
    end
end=#

macro MArray(ex)
    if !isa(ex, Expr)
        error("Bad input for @MArray")
    end

    if ex.head == :vect  # vector
        return esc(Expr(:call, MArray{(length(ex.args),)}, Expr(:tuple, ex.args...)))
    elseif ex.head == :ref # typed, vector
        return esc(Expr(:call, Expr(:curly, :MArray, ((length(ex.args)-1),), ex.args[1]), Expr(:tuple, ex.args[2:end]...)))
    elseif ex.head == :hcat # 1 x n
        s1 = 1
        s2 = length(ex.args)
        return esc(Expr(:call, MArray{(s1, s2)}, Expr(:tuple, ex.args...)))
    elseif ex.head == :typed_hcat # typed, 1 x n
        s1 = 1
        s2 = length(ex.args) - 1
        return esc(Expr(:call, Expr(:curly, :MArray, (s1, s2), ex.args[1]), Expr(:tuple, ex.args[2:end]...)))
    elseif ex.head == :vcat
        if isa(ex.args[1], Expr) && ex.args[1].head == :row # n x m
            # Validate
            s1 = length(ex.args)
            s2s = map(i -> ((isa(ex.args[i], Expr) && ex.args[i].head == :row) ? length(ex.args[i].args) : 1), 1:s1)
            s2 = minimum(s2s)
            if maximum(s2s) != s2
                error("Rows must be of matching lengths")
            end

            exprs = [ex.args[i].args[j] for i = 1:s1, j = 1:s2]
            return esc(Expr(:call, MArray{(s1, s2)}, Expr(:tuple, exprs...)))
        else # n x 1
            return esc(Expr(:call, MArray{(length(ex.args), 1)}, Expr(:tuple, ex.args...)))
        end
    elseif ex.head == :typed_vcat
        if isa(ex.args[2], Expr) && ex.args[2].head == :row # typed, n x m
            # Validate
            s1 = length(ex.args) - 1
            s2s = map(i -> ((isa(ex.args[i+1], Expr) && ex.args[i+1].head == :row) ? length(ex.args[i+1].args) : 1), 1:s1)
            s2 = minimum(s2s)
            if maximum(s2s) != s2
                error("Rows must be of matching lengths")
            end

            exprs = [ex.args[i+1].args[j] for i = 1:s1, j = 1:s2]
            return esc(Expr(:call, Expr(:curly, :MArray, (s1, s2), ex.args[1]), Expr(:tuple, exprs...)))
        else # typed, n x 1
            return esc(Expr(:call, Expr(:curly, :MArray, (length(ex.args)-1, 1), ex.args[1]), Expr(:tuple, ex.args[2:end]...)))
        end
    end

    error("Bad input for @MArray")
end
