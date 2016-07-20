type MMatrix{S1, S2, T, L} <: StaticMatrix{T}
    data::NTuple{L, T}

    function MMatrix(d::NTuple{L,T})
        check_MMatrix_params(Val{S1}, Val{S2}, T, Val{L})
        new(d)
    end

    function MMatrix(d::NTuple{L})
        check_MMatrix_params(Val{S1}, Val{S2}, T, Val{L})
        new(convert_ntuple(T, d))
    end

    function MMatrix()
        check_MMatrix_params(Val{S1}, Val{S2}, T, Val{L})
        new()
    end
end

@generated function check_MMatrix_params{S1,S2,L}(::Type{Val{S1}}, ::Type{Val{S2}}, T, ::Type{Val{L}})
    if !(T <: DataType) # I think the way types are handled in generated fnctions might have changed in 0.5?
        return :(error("MMatrix: Parameter T must be a DataType. Got $T"))
    end

    if !isa(S1, Int) || !isa(S2, Int) || !isa(L, Int) || S1 < 0 || S2 < 0 || L < 0
        return :(error("MMatrix: Sizes must be positive integers. Got $S1 × $S2 ($L elements)"))
    end

    if S1*S2 == L
        return nothing
    else
        str = "Size mismatch in MMatrix. S1 = $S1, S2 = $S2, but recieved $L elements"
        return :(error(str))
    end
end

@generated function (::Type{MMatrix{S1}}){S1,L}(x::NTuple{L})
    S2 = div(L, S1)
    if S1*S2 != L
        error("Incorrect matrix sizes. $S1 does not divide $L elements")
    end
    T = promote_tuple_eltype(x)

    return quote
        $(Expr(:meta, :inline))
        MMatrix{S1, $S2, $T, L}(x)
    end
end

@generated function (::Type{MMatrix{S1,S2}}){S1,S2,L}(x::NTuple{L})
    T = promote_tuple_eltype(x)

    return quote
        $(Expr(:meta, :inline))
        MMatrix{S1, S2, $T, L}(x)
    end
end

@generated function (::Type{MMatrix{S1,S2,T}}){S1,S2,T,L}(x::NTuple{L})
    return quote
        $(Expr(:meta, :inline))
        MMatrix{S1, S2, T, L}(x)
    end
end

@generated function (::Type{MMatrix{S1,S2,T}}){S1,S2,T}()
    return quote
        $(Expr(:meta, :inline))
        MMatrix{S1, S2, T, $(S1*S2)}()
    end
end

@inline convert{S1,S2,T}(::Type{MMatrix{S1,S2}}, a::StaticArray{T}) = MMatrix{S1,S2,T}(Tuple(a))
@inline convert{S1,S2,T}(::Type{MMatrix{S1,S2}}, a::AbstractArray{T}) = MMatrix{S1,S2,T}((a...))

# Some more advanced constructor-like functions
@inline eye{N}(::Type{MMatrix{N}}) = eye(MMatrix{N,N})
@inline eye{N,M}(::Type{MMatrix{N,M}}) = eye(MMatrix{N,M,Float64})
@inline zeros{N,M}(::Type{MMatrix{N,M}}) = zeros(MMatrix{N,M,Float64})
@inline ones{N,M}(::Type{MMatrix{N,M}}) = ones(MMatrix{N,M,Float64})

#####################
## MMatrix methods ##
#####################

@pure size{S1,S2}(::Union{MMatrix{S1,S2},Type{MMatrix{S1,S2}}}) = (S1, S2)
@pure size{S1,S2,T}(::Type{MMatrix{S1,S2,T}}) = (S1, S2)
@pure size{S1,S2,T,L}(::Type{MMatrix{S1,S2,T,L}}) = (S1, S2)

function getindex(v::MMatrix, i::Integer)
    Base.@_inline_meta
    v.data[i]
end

@propagate_inbounds setindex!{S1,S2,T}(m::MMatrix{S1,S2,T}, val, i::Integer) = setindex!(m, convert(T, val), i)
@inline function setindex!{S1,S2,T}(m::MMatrix{S1,S2,T}, val::T, i::Integer)
    #@boundscheck if i < 1 || i > length(m)
    #    throw(BoundsError())
    #end

    if isbits(T)
        unsafe_store!(Base.unsafe_convert(Ptr{T}, Base.data_pointer_from_objref(m)), val, i)
    else # TODO check that this isn't crazy. Also, check it doesn't cause problems with GC...
        unsafe_store!(Base.unsafe_convert(Ptr{Ptr{Void}}, Base.data_pointer_from_objref(m.data)), Base.data_pointer_from_objref(val), i)
    end

    return val
end

@inline Tuple(v::MMatrix) = v.data

macro MMatrix(ex)
    if !isa(ex, Expr)
        error("Bad input for @MMatrix")
    end
    if ex.head == :vect && length(ex.args) == 1 # 1 x 1
        return esc(Expr(:call, MMatrix{1, 1}, Expr(:tuple, ex.args[1])))
    elseif ex.head == :ref && length(ex.args) == 2 # typed, 1 x 1
        return esc(Expr(:call, Expr(:curly, :MMatrix, 1, 1, ex.args[1]), Expr(:tuple, ex.args[2])))
    elseif ex.head == :hcat # 1 x n
        s1 = 1
        s2 = length(ex.args)
        return esc(Expr(:call, MMatrix{s1, s2}, Expr(:tuple, ex.args...)))
    elseif ex.head == :typed_hcat # typed, 1 x n
        s1 = 1
        s2 = length(ex.args) - 1
        return esc(Expr(:call, Expr(:curly, :MMatrix, s1, s2, ex.args[1]), Expr(:tuple, ex.args[2:end]...)))
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
            return esc(Expr(:call, MMatrix{s1, s2}, Expr(:tuple, exprs...)))
        else # n x 1
            return esc(Expr(:call, MMatrix{length(ex.args), 1}, Expr(:tuple, ex.args...)))
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
            return esc(Expr(:call, Expr(:curly, :MMatrix,s1, s2, ex.args[1]), Expr(:tuple, exprs...)))
        else # typed, n x 1
            return esc(Expr(:call, Expr(:curly, :MMatrix, length(ex.args)-1, 1, ex.args[1]), Expr(:tuple, ex.args[2:end]...)))
        end
    else
        error("Bad input for @MMatrix")
    end
end
