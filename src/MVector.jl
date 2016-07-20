type MVector{S, T} <: StaticVector{T}
    data::NTuple{S, T}

    function MVector(in::NTuple{S, T})
        new(in)
    end

    function MVector(in::NTuple{S})
        new(convert_ntuple(T,in))
    end

    function MVector(in::T)
        new((in,))
    end

    function MVector()
        new()
    end
end

@inline (::Type{MVector}){S}(x::NTuple{S}) = MVector{S}(x)
@inline (::Type{MVector{S}}){S, T}(x::NTuple{S,T}) = MVector{S,T}(x)
@inline (::Type{MVector{S}}){S, T <: Tuple}(x::T) = MVector{S,promote_tuple_eltype(T)}(x)

# Some more advanced constructor-like functions
@inline zeros{N}(::Type{MVector{N}}) = zeros(MVector{N,Float64})
@inline ones{N}(::Type{MVector{N}}) = ones(MVector{N,Float64})

#####################
## MVector methods ##
#####################

@pure size{S}(::Union{MVector{S},Type{MVector{S}}}) = (S, )
@pure size{S,T}(::Type{MVector{S,T}}) = (S,)

@propagate_inbounds function getindex(v::MVector, i::Integer)
    v.data[i]
end

# Mutating setindex!
@propagate_inbounds setindex!{S,T}(v::MVector{S,T}, val, i::Integer) = setindex!(v, convert(T, val), i)
@inline function setindex!{S,T}(v::MVector{S,T}, val::T, i::Integer)
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

@inline Tuple(v::MVector) = v.data

macro MVector(ex)
    if isa(ex, Expr) && ex.head == :vect
        return esc(Expr(:call, MVector{length(ex.args)}, Expr(:tuple, ex.args...)))
    elseif isa(ex, Expr) && ex.head == :ref
        return esc(Expr(:call, Expr(:curly, :MVector, length(ex.args[2:end]), ex.args[1]), Expr(:tuple, ex.args[2:end]...)))
    elseif isa(ex, Expr) && ex.head == :comprehension
        if length(ex.args) != 1 || !isa(ex.args[1], Expr) || ex.args[1].head != :generator
            error("Expected generator in comprehension, e.g. [f(i) for i = 1:3]")
        end
        ex = ex.args[1]
        if length(ex.args) != 2
            error("Use a one-dimensional comprehension for @MVector")
        end

        rng = eval(current_module(), ex.args[2].args[2])
        f = gensym()
        f_expr = :($f = ($(ex.args[2].args[1]) -> $(ex.args[1])))
        exprs = [:($f($j)) for j in rng]

        return quote
            $(Expr(:meta, :inline))
            $(esc(f_expr))
            $(esc(Expr(:call, Expr(:curly, :MVector, length(rng)), Expr(:tuple, exprs...))))
        end
    elseif isa(ex, Expr) && ex.head == :typed_comprehension
        if length(ex.args) != 2 || !isa(ex.args[2], Expr) !! ex.args[2].head != :generator
            error("Expected generator in typed comprehension, e.g. Float64[f(i) for i = 1:3]")
        end
        T = ex.args[1]
        ex = ex.args[2]
        if length(ex.args) != 2
            error("Use a one-dimensional comprehension for @MVector")
        end

        rng = eval(current_module(), ex.args[2].args[2])
        f = gensym()
        f_expr = :($f = ($(ex.args[2].args[1]) -> $(ex.args[1])))
        exprs = [:($f($j)) for j in rng]

        return quote
            $(Expr(:meta, :inline))
            $(esc(f_expr))
            $(esc(Expr(:call, Expr(:curly, :MVector, length(rng), T), Expr(:tuple, exprs...))))
        end
    else
        error("Use @MVector [a,b,c] or @MVector([a,b,c])")
    end
end
