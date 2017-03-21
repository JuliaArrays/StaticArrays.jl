"""
    MVector{S,T}()
    MVector{S,T}(x::NTuple{S, T})
    MVector{S,T}(x1, x2, x3, ...)

Construct a statically-sized, mutable vector `MVector`. Data may optionally be
provided upon construction, and can be mutated later. Constructors may drop the
`T` and `S` parameters if they are inferrable from the input (e.g.
`MVector(1,2,3)` constructs an `MVector{3, Int}`).

    MVector{S}(vec::Vector)

Construct a statically-sized, mutable vector of length `S` using the data from
`vec`. The parameter `S` is mandatory since the length of `vec` is unknown to the
compiler (the element type may optionally also be specified).
"""
type MVector{S, T} <: StaticVector{T}
    data::NTuple{S, T}

    function (::Type{MVector{S,T}}){S,T}(in::NTuple{S, T})
        new{S,T}(in)
    end

    function (::Type{MVector{S,T}}){S,T}(in::NTuple{S, Any})
        new{S,T}(convert_ntuple(T,in))
    end

    function (::Type{MVector{S,T}}){S,T}(in::T)
        new{S,T}((in,))
    end

    function (::Type{MVector{S,T}}){S,T}()
        new{S,T}()
    end
end

@inline (::Type{MVector}){S}(x::NTuple{S,Any}) = MVector{S}(x)
@inline (::Type{MVector{S}}){S, T}(x::NTuple{S,T}) = MVector{S,T}(x)
@inline (::Type{MVector{S}}){S, T <: Tuple}(x::T) = MVector{S,promote_tuple_eltype(T)}(x)

# Some more advanced constructor-like functions
@inline zeros{N}(::Type{MVector{N}}) = zeros(MVector{N,Float64})
@inline ones{N}(::Type{MVector{N}}) = ones(MVector{N,Float64})

#####################
## MVector methods ##
#####################

@pure Size{S}(::Type{MVector{S}}) = Size(S)
@pure Size{S,T}(::Type{MVector{S,T}}) = Size(S)

@propagate_inbounds function getindex(v::MVector, i::Int)
    v.data[i]
end

# Mutating setindex!
@propagate_inbounds setindex!{S,T}(v::MVector{S,T}, val, i::Int) = setindex!(v, convert(T, val), i)
@inline function setindex!{S,T}(v::MVector{S,T}, val::T, i::Int)
    @boundscheck if i < 1 || i > length(v)
        throw(BoundsError())
    end

    if isbits(T)
        unsafe_store!(Base.unsafe_convert(Ptr{T}, Base.data_pointer_from_objref(v)), val, i)
    else
        # This one is unsafe (#27)
        #unsafe_store!(Base.unsafe_convert(Ptr{Ptr{Void}}, Base.data_pointer_from_objref(v.data)), Base.data_pointer_from_objref(val), i)
        error("setindex!() with non-isbits eltype is not supported by StaticArrays. Consider using SizedArray.")
    end

    return val
end

@inline Tuple(v::MVector) = v.data

@inline function Base.unsafe_convert{N,T}(::Type{Ptr{T}}, v::MVector{N,T})
    Base.unsafe_convert(Ptr{T}, Base.data_pointer_from_objref(v))
end

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
            $(esc(f_expr))
            $(esc(Expr(:call, Expr(:curly, :MVector, length(rng), T), Expr(:tuple, exprs...))))
        end
    elseif isa(ex, Expr) && ex.head == :call
        if ex.args[1] == :zeros || ex.args[1] == :ones || ex.args[1] == :rand ||ex.args[1] == :randn
            if length(ex.args) == 2
                return quote
                    $(esc(ex.args[1]))(MVector{$(esc(ex.args[2]))})
                end
            elseif length(ex.args) == 3
                return quote
                    $(esc(ex.args[1]))(MVector{$(esc(ex.args[3])), $(esc(ex.args[2]))})
                end
            else
                error("@MVector expected a 1-dimensional array expression")
            end
        elseif ex.args[1] == :fill
            if length(ex.args) == 3
                return quote
                    $(esc(ex.args[1]))($(esc(ex.args[2])), MVector{$(esc(ex.args[3]))})
                end
            else
                error("@MVector expected a 1-dimensional array expression")
            end
        else
            error("@MVector only supports the zeros(), ones(), fill(), rand() and randn() functions.")
        end
    else
        error("Use @MVector [a,b,c] or @MVector([a,b,c])")
    end
end
