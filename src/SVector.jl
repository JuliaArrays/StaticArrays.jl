"""
    SVector{S, T}(x::NTuple{S, T})
    SVector{S, T}(x1, x2, x3, ...)

Construct a statically-sized vector `SVector`. Since this type is immutable,
the data must be provided upon construction and cannot be mutated later.
Constructors may drop the `T` and `S` parameters if they are inferrable from the
input (e.g. `SVector(1,2,3)` constructs an `SVector{3, Int}`).

    SVector{S}(vec::Vector)

Construct a statically-sized vector of length `S` using the data from `vec`.
The parameter `S` is mandatory since the length of `vec` is unknown to the
compiler (the element type may optionally also be specified).
"""
const SVector{S, T} = SArray{Tuple{S}, T, 1, S}

@inline (::Type{SVector}){S}(x::NTuple{S,Any}) = SVector{S}(x)
@inline (::Type{SVector{S}}){S, T}(x::NTuple{S,T}) = SVector{S,T}(x)
@inline (::Type{SVector{S}}){S, T <: Tuple}(x::T) = SVector{S,promote_tuple_eltype(T)}(x)

# conversion from AbstractVector / AbstractArray (better inference than default)
#@inline convert{S,T}(::Type{SVector{S}}, a::AbstractArray{T}) = SVector{S,T}((a...))

# Simplified show for the type
show(io::IO, ::Type{SVector{N, T}}) where {N, T} = print(io, "SVector{$N,$T}")

# Some more advanced constructor-like functions
@inline zeros{N}(::Type{SVector{N}}) = zeros(SVector{N,Float64})
@inline ones{N}(::Type{SVector{N}}) = ones(SVector{N,Float64})

#####################
## SVector methods ##
#####################

@propagate_inbounds function getindex(v::SVector, i::Int)
    v.data[i]
end

@inline Tuple(v::SVector) = v.data

# See #53
Base.cconvert{T}(::Type{Ptr{T}}, v::SVector) = Ref(v)
Base.unsafe_convert{N,T}(::Type{Ptr{T}}, v::Ref{SVector{N,T}}) =
    Ptr{T}(Base.unsafe_convert(Ptr{SVector{N,T}}, v))

# Converting a CartesianIndex to an SVector
convert(::Type{SVector}, I::CartesianIndex) = SVector(I.I)
convert{N}(::Type{SVector{N}}, I::CartesianIndex{N}) = SVector{N}(I.I)
convert{N,T}(::Type{SVector{N,T}}, I::CartesianIndex{N}) = SVector{N,T}(I.I)

Base.promote_rule{N,T}(::Type{SVector{N,T}}, ::Type{CartesianIndex{N}}) = SVector{N,promote_type(T,Int)}

macro SVector(ex)
    if isa(ex, Expr) && ex.head == :vect
        return esc(Expr(:call, SVector{length(ex.args)}, Expr(:tuple, ex.args...)))
    elseif isa(ex, Expr) && ex.head == :ref
        return esc(Expr(:call, Expr(:curly, :SVector, length(ex.args[2:end]), ex.args[1]), Expr(:tuple, ex.args[2:end]...)))
    elseif isa(ex, Expr) && ex.head == :comprehension
        if length(ex.args) != 1 || !isa(ex.args[1], Expr) || ex.args[1].head != :generator
            error("Expected generator in comprehension, e.g. [f(i) for i = 1:3]")
        end
        ex = ex.args[1]
        if length(ex.args) != 2
            error("Use a one-dimensional comprehension for @SVector")
        end

        rng = eval(current_module(), ex.args[2].args[2])
        f = gensym()
        f_expr = :($f = ($(ex.args[2].args[1]) -> $(ex.args[1])))
        exprs = [:($f($j)) for j in rng]

        return quote
            $(esc(f_expr))
            $(esc(Expr(:call, Expr(:curly, :SVector, length(rng)), Expr(:tuple, exprs...))))
        end
    elseif isa(ex, Expr) && ex.head == :typed_comprehension
        if length(ex.args) != 2 || !isa(ex.args[2], Expr) || ex.args[2].head != :generator
            error("Expected generator in typed comprehension, e.g. Float64[f(i) for i = 1:3]")
        end
        T = ex.args[1]
        ex = ex.args[2]
        if length(ex.args) != 2
            error("Use a one-dimensional comprehension for @SVector")
        end

        rng = eval(current_module(), ex.args[2].args[2])
        f = gensym()
        f_expr = :($f = ($(ex.args[2].args[1]) -> $(ex.args[1])))
        exprs = [:($f($j)) for j in rng]

        return quote
            $(esc(f_expr))
            $(esc(Expr(:call, Expr(:curly, :SVector, length(rng), T), Expr(:tuple, exprs...))))
        end
    elseif isa(ex, Expr) && ex.head == :call
        if ex.args[1] == :zeros || ex.args[1] == :ones || ex.args[1] == :rand ||ex.args[1] == :randn
            if length(ex.args) == 2
                return quote
                    $(esc(ex.args[1]))(SVector{$(esc(ex.args[2]))})
                end
            elseif length(ex.args) == 3
                return quote
                    $(esc(ex.args[1]))(SVector{$(esc(ex.args[3])), $(esc(ex.args[2]))})
                end
            else
                error("@SVector expected a 1-dimensional array expression")
            end
        elseif ex.args[1] == :fill
            if length(ex.args) == 3
                return quote
                    $(esc(ex.args[1]))($(esc(ex.args[2])), SVector{$(esc(ex.args[3]))})
                end
            else
                error("@SVector expected a 1-dimensional array expression")
            end
        else
            error("@SVector only supports the zeros(), ones(), fill(), rand() and randn() functions.")
        end
    else # TODO Expr(:call, :zeros), Expr(:call, :ones), Expr(:call, :eye) ?
        error("Use @SVector [a,b,c], @SVector Type[a,b,c] or a comprehension like [f(i) for i = i_min:i_max]")
    end
end
