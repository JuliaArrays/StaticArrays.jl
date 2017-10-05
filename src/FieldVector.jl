"""
    abstract FieldVector{N, T} <: StaticVector{N, T}

Inheriting from this type will make it easy to create your own vector types. A `FieldVector`
will automatically define `getindex` and `setindex!` appropriately. An immutable
`FieldVector` will be as performant as an `SVector` of similar length and element type,
while a mutable `FieldVector` will behave similarly to an `MVector`.

For example:

    struct Point3D <: FieldVector{3, Float64}
        x::Float64
        y::Float64
        z::Float64
    end
"""
abstract type FieldVector{N, T} <: StaticVector{N, T} end

# Is this a good idea?? Should people just define constructors that accept tuples?
@inline (::Type{FV})(x::Tuple) where {FV <: FieldVector} = FV(x...)

@propagate_inbounds getindex(v::FieldVector, i::Int) = getfield(v, i)
@propagate_inbounds setindex!(v::FieldVector, x, i::Int) = setfield!(v, i, x)

# See #53
Base.cconvert(::Type{<:Ptr}, v::FieldVector) = Base.RefValue(v)
Base.unsafe_convert(::Type{Ptr{T}}, m::Base.RefValue{FV}) where {N,T,FV<:FieldVector{N,T}} =
    Ptr{T}(Base.unsafe_convert(Ptr{FV}, m))

# Anonymous FieldVector Macros

"""
    @FVector Type Names
    @FVector Type Names Values

Creates an anonymous immutable `FieldVector` with names determined from the `Names`
vector and values determined from the `Values` vector (if no values are provided,
it defaults to not setting the values like `similar`). All of the values are converted
to the type of the `Type` input.

For example:

    a = @FVector Float64 [a,b,c]
    b = @FVector Float64 [a,b,c] [1,2,3]
"""
macro FVector(T,_names,vals=nothing)
    names = Symbol.(_names.args)
    type_name = gensym(:FVector)
    construction_call = vals==nothing ?
    quote
        ($(type_name))()
    end : quote
        ($(type_name))($(vals)...)
    end

    quote
        struct $(type_name) <: FieldVector{$(length(names)),$T}
            $((:($n::$(T)) for n in names)...)
            $(type_name)() = new()
            $(type_name)($((:($n) for n in names)...)) = new($((:($n) for n in names)...))
        end
        $(esc(construction_call))
    end
end

"""
    @MFVector Type Names
    @MFVector Type Names Values

Creates an anonymous immutable `FieldVector` with names determined from the `Names`
vector and values determined from the `Values` vector (if no values are provided,
it defaults to not setting the values like `similar`). All of the values are converted
to the type of the `Type` input.

For example:

    a = @MFVector Float64 [a,b,c]
    b = @MFVector Float64 [a,b,c] [1,2,3]
"""
macro MFVector(T,_names,vals=nothing)
    names = Symbol.(_names.args)
    type_name = gensym(:MFVector)
    construction_call = vals==nothing ?
    quote
        ($(type_name))()
    end : quote
        ($(type_name))($(vals)...)
    end

    quote
        mutable struct $(type_name) <: FieldVector{$(length(names)),$T}
            $((:($n::$(T)) for n in names)...)
            $(type_name)() = new()
            $(type_name)($((:($n) for n in names)...)) = new($((:($n) for n in names)...))
        end
        $(esc(construction_call))
    end
end
