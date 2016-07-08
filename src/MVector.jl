type MVector{S, T} <: StaticVector{T}
    data::NTuple{S, T}

    function MVector(in)
        new(in)
    end

    function MVector()
        new()
    end
end

@inline (::Type{MVector}){S}(x::NTuple{S}) = MVector{S}(x)
@inline (::Type{MVector{S}}){S, T}(x::NTuple{S,T}) = MVector{S,T}(x)
@inline (::Type{MVector{S}}){S, T <: Tuple}(x::T) = MVector{S,promote_tuple_eltype(T)}(x)

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
        return Expr(:call, MVector{length(ex.args)}, Expr(:tuple, ex.args...))
    else
        error("Use @MVector [a,b,c] or @MVector([a,b,c])")
    end
end
