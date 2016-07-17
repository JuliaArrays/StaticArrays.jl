immutable SVector{S, T} <: StaticVector{T}
    data::NTuple{S, T}
end

@inline (::Type{SVector}){S}(x::NTuple{S}) = SVector{S}(x)
@inline (::Type{SVector{S}}){S, T}(x::NTuple{S,T}) = SVector{S,T}(x)
@inline (::Type{SVector{S}}){S, T <: Tuple}(x::T) = SVector{S,promote_tuple_eltype(T)}(x)

# conversion from AbstractVector / AbstractArray (better inference than default)
#@inline convert{S,T}(::Type{SVector{S}}, a::AbstractArray{T}) = SVector{S,T}((a...))


#####################
## SVector methods ##
#####################

@pure size{S}(::Union{SVector{S},Type{SVector{S}}}) = (S, )
@pure size{S,T}(::Type{SVector{S,T}}) = (S,)

@propagate_inbounds function getindex(v::SVector, i::Integer)
    v.data[i]
end

@inline Tuple(v::SVector) = v.data

macro SVector(ex)
    if isa(ex, Expr) && ex.head == :vect
        return esc(Expr(:call, SVector{length(ex.args)}, Expr(:tuple, ex.args...)))
    elseif isa(ex, Expr) && ex.head == :ref
        return esc(Expr(:call, Expr(:curly, :SVector, length(ex.args[2:end]), ex.args[1]), Expr(:tuple, ex.args[2:end]...)))
    else
        error("Use @SVector [a,b,c] or @SVector([a,b,c])")
    end
end
