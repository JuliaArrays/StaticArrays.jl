
@inline (::Type{FA})(x::Tuple) where {FA <: FieldArray} = construct_type(FA, x)(x...)

function construct_type(::Type{FA}, x) where {FA <: FieldArray}
    has_size(FA) || error("$FA has no static size!")
    length_match_size(FA, x)
    return adapt_eltype(FA, x)
end

@propagate_inbounds getindex(a::FieldArray, i::Int) = getfield(a, i)
@propagate_inbounds setindex!(a::FieldArray, x, i::Int) = (setfield!(a, i, x); a)

Base.cconvert(::Type{<:Ptr}, a::FieldArray) = Base.RefValue(a)
Base.unsafe_convert(::Type{Ptr{T}}, m::Base.RefValue{FA}) where {N,T,D,FA<:FieldArray{N,T,D}} =
    Ptr{T}(Base.unsafe_convert(Ptr{FA}, m))

function similar_type(::Type{A}, ::Type{T}, S::Size) where {T,A<:FieldArray}
    # We can preserve FieldArrays in array operations which do not change their `Size` and `eltype`.
    has_eltype(A) && eltype(A) === T && has_size(A) && Size(A) === S && return A
    # FieldArrays with parametric `eltype` would be adapted to the new `eltype` automatically.
    A′ = Base.typeintersect(base_type(A), StaticArray{Tuple{Tuple(S)...},T,length(S)})
    # But extra parameters are disallowed here. Also we check `fieldtypes` to make sure the result is valid.
    isconcretetype(A′) && fieldtypes(A′) === ntuple(_ -> T, Val(prod(S))) && return A′
    # Otherwise, we fallback to `S/MArray` based on it's mutability.
    if ismutabletype(A)
        return mutable_similar_type(T, S, length_val(S))
    else
        return default_similar_type(T, S, length_val(S))
    end
end

# return `Union{}` for Union Type. Otherwise return the constructor with no parameters.
@pure base_type(@nospecialize(T::Type)) = (T′ = Base.unwrap_unionall(T);
T′ isa DataType ? T′.name.wrapper : Union{})
if VERSION < v"1.8"
    fieldtypes(::Type{T}) where {T} = ntuple(i -> fieldtype(T, i), Val(fieldcount(T)))
    @eval @pure function ismutabletype(@nospecialize(T::Type))
        T′ = Base.unwrap_unionall(T)
        T′ isa DataType && $(VERSION < v"1.7" ? :(T′.mutable) : :(T′.name.flags & 0x2 == 0x2))
    end
end
