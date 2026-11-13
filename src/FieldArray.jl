
@inline (::Type{FA})(x::Tuple) where {FA <: FieldArray} = construct_type(FA, x)(x...)

function construct_type(::Type{FA}, x) where {FA <: FieldArray}
    has_size(FA) || error("$FA has no static size!")
    length_match_size(FA, x)
    FA′ = adapt_eltype(FA, x)
    FA′ === FA && x isa Args && _missing_fa_constructor(FA, typeof(x.args))
    return FA′
end
@noinline function _missing_fa_constructor(@nospecialize(FA), @nospecialize(AT))
    Ts = join(("::$T" for T in fieldtypes(AT)), ", ")
    error("The constructor for $FA($Ts) is missing!")
end

@propagate_inbounds getindex(a::FieldArray, i::Int) = getfield(a, i)
@propagate_inbounds setindex!(a::FieldArray, x, i::Int) = (setfield!(a, i, x); a)

Base.cconvert(::Type{Ptr{T}}, a::FieldArray) where {T} = ref_pointer(T, Base.RefValue(a))
Base.cconvert(::Type{Ptr{T}}, r::Base.RefValue{<:FieldArray}) where {T} = ref_pointer(T, r)
ref_pointer(::Type{T}, r::Base.RefValue{<:FieldArray{N,T,D}}) where {N,T,D} = RefElements(r)

# We can automatically preserve FieldArrays in array operations which do not
# change their eltype or Size. This should cover all non-parametric FieldArray,
# but for those which are parametric on the eltype the user will still need to
# overload similar_type themselves.
similar_type(::Type{A}, ::Type{T}, S::Size) where {N, T, A<:FieldArray{N, T}} =
    _fieldarray_similar_type(A, T, S, Size(A))

# Extra layer of dispatch to match NewSize and OldSize
_fieldarray_similar_type(A, T, NewSize::S, OldSize::S) where {S} = A
_fieldarray_similar_type(A, T, NewSize, OldSize) =
    default_similar_type(T, NewSize, length_val(NewSize))

# Convenience constructors for NamedTuple types 
Base.NamedTuple(array::FieldArray) = Base.NamedTuple{propertynames(array)}(array)
