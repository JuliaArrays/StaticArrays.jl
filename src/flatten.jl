# Special case flatten of iterators of static arrays.
import Base.Iterators: flatten_iteratorsize, flatten_length
flatten_iteratorsize(::Union{Base.HasShape, Base.HasLength}, ::Type{<:StaticArray{S}}) where {S} = Base.HasLength()
function flatten_length(f, T::Type{<:StaticArray{S}}) where {S}
    length(T)*length(f.it)
end
