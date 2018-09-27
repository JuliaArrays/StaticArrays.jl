# Special case flatten of iterators of static arrays.
import Base.Iterators: flatten_iteratorsize, flatten_length
flatten_iteratorsize(::Union{Base.HasShape, Base.HasLength}, ::Type{<:Union{SArray,MArray}}) = Base.HasLength()
function flatten_length(f, T::Type{<:Union{SArray,MArray}})
    length(T)*length(f.it)
end
