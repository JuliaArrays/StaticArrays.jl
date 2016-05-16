#=
# This map returns an SArray
@generated function Base.map{Sizes}(f::Function, a::SArray{Sizes}...)

end

# This map has at least one MArray so returns an MArray using Base's map
@generated function Base.map{Sizes}(f::Function, a::StaticArray{Sizes}...)
    exprs = Vector{Expr}(length(a))
    for i = 1:length(a)
        if isa(a,MArray)
            exprs[i] = :(a[:i])
        else
            exprs[i] = :(MArray(a[:i]))
        end
    end
    return :($(Expr(:call,:map,:f,exprs...)))
end
=#
