#####################
## MVector methods ##
#####################
"""
    @MVector [a, b, c, d]
    @MVector [i for i in 1:2]
    @MVector ones(2)

A convenience macro to construct `MVector`.
See [`@SArray`](@ref) for detailed features.
"""
macro MVector(ex)
    static_vector_gen(MVector, ex, __module__)
end

# Named field access for the first four elements, using the conventional field
# names from low-dimensional geometry (x,y,z) and computer graphics (w).
let dimension_names = QuoteNode.([:x, :y, :z, :w])
    body = :(getfield(v, name))
    for (i,dim_name) in enumerate(dimension_names)
        @eval @inline function Base.propertynames(v::Union{SVector{$i},MVector{$i}}, private::Bool = false)
            named_dims = ($(first(dimension_names, i)...),)
            private ? (named_dims..., :data) : named_dims
        end

        body = :(name === $(dimension_names[i]) ? getfield(v, :data)[$i] : $body)
        @eval @inline function Base.getproperty(v::Union{SVector{$i},MVector{$i}},
                                                name::Symbol)
            $body
        end
    end

    body = :(setfield!(v, name, e))
    for (i,dim_name) in enumerate(dimension_names)
        body = :(name === $dim_name ? @inbounds(v[$i] = e) : $body)
        @eval @inline function Base.setproperty!(v::MVector{$i}, name::Symbol, e)
            $body
        end
    end
end

# for longer S/MVectors and other S/MArrays, the only property is data, and it's private
Base.propertynames(::Union{SArray,MArray}, private::Bool = false) = private ? (:data,) : ()
