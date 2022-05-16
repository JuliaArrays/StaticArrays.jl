"""
    MVector{S,T}(undef)
    MVector{S,T}(x::NTuple{S, T})
    MVector{S,T}(x1, x2, x3, ...)

Construct a statically-sized, mutable vector `MVector`. Data may optionally be
provided upon construction, and can be mutated later. Constructors may drop the
`T` and `S` parameters if they are inferrable from the input (e.g.
`MVector(1,2,3)` constructs an `MVector{3, Int}`).

    MVector{S}(vec::Vector)

Construct a statically-sized, mutable vector of length `S` using the data from
`vec`. The parameter `S` is mandatory since the length of `vec` is unknown to the
compiler (the element type may optionally also be specified).
"""
const MVector{S, T} = MArray{Tuple{S}, T, 1, S}

# Some more advanced constructor-like functions
@inline zeros(::Type{MVector{N}}) where {N} = zeros(MVector{N,Float64})
@inline ones(::Type{MVector{N}}) where {N} = ones(MVector{N,Float64})

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
    esc(static_vector_gen(MVector, ex, __module__))
end

# Named field access for the first four elements, using the conventional field
# names from low-dimensional geometry (x,y,z) and computer graphics (w).
let dimension_names = QuoteNode.([:x, :y, :z, :w])
    body = :(getfield(v, name))
    for (i,dim_name) in enumerate(dimension_names)
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
