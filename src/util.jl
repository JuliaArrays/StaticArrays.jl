# For convenience
TupleN{T,N} = NTuple{N,T}

# Cast any Tuple to an TupleN{T}
@inline convert_ntuple{T}(::Type{T},d::T) = T # For zero-dimensional arrays
@inline convert_ntuple{N,T}(::Type{T},d::NTuple{N,T}) = d
@generated function convert_ntuple{N,T}(::Type{T}, d::NTuple{N,Any})
    exprs = ntuple(i -> :(convert(T, d[$i])), Val{N})
    return quote
        $(Expr(:meta, :inline))
        $(Expr(:tuple, exprs...))
    end
end

# Base gives up on tuples for promote_eltype... (TODO can we improve Base?)
@generated function promote_tuple_eltype{T <: Tuple}(::Union{T,Type{T}})
    t = Union{}
    for i = 1:length(T.parameters)
        tmp = T.parameters[i]
        if tmp <: Vararg
            tmp = tmp.parameters[1]
        end
        t = promote_type(t, tmp)
    end
    return quote
        $(Expr(:meta,:pure))
        $t
    end
end

# The ::Tuple variants exist to make sure that anything that calls with a tuple
# instead of a Tuple gets through to the constructor, so the user gets a nice
# error message
@pure tuple_length(T::Type{<:Tuple}) = length(T.parameters)
@pure tuple_length(T::Tuple) = length(T)
@pure tuple_prod(T::Type{<:Tuple}) = length(T.parameters) == 0 ? 1 : *(T.parameters...)
@pure tuple_prod(T::Tuple) = prod(T)
@pure tuple_minimum(T::Type{<:Tuple}) = length(T.parameters) == 0 ? 0 : minimum(tuple(T.parameters...))
@pure tuple_minimum(T::Tuple) = minimum(T)

# Something doesn't match up type wise
function check_array_parameters(Size, T, N, L)
    (!isa(Size, DataType) || (Size.name !== Tuple.name)) && throw(ArgumentError("Static Array parameter Size must be a Tuple type, got $Size"))
    !isa(T, Type) && throw(ArgumentError("Static Array parameter T must be a type, got $T"))
    !isa(N.parameters[1], Int) && throw(ArgumenError("Static Array parameter N must be an integer, got $(N.parameters[1])"))
    !isa(L.parameters[1], Int) && throw(ArgumentError("Static Array parameter L must be an integer, got $(L.parameters[1])"))
    # shouldn't reach here. Anything else should have made it to the function below
    error("Internal error. Please file a bug")
end

@generated function check_array_parameters{Size,T,N,L}(::Type{Size}, ::Type{T}, ::Type{Val{N}}, ::Type{Val{L}})
    if !all(x->isa(x, Int), Size.parameters)
        return :(throw(ArgumentError("Static Array parameter Size must be a tuple of Ints (e.g. `SArray{Tuple{3,3}}` or `SMatrix{3,3}`).")))
    end

    if L != tuple_prod(Size) || L < 0 || tuple_minimum(Size) < 0 || tuple_length(Size) != N
        return :(throw(ArgumentError("Size mismatch in Static Array parameters. Got size $Size, dimension $N and length $L.")))
    end

    return nothing
end
