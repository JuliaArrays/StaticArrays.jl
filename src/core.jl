# These arrays are AbstractArrays
abstract StaticArray{Sizes,T,N} <: AbstractArray{T, N}

##############################################################
## An immutable array - can't be changed after construction ##
##############################################################

immutable SArray{Sizes,T,N,D} <: StaticArray{Sizes,T,N}
    data::D

    function SArray(d)
        check_parameters(Val{Sizes}, T, Val{N}, D)
        new(convert_ntuple(T, d))
    end

    function SArray() # Only useful with MArray...
        error("Immutable SArray must be initialized with data. Supply data or consider using MArray.")
    end
end


###############################################################################
## A mutable version - we can use pointers to the heap to define setindex!() ##
## Also very useful for behaving more AbstractArray-like in construction...  ##
## just grab some RAM and call setindex.                                     ##
###############################################################################
type MArray{Sizes,T,N,D} <: StaticArray{Sizes,T,N}
    data::D

    function MArray(d)
        check_parameters(Val{Sizes}, T, Val{N}, D)
        new(convert_ntuple(T, d))
    end

    function MArray()
        check_parameters(Val{Sizes}, T, Val{N}, D)
        new()
    end
end

# Make sure that SArray's and MArray's type parameters are correctly formed
@generated function check_parameters{Sizes,T,N,D}(::Type{Val{Sizes}},::Type{T},::Type{Val{N}},::Type{D})
    if !isa(N,Int) && N >= 0
        str = "Expected non-negative dimensionality, got N = $N"
    end
    if !isa(Sizes,NTuple{N,Int})
        str = "Expected a $N-tuple of sizes, got size = $Sizes"
        return :(error($str))
    else
        for i = 1:N
            if Sizes[i] < 1
                str = "Dimension $i of size $Sizes is non-positive"
                return :(error($str))
            end
        end
    end

    numel = prod(Sizes)
    if D <: NTuple{numel,T} || (N == 0 && D == T)
        return nothing
    else
        str = "Data storage incorrect - was expecting NTuple{$numel, $T}, but got $D"
        return :(error($str))
    end
end


####################
## SArray methods ##
####################

# Make construction less painful (TODO automatic type promotion)
Base.call{Sizes,T,N,M}(::Type{SArray{Sizes,T,N}}, d::NTuple{M,T}) = SArray{Sizes,T,N,NTuple{M,T}}(d)
@generated function Base.call{Sizes,T,M}(::Type{SArray{Sizes,T}}, d::NTuple{M,T})
    N = length(Sizes)
    :($(SArray{Sizes,T,N,NTuple{M,T}})(d))
end
@generated function Base.call{Sizes,T,M}(::Type{SArray{Sizes}}, d::NTuple{M,T})
    N = length(Sizes)
    :($(SArray{Sizes,T,N,NTuple{M,T}})(d))
end
@generated function Base.call{T,M}(::Type{SArray}, d::NTuple{M,T})
    N = 1
    Sizes = (M,)
    :($(SArray{Sizes,T,N,NTuple{M,T}})(d))
end

# Type-promotion versions
@generated function Base.call{Sizes}(::Type{SArray{Sizes}}, d::Tuple)
    T = promote_eltype(d)
    N = length(Sizes)
    M = prod(Sizes)
    :($(SArray{Sizes,T,N,NTuple{M,T}})(d))
end
@generated function Base.call(::Type{SArray}, d::Tuple)
    T = promote_eltype(d)
    M = length(d.parameters)
    N = 1
    Sizes = (M,)
    :($(SArray{Sizes,T,N,NTuple{M,T}})(d))
end

# Uninitialized arrays are only useful with MArray... but give a friendly error message
function Base.call{Sizes,T,N}(::Type{SArray{Sizes,T,N}})
    error("Immutable SArray must be initialized with data. Supply data or consider using MArray.")
end
@generated function Base.call{Sizes,T}(::Type{SArray{Sizes,T}})
    error("Immutable SArray must be initialized with data. Supply data or consider using MArray.")
end


####################
## MArray methods ##
####################

# Make construction less painful (TODO automatic type promotion)
Base.call{Sizes,T,N,M}(::Type{MArray{Sizes,T,N}}, d::NTuple{M,T}) = MArray{Sizes,T,N,NTuple{M,T}}(d)
@generated function Base.call{Sizes,T,M}(::Type{MArray{Sizes,T}}, d::NTuple{M,T})
    N = length(Sizes)
    :($(MArray{Sizes,T,N,NTuple{M,T}})(d))
end
@generated function Base.call{Sizes,T,M}(::Type{MArray{Sizes}}, d::NTuple{M,T})
    N = length(Sizes)
    :($(MArray{Sizes,T,N,NTuple{M,T}})(d))
end
@generated function Base.call{T,M}(::Type{MArray}, d::NTuple{M,T})
    N = 1
    Sizes = (M,)
    :($(MArray{Sizes,T,N,NTuple{M,T}})(d))
end

# Type-promotion versions
@generated function Base.call{Sizes}(::Type{MArray{Sizes}}, d::Tuple)
    T = promote_eltype(d)
    N = length(Sizes)
    M = prod(Sizes)
    :($(MArray{Sizes,T,N,NTuple{M,T}})(d))
end
@generated function Base.call(::Type{MArray}, d::Tuple)
    T = promote_eltype(d)
    M = length(d.parameters)
    N = 1
    Sizes = (M,)
    :($(MArray{Sizes,T,N,NTuple{M,T}})(d))
end

# Same for uninitialized arrays (only useful with MArray?)
@generated function Base.call{Sizes,T,N}(::Type{MArray{Sizes,T,N}})
    M = prod(Sizes)
    :($(MArray{Sizes,T,N,NTuple{M,T}})())
end
@generated function Base.call{Sizes,T}(::Type{MArray{Sizes,T}})
    N = length(Sizes)
    M = prod(Sizes)
    :($(MArray{Sizes,T,N,NTuple{M,T}})())
end

#######################
## convert() methods ##
#######################

# Converting element types
Base.convert{Sizes,T1,T2,N,D}(::Type{SArray{Sizes,T1}}, a::SArray{Sizes,T2,N,D}) = SArray{Sizes,T1,N,D}(a.data)
Base.convert{Sizes,T1,T2,N,D}(::Type{SArray{Sizes,T1,N}}, a::SArray{Sizes,T2,N,D}) = SArray{Sizes,T1,N,D}(a.data)
Base.convert{Sizes,T1,T2,N,D1,D2}(::Type{SArray{Sizes,T1,N,D1}}, a::SArray{Sizes,T2,N,D2}) = SArray{Sizes,T1,N,D1}(a.data)

Base.convert{Sizes,T1,T2,N,D}(::Type{MArray{Sizes,T1}}, a::MArray{Sizes,T2,N,D}) = MArray{Sizes,T1,N,D}(a.data)
Base.convert{Sizes,T1,T2,N,D}(::Type{MArray{Sizes,T1,N}}, a::MArray{Sizes,T2,N,D}) = MArray{Sizes,T1,N,D}(a.data)
Base.convert{Sizes,T1,T2,N,D1,D2}(::Type{MArray{Sizes,T1,N,D1}}, a::MArray{Sizes,T2,N,D2}) = MArray{Sizes,T1,N,D1}(a.data)

# Conversion between MArray's and SArrays
Base.convert{Sizes,T,N,D}(::Type{MArray}, a::SArray{Sizes,T,N,D}) = MArray{Sizes,T,N,D}(a.data)
Base.convert{Sizes,T,N,D}(::Type{MArray{Sizes}}, a::SArray{Sizes,T,N,D}) = MArray{Sizes,T,N,D}(a.data)
Base.convert{Sizes,T1,T2,N,D}(::Type{MArray{Sizes,T1}}, a::SArray{Sizes,T2,N,D}) = MArray{Sizes,T1,N,D}(a.data)
Base.convert{Sizes,T1,T2,N,D}(::Type{MArray{Sizes,T1,N}}, a::SArray{Sizes,T2,N,D}) = MArray{Sizes,T1,N,D}(a.data)
Base.convert{Sizes,T1,T2,N,D1,D2}(::Type{MArray{Sizes,T1,N,D1}}, a::SArray{Sizes,T2,N,D2}) = MArray{Sizes,T1,N,D1}(a.data)

Base.convert{Sizes,T,N,D}(::Type{SArray}, a::MArray{Sizes,T,N,D}) = SArray{Sizes,T,N,D}(a.data)
Base.convert{Sizes,T,N,D}(::Type{SArray{Sizes}}, a::MArray{Sizes,T,N,D}) = SArray{Sizes,T,N,D}(a.data)
Base.convert{Sizes,T1,T2,N,D}(::Type{SArray{Sizes,T1}}, a::MArray{Sizes,T2,N,D}) = SArray{Sizes,T1,N,D}(a.data)
Base.convert{Sizes,T1,T2,N,D}(::Type{SArray{Sizes,T1,N}}, a::MArray{Sizes,T2,N,D}) = SArray{Sizes,T1,N,D}(a.data)
Base.convert{Sizes,T1,T2,N,D1,D2}(::Type{SArray{Sizes,T1,N,D1}}, a::MArray{Sizes,T2,N,D2}) = SArray{Sizes,T1,N,D1}(a.data)
