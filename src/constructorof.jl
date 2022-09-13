# keep the size when reconstructing arrays
# eltype can be different
constructorof(sa::Type{<:SArray{S}}) where {S} = SArray{S}
constructorof(sa::Type{<:MArray{S}}) where {S} = MArray{S}

# don't keep neither size nor eltype for vectors:
# both are unambiguously determined by the values
constructorof(::Type{<:SVector}) = SVector
constructorof(::Type{<:MVector}) = MVector
