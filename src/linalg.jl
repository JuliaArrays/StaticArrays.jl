typealias SVector{Size,T,D} SArray{Size,T,1,D}
typealias MVector{Size,T,D} MArray{Size,T,1,D}
typealias StaticVector{Size,T,D} Union{SVector{Size,T,D},MVector{Size,T,D}}

typealias SMatrix{Sizes,T,D} SArray{Sizes,T,2,D}
typealias MMatrix{Sizes,T,D} MArray{Sizes,T,2,D}
typealias StaticMatrix{Size,T,D} Union{SMatrix{Size,T,D},MMatrix{Size,T,D}}

# Constructors not covered by SArray
@inline Base.call(::Type{SVector}, d::Tuple) = SVector(convert_ntuple(promote_tuple_eltype(d), d))
@generated Base.call{N,T}(::Type{SVector}, d::NTuple{N,T}) = :($(SVector{(N,),T,d})(d))
@inline Base.call{Size}(::Type{SVector{Size}}, d::Tuple) = SVector{Size}(convert_ntuple(promote_tuple_eltype(d), d))
@generated Base.call{Size,N,T}(::Type{SVector{Size}}, d::NTuple{N,T}) = :($(SVector{Size,T,d})(d))

@inline Base.call{Sizes}(::Type{SMatrix{Sizes}}, d::Tuple) = SMatrix{Sizes}(convert_ntuple(promote_tuple_eltype(d), d))
@generated Base.call{Sizes,N,T}(::Type{SMatrix{Sizes}}, d::NTuple{N,T}) = :($(SMatrix{Sizes,T,d})(d))

# Constructors not covered by MArray
@inline Base.call(::Type{MVector}, d::Tuple) = MVector(convert_ntuple(promote_tuple_eltype(d), d))
@generated Base.call{N,T}(::Type{MVector}, d::NTuple{N,T}) = :($(MVector{(N,),T,d})(d))
@inline Base.call{Size}(::Type{MVector{Size}}, d::Tuple) = MVector{Size}(convert_ntuple(promote_tuple_eltype(d), d))
@generated Base.call{Size,N,T}(::Type{MVector{Size}}, d::NTuple{N,T}) = :($(MVector{Size,T,d})(d))

@inline Base.call{Sizes}(::Type{MMatrix{Sizes}}, d::Tuple) = MMatrix{Sizes}(convert_ntuple(promote_tuple_eltype(d), d))
@generated Base.call{Sizes,N,T}(::Type{MMatrix{Sizes}}, d::NTuple{N,T}) = :($(MMatrix{Sizes,T,d})(d))


# Level-0 ops: Reductions *should* already work

# Level-1 ops: Mostly covered by map(), etc. Vector transpose... need operators for Julia 0.4.

# Level-2 ops: matrix-vector multiplication and matrix transpose

# Level-3 ops: matrix-matrix multiplication
