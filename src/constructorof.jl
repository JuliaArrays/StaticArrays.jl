struct SArrayConstructor{S,N,L} end
struct MArrayConstructor{S,N,L} end
struct SizedArrayConstructor{S,N,M} end

(::SArrayConstructor{S,N,L})(data::NTuple{L,T}) where {S,T,N,L} = SArray{S,T,N,L}(data)
(::MArrayConstructor{S,N,L})(data::NTuple{L,T}) where {S,T,N,L} = MArray{S,T,N,L}(data)
(::SizedArrayConstructor{S,N,M})(data::AbstractArray{T,M}) where {S,T,N,M} = 
    SizedArray{S,T,N,M}(data)

ConstructionBase.constructorof(sa::Type{<:SArray{S,<:Any,N,L}}) where {S,N,L} = 
    SArrayConstructor{S,N,L}()
ConstructionBase.constructorof(sa::Type{<:MArray{S,<:Any,N,L}}) where {S,N,L} = 
    MArrayConstructor{S,N,L}()
ConstructionBase.constructorof(sa::Type{<:SizedArray{S,<:Any,N,M}}) where {S,N,M} = 
    SizedArrayConstructor{S,N,M}()
