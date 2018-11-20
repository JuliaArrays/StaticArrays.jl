struct SVecMat{N, T, Mtyp<:AbstractMatrix{T}} <: AbstractMatrix{T}
    parent::Mtyp
end

Base.@propagate_inbounds function SVecMat(::Val{N}, mat) where N 
    @boundscheck begin 
        size(mat, 1) != N && throw(DimensionMismatch("Cannot construct N=$N dimensional SvecMat from s=$(size(mat,1)) columns"))
    end
    return SVecMat{N}(mat)
end

@inline SVecMat(mat::AbstractArray{T}) where T = SVecMat{size(mat,1), T, typeof(mat)}(mat)

Base.@propagate_inbounds Base.getindex(M::SVecMat, idxs...) = getindex(M.parent, idxs...)
Base.@propagate_inbounds Base.setindex!(M::SVecMat, v, idxs...) = setindex!(M.parent, v, idxs...)
@inline Base.length(M::SVecMat) = length(M.parent)
@inline Base.size(M::SVecMat{N}) where N = (N, size(M.parent, 2) )
Base.IndexStyle(::SVecMat{N, T, Mtyp}) where {N,T,Mtyp} = IndexStyle(Mtyp)
Base.IndexStyle(::Type{SVecMat{N, T, Mtyp}}) where {N,T,Mtyp} = IndexStyle(Mtyp)
Base.eltype(M::SVecMat) = eltype(M.parent)


@inline function Base.getindex(M::SVecMat{N}, ::Colon, j::Integer) where N
    @boundscheck checkbounds(M, 1, j)
    return SVector(ntuple(i-> (@inbounds M.parent[i,j]) , N))
end

@inline function Base.setindex!(M::SVecMat{N,T}, v::SVector{N,T}, ::Colon, j::Integer) where {N,T}
    @boundscheck checkbounds(M, 1, j)
    for i=1:N
        @inbounds M[i,j] = v[i]
    end
    v
end



Base.@propagate_inbounds function getbit(B::SVecMat{N,UInt64}, i) where N
    i1,i2 = Base.get_chunks_id(i)
    return !iszero(B.parent[i1] & (1<<i2))
end

Base.@propagate_inbounds function getbit(B::SVecMat{N,UInt64}, i, j) where N
    i1,i2 = Base.get_chunks_id(i)
    return !iszero(B.parent[i1,j] & (1<<i2))
end

Base.@propagate_inbounds function setbit!(B::SVecMat{N,UInt64}, b::Bool, i) where N
    i1, i2 = Base.get_chunks_id(i)
    u = UInt64(1) << i2
    c = B.parent[i1]
    B.parent[i1] = ifelse(b, c | u, c & ~u)
    b
end

Base.@propagate_inbounds function setbit!(B::SVecMat{N,UInt64}, b::Bool, i, j) where N
    i1, i2 = Base.get_chunks_id(i)
    u = UInt64(1) << i2
    c = B.parent[i1, j]
    B.parent[i1, j] = ifelse(b, c | u, c & ~u)
    b
end