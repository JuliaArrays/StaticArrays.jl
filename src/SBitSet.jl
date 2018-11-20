struct SBitSet{N}<:AbstractSet{Int64}
    chunks::SVector{N,UInt64}
end

@inline function Base.iterate(B::SBitSet{N}) where N
    N==0 && return nothing
    return iterate(B, (1, @inbounds B.chunks[1]))
end

@inline function Base.iterate(B::SBitSet{N}, s) where N
    N==0 && return nothing
    i1, c = s
    while c==0
        i1 % UInt >= N % UInt && return nothing
        i1 += 1
        @inbounds c = B.chunks[i1]
    end
    tz = trailing_zeros(c) + 1
    c = _blsr(c)
    return ((i1-1)<<6 + tz, (i1, c))
end

@inline Base.isempty(B::SBitSet) = iszero(B.chunks)
@inline Base.length(B::SBitSet) = count_ones(B.chunks)
@inline Base.union(B::SBitSet, C::SBitSet) = SBitSet(B.chunks | C.chunks)
@inline Base.intersect(B::SBitSet, C::SBitSet) = SBitSet(B.chunks & C.chunks)
@inline Base.symdiff(B::SBitSet, C::SBitSet) = SBitSet(xor(B.chunks, C.chunks))
@inline Base.setdiff(B::SBitSet, C::SBitSet) = SBitSet(B.chunks & ~C.chunks)
@inline Base.issubset(B::SBitSet, C::SBitSet) = iszero(B.chunks & ~C.chunks)

@inline function Base.in(B::SBitSet{N}, k::Integer) where N
    (0%UInt < k%UInt <= (N<<6)) || return false
    i1,i2 = Base.get_chunks_id(k)
    return !iszero(B.chunks[i1] & (1<<i2))
end

Base.show(io::IO, ::MIME{Symbol("text/plain")}, B::SBitSet) = show(io, B)
Base.show(io::IO, B::SBitSet{N}) where N = print(io, "SBitSet{$N}($(collect(B)))")
