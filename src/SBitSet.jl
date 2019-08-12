struct SBitSet{N}
    chunks::NTuple{N, UInt64}
end

mutable struct MBitSet{N}
    chunks::NTuple{N, UInt64}
end

const UBitSet = Union{SBitSet{N}, MBitSet{N}} where N

## Basic constructors
SBitSet(a::MBitSet) = SBitSet(a.chunks)
SBitSet(a::SBitSet) = a

MBitSet(a::SBitSet) = MBitSet(a.chunks)
MBitSet(a::MBitSet) = MBitSet(a.chunks)

zero(::Type{SBitSet{N}}) where N = SBitSet(ntuple(i->UInt64(0), Val{N}()))
zero(::Type{MBitSet{N}}) where N = MBitSet(ntuple(i->UInt64(0), Val{N}()))

function Base.empty!(a::MBitSet{N}) where N 
    a.chunks = ntuple(i->UInt64(0), Val{N}())
    a
end

@inline function iszero(a::UBitSet{N}) where N
    s = UInt64(0)
    @inbounds for i=1:N
        s |= a.chunks[i]
    end
    return iszero(s)
end
isempty(a::UBitSet) = iszero(a)

@inline function count_ones(a::UBitSet{N}) where N 
    s=0
    @inbounds for i=1:N 
        s+= count_ones(a.chunks[i])
    end
    s
end

## Iteration Protocol
_blsr(x) = x & (x-Int64(1))
length(a::UBitSet) = count_ones(a)
eltype(::UBitSet) = Int

@inline function iterate(a::UBitSet{N}) where N
    N > 0 || return nothing
    return iterate(a, (1, @inbounds a.chunks[1]))
end

@inline function iterate(a::UBitSet{N}, s) where N
    chunks = a.chunks
    i1, c = s
    while c==0
        i1 % UInt >= N % UInt && return nothing
        i1 += 1
        @inbounds c = chunks[i1]
    end
    tz = trailing_zeros(c) + 1
    c = _blsr(c)
    return ((i1-1)<<6 + tz, (i1, c))
end

## Bitwise operations
@inline Base.:&(L::UBitSet{N}, R::UBitSet{N}) where N =  SBitSet(ntuple(i->(L.chunks[i] & R.chunks[i]), Val{N}()))
@inline Base.:|(L::UBitSet{N}, R::UBitSet{N}) where N =  SBitSet(ntuple(i->(L.chunks[i] | R.chunks[i]), Val{N}()))
@inline xor(L::UBitSet{N}, R::UBitSet{N}) where N =  SBitSet(ntuple(i->xor(L.chunks[i], R.chunks[i]), Val{N}()))
@inline ~(a::UBitSet{N}) where N = SBitSet(ntuple(i->~a.chunks[i], Val{N}()))

## Broadcast support
broadcastable(a::UBitSet{N}) where N = Ref(SBitSet{N}(a.chunks))

@inline function materialize!(dst::MBitSet{N}, src::Base.Broadcast.Broadcasted{A, B, typeof(Base.identity),Tuple{Base.RefValue{SBitSet{N}}}} ) where {N,A,B}
    dst.chunks = src.args[1][].chunks
    dst
end

@inline function setindex!(dst::MBitSet{N}, src::UBitSet{N}) where N
    dst.chunks = src.chunks
    src
end
@inline function getindex(a::MBitSet{N}) where N
    return SBitSet(a)
end

## Bounds checking
@propagate_inbounds function _checkbounds(::UBitSet{N}, idx) where N
    @boundscheck begin
    0 < idx <= 64*N || throw(BoundsError(SBitSet{N}, idx))
    end
end

@propagate_inbounds function _checkbounds(::Type{Bool}, ::UBitSet{N}, idx) where N
    @boundscheck begin
    0 < idx <= 64*N || return false
    end
    true
end

## Individual bit access
@propagate_inbounds function in(idx, a::UBitSet{N}) where N
    i1,i2 = Base.get_chunks_id(idx)
    _checkbounds(Bool, a, idx) || return false
    @inbounds u = a.chunks[i1]
    return iszero(u & (1<< (i2 & 63)))
end

@propagate_inbounds function getindex(a::UBitSet{N}, idx) where N
    i1,i2 = Base.get_chunks_id(idx)
    _checkbounds(a, idx)
    @inbounds u = a.chunks[i1]
    return iszero(u & (1<< (i2 & 63)))
end

@propagate_inbounds function setindex!(a::MBitSet{N}, b, idx) where N
    i1,i2 = Base.get_chunks_id(idx)
    _checkbounds(a, idx)
    @inbounds u = a.chunks[i1]
    nu = ifelse(b, u | (1<< (i2 & 63)), u & ~(1<< (i2 & 63)) )
    ptr = convert(Ptr{UInt64}, pointer_from_objref(a))
    GC.@preserve a unsafe_store!(ptr, nu, i1)
    return b
end

@propagate_inbounds function Base.setindex(a::SBitSet{N}, b, idx) where N
    m = MBitSet(a)
    m[idx] = b
    return SBitSet(m)
end

## Convenience constructors
@propagate_inbounds function MBitSet{N}(idx::Integer) where N
    a = zero(MBitSet{N})
    a[idx] = true
    return a
end
@propagate_inbounds function SBitSet{N}(idx::Integer) where N 
    a = zero(MBitSet{N})
    _checkbounds(a, idx)
    @inbounds a[idx] = true
    return SBitSet(a)
end

@propagate_inbounds function MBitSet{N}(idxs::Integer...) where N
    a = zero(MBitSet{N})
    for idx in idxs
        a[idx] = true
    end
    return a
end
@propagate_inbounds SBitSet{N}(idxs::Integer...) where N = SBitSet(MBitSet{N}(idxs...))


@propagate_inbounds function MBitSet{N}(idxs::Vector) where N
    a = zero(MBitSet{N})
    for idx in idxs
        a[idx] = true
    end
    return a
end
@propagate_inbounds SBitSet{N}(idxs::Vector) where N = SBitSet(MBitSet{N}(idxs...))


#TODO: Maybe replace loop by branch-free ifelse code
@propagate_inbounds function MBitSet{N}(idxs::Base.OneTo{<:Integer}) where N
    a = zero(MBitSet{N})
    idxs.stop < 1 && return a
    _checkbounds(a, idxs.stop)
    i1, i2 = Base.get_chunks_id(idxs.stop)
    ptr = convert(Ptr{UInt64}, pointer_from_objref(a))
    GC.@preserve a begin
        @inbounds for i = 1:i1-1
            unsafe_store!(ptr, 0xffffffffffffffff, i)
        end
        unsafe_store!(ptr, (0xffffffffffffffff>> ((63-i2) & 63)), i1)
    end
    return a
end

@propagate_inbounds SBitSet{N}(idxs::Base.OneTo{<:Integer}) where N = SBitSet(MBitSet{N}(idxs))

@propagate_inbounds function MBitSet{N}(idxs::Base.UnitRange{<:Integer}) where N
    idxs.start == 1 && return MBitSet{N}(Base.OneTo(idxs.stop))
    @boundscheck begin 
         (0 < idxs.start && idxs.stop <= 64*N) || length(idxs)==0 || throw(BoundsError(MBitSet{N}, idxs))
    end
    @inbounds return MBitSet{N}(Base.OneTo(idxs.stop)) & ~MBitSet{N}(Base.OneTo(idxs.start-1))
end

@propagate_inbounds SBitSet{N}(idxs::Base.UnitRange{<:Integer}) where N = SBitSet(MBitSet{N}(idxs))

## Conversion to and from BitVector
function convert(::Type{BitVector}, a::UBitSet{N})  where N
    res = BitVector(undef, 64*N)
    @inbounds for i = 1:N
        res.chunks[i] = a.chunks[i]
    end
    res
end
convert(::Type{SBitSet{N}}, a::BitVector) where N = SBitSet(ntuple(i->a.chunks[i], Val{N}()))
convert(::Type{MBitSet{N}}, a::BitVector) where N = MBitSet(ntuple(i->a.chunks[i], Val{N}()))
convert(::Type{SBitSet}, a::BitVector) where N = convert(SBitSet{length(a.chunks)}, a)
convert(::Type{MBitSet}, a::BitVector) where N = convert(MBitSet{length(a.chunks)}, a)

## Equality, Sorting, Hashing
isequal(L::UBitSet, R::UBitSet) = L.chunks==R.chunks
==(L::UBitSet, R::UBitSet) where N = L.chunks==R.chunks

#This is taken from Base/hashing2.jl. Make faster, don't rely on unexported internals.
@inline function hash(x::UBitSet{N}, h::UInt = UInt(0)) where N
    @inbounds for i=1:N
        h = xor(h, Base.hash_uint(xor(x.chunks[i], h)))
    end
    return h
end

@inline _blsi(a) = a & -a
@inline function isless(a::UBitSet{N}, b::UBitSet{N}) where N
    @inbounds for i = 1:N
        ac = a.chunks[i]
        bc = b.chunks[i]
        ac == bc && continue
        msk = _blsi(xor(ac, bc))
        return iszero(bc & msk)
    end
    return false
end

## Random
@inline rand(rng::Random.AbstractRNG, ::Random.SamplerType{SBitSet{N}}) where {N} = SBitSet(ntuple(i->rand(rng, UInt64), Val{N}()))


## SIMD opt out. Make less ugly once github.com/JuliaLang/julia/pull/31113 has aged.
Base.SimdLoop.simd_index(v::UBitSet, j::Int64, i) = j
Base.SimdLoop.simd_inner_length(v::UBitSet, j::Int64) = 1
Base.SimdLoop.simd_outer_range(v::UBitSet) = v

@static if VERSION < v"1.1"
    #simd lowering changed from 1.0 -> 1.1.
    #this is still inferrable
    Base.last(a::UBitSet) = nothing
end

## Display
show(io::IO, ::MIME{Symbol("text/plain")}, B::UBitSet) = show(io, B)
show(io::IO, B::SBitSet{N}) where N = print(io, "SBitSet{$N}($(collect(B)))")
show(io::IO, B::MBitSet{N}) where N = print(io, "MBitSet{$N}($(collect(B)))")

## Set interface. 
issubset(a::UBitSet{N}, b::UBitSet{N}) where N = iszero(a & ~b)
union(a::UBitSet{N}, b::UBitSet{N}) where N = a | b
intersect(a::UBitSet{N}, b::UBitSet{N}) where N = a & b
symdiff(a::UBitSet{N}, b::UBitSet{N}) where N = xor(a,b)
setdiff(a::UBitSet{N}, b::UBitSet{N}) where N = a & ~b

function union!(a::MBitSet{N}, bs::UBitSet{N}...) where N 
    for b in bs
        a[] = a | b
    end
    a
end

function intersect!(a::MBitSet{N}, bs::UBitSet{N}...) where N 
    for b in bs
        a[] = a & b
    end
    a
end

function symdiff!(a::MBitSet{N}, bs::UBitSet{N}...) where N 
    for b in bs
        a[] = xor(a, b)
    end
    a
end

function setdiff!(a::MBitSet{N}, bs::UBitSet{N}...) where N 
    for b in bs
        a[] = a & ~b
    end
    a
end