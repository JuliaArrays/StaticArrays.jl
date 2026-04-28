module StaticArraysMooncakeExt

using Mooncake
using Random: AbstractRNG
using Base: IEEEFloat
using StaticArrays: StaticArrays, SArray

using Mooncake: @foldable

import Mooncake:
    MaybeCache,
    IncCache,
    SetToZeroCache,
    NoFData,
    NoRData,
    CoDual,
    Dual,
    MinimalCtx,
    primal,
    tangent,
    zero_fcodual,
    tangent_type,
    fdata_type,
    rdata_type,
    zero_tangent_internal,
    randn_tangent_internal,
    set_to_zero_internal!!,
    increment_internal!!,
    _add_to_primal_internal,
    tangent_to_primal_internal!!,
    primal_to_tangent_internal!!,
    _dot_internal,
    _scale_internal,
    zero_rdata,
    zero_rdata_from_type,
    can_produce_zero_rdata_from_type,
    _verify_rdata_value,
    __verify_fdata_value,
    _new_

# Element types treated as differentiable: real and complex IEEE floats.
const _SElt = Union{IEEEFloat,Complex{<:IEEEFloat}}

# An SArray with a supported element type uses *itself* as its tangent type.
# It is immutable and stores only by-value scalar data, so its fdata is empty
# and its rdata carries the full tangent. This mirrors how Mooncake handles
# `Complex{<:IEEEFloat}` in `src/rules/complex.jl`.

@foldable function tangent_type(::Type{SArray{S,T,N,L}}) where {S,T<:_SElt,N,L}
    return SArray{S,T,N,L}
end

@foldable function tangent_type(
    ::Type{NoFData}, ::Type{SArray{S,T,N,L}}
) where {S,T<:_SElt,N,L}
    return SArray{S,T,N,L}
end

# Non-parametric alias used as a constraint, analogous to `CF` in complex.jl.
const _SAFloat = SArray{S,T,N,L} where {S,T<:_SElt,N,L}

@foldable fdata_type(::Type{T}) where {T<:_SAFloat} = NoFData
@foldable rdata_type(::Type{T}) where {T<:_SAFloat} = T

tangent(::NoFData, t::_SAFloat) = t

# Core tangent operations.

zero_tangent_internal(p::_SAFloat, ::MaybeCache) = zero(p)

zero_rdata(p::_SAFloat) = zero(p)
zero_rdata_from_type(::Type{P}) where {P<:_SAFloat} = zero(P)
@foldable can_produce_zero_rdata_from_type(::Type{<:_SAFloat}) = true

set_to_zero_internal!!(::SetToZeroCache, p::_SAFloat) = zero(p)

function randn_tangent_internal(
    rng::AbstractRNG, ::SArray{S,T,N,L}, ::MaybeCache
) where {S,T<:_SElt,N,L}
    return SArray{S,T,N,L}(ntuple(_ -> randn(rng, T), Val(L)))
end

increment_internal!!(::IncCache, t::T, s::T) where {T<:_SAFloat} = t + s

_add_to_primal_internal(::MaybeCache, x::T, t::T, ::Bool) where {T<:_SAFloat} = x + t

tangent_to_primal_internal!!(::T, t::T, ::MaybeCache) where {T<:_SAFloat} = t
primal_to_tangent_internal!!(::T, x::T, ::MaybeCache) where {T<:_SAFloat} = x

# By-value type, so there is no primal address to record.
function Mooncake.TestUtils.populate_address_map_internal(
    m::Mooncake.TestUtils.AddressMap, ::P, ::P
) where {P<:_SAFloat}
    return m
end

# rdata/fdata are validated structurally for non-primitive aggregates; by-value
# SArrays are leaves like `Complex`, so we short-circuit verification.
_verify_rdata_value(::P, ::P) where {P<:_SAFloat} = nothing
__verify_fdata_value(::IdDict{Any,Nothing}, ::P, ::P) where {P<:_SAFloat} = nothing

# Delegate element-wise reductions to the existing tuple handlers, which in
# turn dispatch to per-element `_dot_internal` / `_scale_internal` (correct for
# both `IEEEFloat` and `Complex{<:IEEEFloat}` element types).
function _dot_internal(c::MaybeCache, t::T, s::T) where {T<:_SAFloat}
    return _dot_internal(c, Tuple(t), Tuple(s))
end

function _scale_internal(c::MaybeCache, a::Float64, t::T) where {T<:_SAFloat}
    return T(_scale_internal(c, a, Tuple(t)))
end

# Rules. `_new_` is already declared a primitive globally
# (`Tuple{typeof(_new_),Vararg}` in `src/rules/new.jl`), so we only need to
# add more-specific `frule!!` / `rrule!!` methods for SArray construction.
# Mooncake's IR normalisation rewrites `SArray(...)` constructor calls to
# `_new_(SArray{S,T,N,L}, data::NTuple{L,T})`.

function Mooncake.frule!!(
    ::Dual{typeof(_new_)}, ::Dual{Type{P}}, data::Dual{NTuple{L,T}}
) where {S,T<:_SElt,N,L,P<:SArray{S,T,N,L}}
    y = _new_(P, primal(data))
    dy = _new_(P, tangent(data))
    return Dual(y, dy)
end

function Mooncake.rrule!!(
    ::CoDual{typeof(_new_)}, ::CoDual{Type{P}}, data::CoDual{NTuple{L,T}}
) where {S,T<:_SElt,N,L,P<:SArray{S,T,N,L}}
    y = _new_(P, primal(data))
    _new_SArray_pb(dy::P) = NoRData(), NoRData(), Tuple(dy)
    return zero_fcodual(y), _new_SArray_pb
end

Mooncake.@is_primitive MinimalCtx Tuple{
    typeof(getindex),SArray{S,T,N,L},Int
} where {S,T<:_SElt,N,L}

function Mooncake.frule!!(
    ::Dual{typeof(getindex)}, x::Dual{P}, i::Dual{Int}
) where {S,T<:_SElt,N,L,P<:SArray{S,T,N,L}}
    idx = primal(i)
    return Dual(primal(x)[idx], tangent(x)[idx])
end

function Mooncake.rrule!!(
    ::CoDual{typeof(getindex)}, x::CoDual{P,NoFData}, i::CoDual{Int}
) where {S,T<:_SElt,N,L,P<:SArray{S,T,N,L}}
    idx = primal(i)
    y = primal(x)[idx]
    function getindex_SArray_pb(dy::T)
        dx = P(ntuple(j -> j == idx ? dy : zero(T), Val(L)))
        return NoRData(), dx, NoRData()
    end
    return zero_fcodual(y), getindex_SArray_pb
end

end # module StaticArraysMooncakeExt
