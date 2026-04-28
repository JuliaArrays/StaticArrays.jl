module StaticArraysMooncakeExt

using StaticArrays: SArray,MArray
using Mooncake: Mooncake

@static if isdefined(Mooncake, :FriendlyTangentCache)  # checks Mooncake >= v0.5.25
    # see https://github.com/JuliaDiff/DifferentiationInterface.jl/issues/998
    function Mooncake.friendly_tangent_cache(x::Union{SArray,MArray})
        return Mooncake.FriendlyTangentCache{Mooncake.AsPrimal}(copy(x))
    end
end

end
