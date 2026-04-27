module StaticArraysMooncakeExt

using StaticArrays: StaticArray
using Mooncake: Mooncake

@static if isdefined(Mooncake, :FriendlyTangentCache)  # checks Mooncake >= v0.5.25
    function Mooncake.friendly_tangent_cache(x::StaticArray)
        return Mooncake.FriendlyTangentCache{Mooncake.AsPrimal}(copy(x))
    end
end

end
