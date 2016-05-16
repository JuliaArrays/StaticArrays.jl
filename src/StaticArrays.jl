module StaticArrays

export StaticArray
export SArray
export MArray

include("util.jl")
include("core.jl")
include("abstractarray.jl")
include("map.jl")
include("linalg.jl")

end # module
