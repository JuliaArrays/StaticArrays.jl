# Allow no new ambiguities (see #18), unless you fix some old ones first!

const allowable_ambiguities = VERSION ≥ v"1.7" ? 0 :
                              VERSION ≥ v"1.6" ? 1 : error("version must be ≥1.6")

# TODO: Revisit and fix. See
#   https://github.com/JuliaLang/julia/pull/36962
#   https://github.com/JuliaLang/julia/issues/36951
# Let's not allow new ambiguities. If any change makes the ambiguity count decrease, the
# change should decrement `allowable_ambiguities` accordingly
@test length(detect_ambiguities(#=LinearAlgebra, =#StaticArrays)) == allowable_ambiguities