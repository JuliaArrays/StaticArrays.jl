# Allow no new ambiguities (see #18), unless you fix some old ones first!

const allowable_ambiguities = 55

# TODO: Revisit and fix. See
#   https://github.com/JuliaLang/julia/pull/36962
#   https://github.com/JuliaLang/julia/issues/36951
@test length(detect_ambiguities(#=LinearAlgebra, =#StaticArrays)) <= allowable_ambiguities