
# Allow no new ambiguities (see #18), unless you fix some old ones first!
@test length(detect_ambiguities(Base, LinearAlgebra, StaticArrays)) <= 5

