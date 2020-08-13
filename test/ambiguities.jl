
# Allow no new ambiguities (see #18), unless you fix some old ones first!

const allowable_ambiguities =
    if VERSION < v"1.1"
        3
    elseif VERSION < v"1.2"
        1
    else
        0
    end

@test length(detect_ambiguities(Base, LinearAlgebra, StaticArrays)) <= allowable_ambiguities
