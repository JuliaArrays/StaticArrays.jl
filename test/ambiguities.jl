
# Allow no new ambiguities (see #18), unless you fix some old ones first!

const allowable_ambiguities =
    if VERSION < v"1.1"
        4
    elseif VERSION < v"1.2"
        2
    else
        1
    end

@static if VERSION < v"1.6-"
   @test length(detect_ambiguities(Base, LinearAlgebra, StaticArrays)) <= allowable_ambiguities
else
   @test_broken length(detect_ambiguities(#=LinearAlgebra, =#StaticArrays)) <= allowable_ambiguities
end
