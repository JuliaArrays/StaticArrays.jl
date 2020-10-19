
# Allow no new ambiguities (see #18), unless you fix some old ones first!

const allowable_ambiguities =
    if VERSION < v"1.1"
        3
    elseif VERSION < v"1.2"
        1
    else
        0
    end

if v"1.6.0-DEV.816" <= VERSION < v"1.6.0-rc"
    # Revisit in 1.6.0-rc1 or before. See
    #   https://github.com/JuliaLang/julia/pull/36962
    #   https://github.com/JuliaLang/julia/issues/36951
    @test_broken length(detect_ambiguities(#=LinearAlgebra, =#StaticArrays)) <= allowable_ambiguities
else
    @test length(detect_ambiguities(Base, LinearAlgebra, StaticArrays)) <= allowable_ambiguities
end
