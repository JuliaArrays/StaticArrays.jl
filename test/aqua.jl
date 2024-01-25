if VERSION ≥ v"1.7-"
    Aqua.test_all(StaticArrays, piracies=false)
    Aqua.test_piracies(StaticArrays, treat_as_own=[
            StaticArray
            Size
            SArray
            MArray
            SizedArray
            FieldArray
            StaticArrays.StaticArrayStyle
            StaticArrays.SOneTo
            StaticArrays.StaticArraysCore
            StaticArrays.check_array_parameters
            StaticArrays.convert_ntuple
        ],
        broken=true)

elseif VERSION ≥ v"1.6-"
    Aqua.test_all(StaticArrays, piracies=false, ambiguities=false)
    Aqua.test_piracies(StaticArrays, treat_as_own=[
            StaticArray
            Size
            SArray
            MArray
            SizedArray
            FieldArray
            StaticArrays.StaticArrayStyle
            StaticArrays.SOneTo
            StaticArrays.StaticArraysCore
            StaticArrays.check_array_parameters
            StaticArrays.convert_ntuple
        ],
        broken=true)

    # Allow no new ambiguities (see #18), unless you fix some old ones first!
    # TODO: Revisit and fix. See
    #   https://github.com/JuliaLang/julia/pull/36962
    #   https://github.com/JuliaLang/julia/issues/36951
    # Let's not allow new ambiguities. If any change makes the ambiguity count decrease, the
    # change should decrement `allowable_ambiguities` accordingly
    allowable_ambiguities = 1
    @test length(detect_ambiguities(StaticArrays)) == allowable_ambiguities

end
