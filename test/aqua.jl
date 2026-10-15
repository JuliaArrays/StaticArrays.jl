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
