var documenterSearchIndex = {"docs": [

{
    "location": "index.html#",
    "page": "Home",
    "title": "Home",
    "category": "page",
    "text": ""
},

{
    "location": "index.html#Static-Arrays-1",
    "page": "Home",
    "title": "Static Arrays",
    "category": "section",
    "text": "Statically sized arrays for JuliaStaticArrays provides a framework for implementing statically sized arrays in Julia (≥ 0.5), using the abstract type StaticArray{Size,T,N} <: AbstractArray{T,N}. Subtypes of StaticArray will provide fast implementations of common array and linear algebra operations. Note that here \"statically sized\" means that the size can be determined from the type, and \"static\" does not necessarily imply immutable.The package also provides some concrete static array types: SVector, SMatrix and SArray, which may be used as-is (or else embedded in your own type). Mutable versions MVector, MMatrix and MArray are also exported, as well as SizedArray for annotating standard Arrays with static size information. Further, the abstract FieldVector can be used to make fast static vectors out of any uniform Julia \"struct\"."
},

{
    "location": "pages/api.html#",
    "page": "API",
    "title": "API",
    "category": "page",
    "text": ""
},

{
    "location": "pages/api.html#API-1",
    "page": "API",
    "title": "API",
    "category": "section",
    "text": ""
},

{
    "location": "pages/api.html#Guide-1",
    "page": "API",
    "title": "Guide",
    "category": "section",
    "text": ""
},

{
    "location": "pages/api.html#SVector-1",
    "page": "API",
    "title": "SVector",
    "category": "section",
    "text": "The simplest static array is the type SVector{N,T}, which provides an immutable vector of fixed length N and type T.SVector defines a series of convenience constructors, so you can just type e.g. SVector(1,2,3). Alternatively there is an intelligent @SVector macro where you can use native Julia array literals syntax, comprehensions, and the zeros(), ones(), fill(), rand() and randn() functions, such as @SVector [1,2,3], @SVector Float64[1,2,3], @SVector [f(i) for i = 1:10], @SVector zeros(3), @SVector randn(Float32, 4), etc (Note: the range of a comprehension is evaluated at global scope by the macro, and must be made of combinations of literal values, functions, or global variables, but is not limited to just simple ranges. Extending this to (hopefully statically known by type-inference) local-scope variables is hoped for the future. The zeros(), ones(), fill(), rand() and randn() functions do not have this limitation.)"
},

{
    "location": "pages/api.html#SMatrix-1",
    "page": "API",
    "title": "SMatrix",
    "category": "section",
    "text": "Statically sized N×M matrices are provided by SMatrix{N,M,T,L}.Here L is the length of the matrix, such that N × M = L. However, convenience constructors are provided, so that L, T and even M are unnecessary. At minimum, you can type SMatrix{2}(1,2,3,4) to create a 2×2 matrix (the total number of elements must divide evenly into N). A convenience macro @SMatrix [1 2; 3 4] is provided (which also accepts comprehensions and the zeros(), ones(), fill(), rand(), randn() and eye() functions)."
},

{
    "location": "pages/api.html#SArray-1",
    "page": "API",
    "title": "SArray",
    "category": "section",
    "text": "A container with arbitrarily many dimensions is defined as struct SArray{Size,T,N,L} <: StaticArray{Size,T,N}, where Size = Tuple{S1, S2, ...} is a tuple of Ints. You can easily construct one with the @SArray macro, supporting all the features of @SVector and @SMatrix (but with arbitrary dimension).The main reason SVector and SMatrix are defined is to make it easier to define the types without the extra tuple characters (compare SVector{3} to SArray{Tuple{3}})."
},

{
    "location": "pages/api.html#Scalar-1",
    "page": "API",
    "title": "Scalar",
    "category": "section",
    "text": "Sometimes you want to broadcast an operation, but not over one of your inputs. A classic example is attempting to displace a collection of vectors by the same vector. We can now do this with the Scalar type:[[1,2,3], [4,5,6]] .+ Scalar([1,0,-1]) # [[2,2,2], [5,5,5]]Scalar is simply an implementation of an immutable, 0-dimensional StaticArray."
},

{
    "location": "pages/api.html#The-Size-trait-1",
    "page": "API",
    "title": "The Size trait",
    "category": "section",
    "text": "The size of a statically sized array is a static parameter associated with the type of the array. The Size trait is provided as an abstract representation of the dimensions of a static array. An array sa::SA of size (dims...) is associated with Size{(dims...)}(). The following are equivalent (@pure) constructors:Size{(dims...)}()\nSize(dims...)\nSize(sa::StaticArray)\nSize(SA) # SA <: StaticArrayThis is extremely useful for (a) performing dispatch depending on the size of an array, and (b) passing array dimensions that the compiler can reason about.An example of size-based dispatch for the determinant of a matrix would be:det(x::StaticMatrix) = _det(Size(x), x)\n_det(::Size{(1,1)}, x::StaticMatrix) = x[1,1]\n_det(::Size{(2,2)}, x::StaticMatrix) = x[1,1]*x[2,2] - x[1,2]*x[2,1]\n# and other definitions as necessaryExamples of using Size as a compile-time constant includereshape(svector, Size(2,2))  # Convert SVector{4} to SMatrix{2,2}\nSize(3,3)(rand(3,3))         # Construct a random 3×3 SizedArray (see below)"
},

{
    "location": "pages/api.html#Indexing-1",
    "page": "API",
    "title": "Indexing",
    "category": "section",
    "text": "Statically sized indexing can be realized by indexing each dimension by a scalar, a StaticVector or :. Indexing in this way will result a statically sized array (even if the input was dynamically sized, in the case of StaticVector indices) of the closest type (as defined by similar_type).Conversely, indexing a statically sized array with a dynamically sized index (such as a Vector{Integer} or UnitRange{Integer}) will result in a standard (dynamically sized) Array."
},

{
    "location": "pages/api.html#similar_type()-1",
    "page": "API",
    "title": "similar_type()",
    "category": "section",
    "text": "Since immutable arrays need to be constructed \"all-at-once\", we need a way of obtaining an appropriate constructor if the element type or dimensions of the output array differs from the input. To this end, similar_type is introduced, behaving just like similar, except that it returns a type. Relevant methods are:similar_type{A <: StaticArray}(::Type{A}) # defaults to A\nsimilar_type{A <: StaticArray, ElType}(::Type{A}, ::Type{ElType}) # Change element type\nsimilar_type{A <: AbstractArray}(::Type{A}, size::Size) # Change size\nsimilar_type{A <: AbstractArray, ElType}(::Type{A}, ::Type{ElType}, size::Size) # Change bothThese setting will affect everything, from indexing, to matrix multiplication and broadcast. Users wanting introduce a new array type should only overload the last method in the above.Use of similar will fall back to a mutable container, such as a MVector (see below), and it requires use of the Size trait if you wish to set a new static size (or else a dynamically sized Array will be generated when specifying the size as plain integers)."
},

{
    "location": "pages/api.html#Mutable-arrays:-MVector,-MMatrix-and-MArray-1",
    "page": "API",
    "title": "Mutable arrays: MVector, MMatrix and MArray",
    "category": "section",
    "text": "These statically sized arrays are identical to the above, but are defined as mutable structs, instead of immutable structs. Because they are mutable, they allow setindex! to be defined (achieved through pointer manipulation, into a tuple).As a consequence of Julia's internal implementation, these mutable containers live on the heap, not the stack. Their memory must be allocated and tracked by the garbage collector. Nevertheless, there is opportunity for speed improvements relative to Base.Array because (a) there may be one less pointer indirection, (b) their (typically small) static size allows for additional loop unrolling and inlining, and consequentially (c) their mutating methods like map! are extremely fast. Benchmarking shows that operations such as addition and matrix multiplication are faster for MMatrix than Matrix, at least for sizes up to 14 × 14, though keep in mind that optimal speed will be obtained by using mutating functions (like map! or A_mul_B!) where possible, rather than reallocating new memory.Mutable static arrays also happen to be very useful containers that can be constructed on the heap (with the ability to use setindex!, etc), and later copied as e.g. an immutable SVector to the stack for use, or into e.g. an Array{SVector} for storage.Convenience macros @MVector, @MMatrix and @MArray are provided."
},

{
    "location": "pages/api.html#SizedArray:-a-decorate-size-wrapper-for-Array-1",
    "page": "API",
    "title": "SizedArray: a decorate size wrapper for Array",
    "category": "section",
    "text": "Another convenient mutable type is the SizedArray, which is just a wrapper-type about a standard Julia Array which declares its knwon size. For example, if we knew that a was a 2×2 Matrix, then we can type sa = SizedArray{Tuple{2,2}}(a) to construct a new object which knows the type (the size will be verified automatically). A more convenient syntax for obtaining a SizedArray is by calling a Size object, e.g. sa = Size(2,2)(a).Then, methods on sa will use the specialized code provided by the StaticArrays pacakge, which in many cases will be much, much faster. For example, calling eig(sa) will be signficantly faster than eig(a) since it will perform a specialized 2×2 matrix diagonalization rather than a general algorithm provided by Julia and LAPACK.In some cases it will make more sense to use a SizedArray, and in other cases an MArray might be preferable."
},

{
    "location": "pages/api.html#FieldVector-1",
    "page": "API",
    "title": "FieldVector",
    "category": "section",
    "text": "Sometimes it might be useful to imbue your own types, having multiple fields, with vector-like properties. StaticArrays can take care of this for you by allowing you to inherit from FieldVector{N, T}. For example, consider:struct Point3D <: FieldVector{3, Float64}\n    x::Float64\n    y::Float64\n    z::Float64\nendWith this type, users can easily access fields to p = Point3D(x,y,z) using p.x, p.y or p.z, or alternatively via p[1], p[2], or p[3]. You may even permute the coordinates with p[SVector(3,2,1)]). Furthermore, Point3D is a complete AbstractVector implementation where you can add, subtract or scale vectors, multiply them by matrices, etc.It is also worth noting that FieldVectors may be mutable or immutable, and that setindex! is defined for use on mutable types. For immutable containers, you may want to define a method for similar_type so that operations leave the type constant (otherwise they may fall back to SVector). For mutable containers, you may want to define a default constructor (no inputs) and an appropriate method for similar,"
},

{
    "location": "pages/api.html#Implementing-your-own-types-1",
    "page": "API",
    "title": "Implementing your own types",
    "category": "section",
    "text": "You can easily create your own StaticArray type, by defining linear getindex (and optionally setindex! for mutable types - see setindex(::MArray, val, i) in MArray.jl for an example of how to achieve this through pointer manipulation). Your type should define a constructor that takes a tuple of the data (and mutable containers may want to define a default constructor).Other useful functions to overload may be similar_type (and similar for mutable containers)."
},

{
    "location": "pages/api.html#Conversions-from-Array-1",
    "page": "API",
    "title": "Conversions from Array",
    "category": "section",
    "text": "In order to convert from a dynamically sized AbstractArray to one of the statically sized array types, you must specify the size explicitly.  For example,v = [1,2]\n\nm = [1 2;\n     3 4]\n\n# ... a lot of intervening code\n\nsv = SVector{2}(v)\nsm = SMatrix{2,2}(m)\nsa = SArray{(2,2)}(m)\n\nsized_v = Size(2)(v)     # SizedArray{(2,)}(v)\nsized_m = Size(2,2)(m)   # SizedArray{(2,2)}(m)We have avoided adding SVector(v::AbstractVector) as a valid constructor to help users avoid the type instability (and potential performance disaster, if used without care) of this innocuous looking expression. However, the simplest way to deal with an Array is to create a SizedArray by calling a Size instance, e.g. Size(2)(v)."
},

{
    "location": "pages/api.html#Arrays-of-static-arrays-1",
    "page": "API",
    "title": "Arrays of static arrays",
    "category": "section",
    "text": "Storing a large number of static arrays is convenient as an array of static arrays. For example, a collection of positions (3D coordinates - SVector{3,Float64}) could be represented as a Vector{SVector{3,Float64}}.Another common way of storing the same data is as a 3×N Matrix{Float64}. Rather conveniently, such types have exactly the same binary layout in memory, and therefore we can use reinterpret to convert between the two formatsfunction svectors(x::Matrix{Float64})\n    @assert size(x,1) == 3\n    reinterpret(SVector{3,Float64}, x, (size(x,2),))\nendSuch a conversion does not copy the data, rather it refers to the same memory referenced by two different Julia Arrays. Arguably, a Vector of SVectors is preferable to a Matrix because (a) it provides a better abstraction of the objects contained in the array and (b) it allows the fast StaticArrays methods to act on elements."
},

{
    "location": "pages/api.html#Working-with-mutable-and-immutable-arrays-1",
    "page": "API",
    "title": "Working with mutable and immutable arrays",
    "category": "section",
    "text": "Generally, it is performant to rebind an immutable array, such asfunction average_position(positions::Vector{SVector{3,Float64}})\n    x = zeros(SVector{3,Float64})\n    for pos ∈ positions\n        x = x + pos\n    end\n    return x / length(positions)\nendso long as the Type of the rebound variable (x, above) does not change.On the other hand, the above code for mutable containers like Array, MArray or SizedArray is not very efficient. Mutable containers in Julia 0.5 must be allocated and later garbage collected, and for small, fixed-size arrays this can be a leading contribution to the cost. In the above code, a new array will be instantiated and allocated on each iteration of the loop. In order to avoid unnecessary allocations, it is best to allocate an array only once and apply mutating functions to it:function average_position(positions::Vector{SVector{3,Float64}})\n    x = zeros(MVector{3,Float64})\n    for pos ∈ positions\n        # Take advantage of Julia 0.5 broadcast fusion\n        x .= (+).(x, pos) # same as broadcast!(+, x, x, positions[i])\n    end\n    x .= (/).(x, length(positions))\n    return x\nendKeep in mind that Julia 0.5 does not fuse calls to .+, etc (or .+= etc), however the .= and (+).() syntaxes are fused into a single, efficient call to broadcast!. The simpler syntax x .+= pos is expected to be non-allocating (and therefore faster) in Julia 0.6.The functions setindex, push, pop, shift, unshift, insert and deleteat are provided for performing certain specific operations on static arrays, in analogy with the standard functions setindex!, push!, pop!, etc. (Note that if the size of the static array changes, the type of the output will differ from the input.)"
},

{
    "location": "pages/api.html#SIMD-optimizations-1",
    "page": "API",
    "title": "SIMD optimizations",
    "category": "section",
    "text": "It seems Julia and LLVM are smart enough to use processor vectorization extensions like SSE and AVX - however they are currently partially disabled by default. Run Julia with julia -O or julia -O3 to enable these optimizations, and many of your (immutable) StaticArray methods should become significantly faster!"
},

{
    "location": "pages/api.html#StaticArrays.FieldVector",
    "page": "API",
    "title": "StaticArrays.FieldVector",
    "category": "Type",
    "text": "abstract FieldVector{N, T} <: StaticVector{N, T}\n\nInheriting from this type will make it easy to create your own vector types. A FieldVector will automatically define getindex and setindex! appropriately. An immutable FieldVector will be as performant as an SVector of similar length and element type, while a mutable FieldVector will behave similarly to an MVector.\n\nFor example:\n\nstruct Point3D <: FieldVector{3, Float64}\n    x::Float64\n    y::Float64\n    z::Float64\nend\n\n\n\n"
},

{
    "location": "pages/api.html#StaticArrays.MArray",
    "page": "API",
    "title": "StaticArrays.MArray",
    "category": "Type",
    "text": "MArray{S, T, L}()\nMArray{S, T, L}(x::NTuple{L, T})\nMArray{S, T, L}(x1, x2, x3, ...)\n\nConstruct a statically-sized, mutable array MArray. The data may optionally be provided upon construction and can be mutated later. The S parameter is a Tuple-type specifying the dimensions, or size, of the array - such as Tuple{3,4,5} for a 3×4×5-sized array. The L parameter is the length of the array and is always equal to prod(S). Constructors may drop the L and T parameters if they are inferrable from the input (e.g. L is always inferrable from S).\n\nMArray{S}(a::Array)\n\nConstruct a statically-sized, mutable array of dimensions S (expressed as a Tuple{...}) using the data from a. The S parameter is mandatory since the size of a is unknown to the compiler (the element type may optionally also be specified).\n\n\n\n"
},

{
    "location": "pages/api.html#StaticArrays.MMatrix",
    "page": "API",
    "title": "StaticArrays.MMatrix",
    "category": "Type",
    "text": "MMatrix{S1, S2, T, L}()\nMMatrix{S1, S2, T, L}(x::NTuple{L, T})\nMMatrix{S1, S2, T, L}(x1, x2, x3, ...)\n\nConstruct a statically-sized, mutable matrix MMatrix. The data may optionally be provided upon construction and can be mutated later. The L parameter is the length of the array and is always equal to S1 * S2. Constructors may drop the L, T and even S2 parameters if they are inferrable from the input (e.g. L is always inferrable from S1 and S2).\n\nMMatrix{S1, S2}(mat::Matrix)\n\nConstruct a statically-sized, mutable matrix of dimensions S1 × S2 using the data from mat. The parameters S1 and S2 are mandatory since the size of mat is unknown to the compiler (the element type may optionally also be specified).\n\n\n\n"
},

{
    "location": "pages/api.html#StaticArrays.MVector",
    "page": "API",
    "title": "StaticArrays.MVector",
    "category": "Type",
    "text": "MVector{S,T}()\nMVector{S,T}(x::NTuple{S, T})\nMVector{S,T}(x1, x2, x3, ...)\n\nConstruct a statically-sized, mutable vector MVector. Data may optionally be provided upon construction, and can be mutated later. Constructors may drop the T and S parameters if they are inferrable from the input (e.g. MVector(1,2,3) constructs an MVector{3, Int}).\n\nMVector{S}(vec::Vector)\n\nConstruct a statically-sized, mutable vector of length S using the data from vec. The parameter S is mandatory since the length of vec is unknown to the compiler (the element type may optionally also be specified).\n\n\n\n"
},

{
    "location": "pages/api.html#StaticArrays.SArray",
    "page": "API",
    "title": "StaticArrays.SArray",
    "category": "Type",
    "text": "SArray{S, T, L}(x::NTuple{L, T})\nSArray{S, T, L}(x1, x2, x3, ...)\n\nConstruct a statically-sized array SArray. Since this type is immutable, the data must be provided upon construction and cannot be mutated later. The S parameter is a Tuple-type specifying the dimensions, or size, of the array - such as Tuple{3,4,5} for a 3×4×5-sized array. The L parameter is the length of the array and is always equal to prod(S). Constructors may drop the L and T parameters if they are inferrable from the input (e.g. L is always inferrable from S).\n\nSArray{S}(a::Array)\n\nConstruct a statically-sized array of dimensions S (expressed as a Tuple{...}) using the data from a. The S parameter is mandatory since the size of a is unknown to the compiler (the element type may optionally also be specified).\n\n\n\n"
},

{
    "location": "pages/api.html#StaticArrays.SMatrix",
    "page": "API",
    "title": "StaticArrays.SMatrix",
    "category": "Type",
    "text": "SMatrix{S1, S2, T, L}(x::NTuple{L, T})\nSMatrix{S1, S2, T, L}(x1, x2, x3, ...)\n\nConstruct a statically-sized matrix SMatrix. Since this type is immutable, the data must be provided upon construction and cannot be mutated later. The L parameter is the length of the array and is always equal to S1 * S2. Constructors may drop the L, T and even S2 parameters if they are inferrable from the input (e.g. L is always inferrable from S1 and S2).\n\nSMatrix{S1, S2}(mat::Matrix)\n\nConstruct a statically-sized matrix of dimensions S1 × S2 using the data from mat. The parameters S1 and S2 are mandatory since the size of mat is unknown to the compiler (the element type may optionally also be specified).\n\n\n\n"
},

{
    "location": "pages/api.html#StaticArrays.SVector",
    "page": "API",
    "title": "StaticArrays.SVector",
    "category": "Type",
    "text": "SVector{S, T}(x::NTuple{S, T})\nSVector{S, T}(x1, x2, x3, ...)\n\nConstruct a statically-sized vector SVector. Since this type is immutable, the data must be provided upon construction and cannot be mutated later. Constructors may drop the T and S parameters if they are inferrable from the input (e.g. SVector(1,2,3) constructs an SVector{3, Int}).\n\nSVector{S}(vec::Vector)\n\nConstruct a statically-sized vector of length S using the data from vec. The parameter S is mandatory since the length of vec is unknown to the compiler (the element type may optionally also be specified).\n\n\n\n"
},

{
    "location": "pages/api.html#StaticArrays.Scalar",
    "page": "API",
    "title": "StaticArrays.Scalar",
    "category": "Type",
    "text": "Scalar{T}(x::T)\n\nConstruct a statically-sized 0-dimensional array that contains a single element, x. This type is particularly useful for influencing broadcasting operations.\n\n\n\n"
},

{
    "location": "pages/api.html#StaticArrays.Size",
    "page": "API",
    "title": "StaticArrays.Size",
    "category": "Type",
    "text": "Size(dims::Int...)\n\nSize is used extensively in throughout the StaticArrays API to describe the size of a static array desired by the user. The dimensions are stored as a type parameter and are statically propagated by the compiler, resulting in efficient, type inferrable code. For example, to create a static matrix of zeros, use zeros(Size(3,3)) (rather than zeros(3,3), which constructs a Base.Array).\n\nSize(a::StaticArray)\nSize(::Type{T<:StaticArray})\n\nExtract the Size corresponding to the given static array. This has multiple uses, including using for \"trait\"-based dispatch on the size of a statically sized array. For example:\n\ndet(x::StaticMatrix) = _det(Size(x), x)\n_det(::Size{(1,1)}, x::StaticMatrix) = x[1,1]\n_det(::Size{(2,2)}, x::StaticMatrix) = x[1,1]*x[2,2] - x[1,2]*x[2,1]\n# and other definitions as necessary\n\n\n\n"
},

{
    "location": "pages/api.html#StaticArrays.Size-Union{Tuple{Array}, Tuple{S}} where S",
    "page": "API",
    "title": "StaticArrays.Size",
    "category": "Method",
    "text": "Size(dims)(array)\n\nCreates a SizedArray wrapping array with the specified statically-known dims, so to take advantage of the (faster) methods defined by the static array package.\n\n\n\n"
},

{
    "location": "pages/api.html#StaticArrays.SizedArray",
    "page": "API",
    "title": "StaticArrays.SizedArray",
    "category": "Type",
    "text": "SizedArray{Tuple{dims...}}(array)\n\nWraps an Array with a static size, so to take advantage of the (faster) methods defined by the static array package. The size is checked once upon construction to determine if the number of elements (length) match, but the array may be reshaped.\n\n(Also, Size(dims...)(array) acheives the same thing)\n\n\n\n"
},

{
    "location": "pages/api.html#StaticArrays.StaticArray",
    "page": "API",
    "title": "StaticArrays.StaticArray",
    "category": "Type",
    "text": "abstract type StaticArray{S, T, N} <: AbstractArray{T, N} end\nStaticScalar{T}     = StaticArray{Tuple{}, T, 0}\nStaticVector{N,T}   = StaticArray{Tuple{N}, T, 1}\nStaticMatrix{N,M,T} = StaticArray{Tuple{N,M}, T, 2}\n\nStaticArrays are Julia arrays with fixed, known size.\n\nDev docs\n\nThey must define the following methods:\n\nConstructors that accept a flat tuple of data.\ngetindex() with an integer (linear indexing) (preferably @inline with @boundscheck).\nTuple(), returning the data in a flat Tuple.\n\nIt may be useful to implement:\n\nsimilar_type(::Type{MyStaticArray}, ::Type{NewElType}, ::Size{NewSize}), returning a type (or type constructor) that accepts a flat tuple of data.\n\nFor mutable containers you may also need to define the following:\n\nsetindex! for a single element (linear indexing).\nsimilar(::Type{MyStaticArray}, ::Type{NewElType}, ::Size{NewSize}).\nIn some cases, a zero-parameter constructor, MyStaticArray{...}() for unintialized data is assumed to exist.\n\n(see also SVector, SMatrix, SArray, MVector, MMatrix, MArray, SizedArray and FieldVector)\n\n\n\n"
},

{
    "location": "pages/api.html#StaticArrays.similar_type",
    "page": "API",
    "title": "StaticArrays.similar_type",
    "category": "Function",
    "text": "similar_type(static_array)\nsimilar_type(static_array, T)\nsimilar_type(array, ::Size)\nsimilar_type(array, T, ::Size)\n\nReturns a constructor for a statically-sized array similar to the input array (or type) static_array/array, optionally with different element type T or size Size. If the input array is not a StaticArray then the Size is mandatory.\n\nThis differs from similar() in that the resulting array type may not be mutable (or define setindex!()), and therefore the returned type may need to be constructed with its data.\n\nNote that the (optional) size must be specified as a static Size object (so the compiler can infer the result statically).\n\nNew types should define the signature similar_type{A<:MyType,T,S}(::Type{A},::Type{T},::Size{S}) if they wish to overload the default behavior.\n\n\n\n"
},

{
    "location": "pages/api.html#Base.LinAlg.qr",
    "page": "API",
    "title": "Base.LinAlg.qr",
    "category": "Function",
    "text": "qr(A::StaticMatrix, pivot=Val{false}; thin=true) -> Q, R, [p]\n\nCompute the QR factorization of A such that A = Q*R or A[:,p] = Q*R, see qr. This function does not support thin=false keyword option due to type inference instability. To use this option call qr(A, pivot, Val{false}) instead.\n\n\n\n"
},

{
    "location": "pages/api.html#Base.similar-Union{Tuple{SA}, Tuple{SA}} where SA<:StaticArrays.StaticArray",
    "page": "API",
    "title": "Base.similar",
    "category": "Method",
    "text": "similar(static_array)\nsimilar(static_array, T)\nsimilar(array, ::Size)\nsimilar(array, T, ::Size)\n\nConstructs and returns a mutable but statically-sized array (i.e. a StaticArray). If the input array is not a StaticArray, then the Size is required to determine the output size (or else a dynamically sized array will be returned).\n\n\n\n"
},

{
    "location": "pages/api.html#StaticArrays._size-Tuple{Any}",
    "page": "API",
    "title": "StaticArrays._size",
    "category": "Method",
    "text": "Return either the statically known Size() or runtime size()\n\n\n\n"
},

{
    "location": "pages/api.html#StaticArrays.arithmetic_closure-Tuple{Any}",
    "page": "API",
    "title": "StaticArrays.arithmetic_closure",
    "category": "Method",
    "text": "arithmetic_closure(T)\n\nReturn the type which values of type T will promote to under a combination of the arithmetic operations +, -, * and /.\n\njulia> import StaticArrays.arithmetic_closure\n\njulia> arithmetic_closure(Bool)\nFloat64\n\njulia> arithmetic_closure(Int32)\nFloat64\n\njulia> arithmetic_closure(BigFloat)\nBigFloat\n\njulia> arithmetic_closure(BigInt)\nBigFloat\n\n\n\n"
},

{
    "location": "pages/api.html#StaticArrays.same_size-Tuple",
    "page": "API",
    "title": "StaticArrays.same_size",
    "category": "Method",
    "text": "Returns the common Size of the inputs (or else throws a DimensionMismatch)\n\n\n\n"
},

{
    "location": "pages/api.html#Docstrings-1",
    "page": "API",
    "title": "Docstrings",
    "category": "section",
    "text": "Pages   = [\"api.md\"]Modules = [StaticArrays]"
},

{
    "location": "pages/quickstart.html#",
    "page": "Quick Start",
    "title": "Quick Start",
    "category": "page",
    "text": ""
},

{
    "location": "pages/quickstart.html#Quick-Start-1",
    "page": "Quick Start",
    "title": "Quick Start",
    "category": "section",
    "text": "Pkg.add(\"StaticArrays\")  # or Pkg.clone(\"https://github.com/JuliaArrays/StaticArrays.jl\")\nusing StaticArrays\n\n# Create an SVector using various forms, using constructors, functions or macros\nv1 = SVector(1, 2, 3)\nv1.data === (1, 2, 3) # SVector uses a tuple for internal storage\nv2 = SVector{3,Float64}(1, 2, 3) # length 3, eltype Float64\nv3 = @SVector [1, 2, 3]\nv4 = @SVector [i^2 for i = 1:10] # arbitrary comprehensions (range is evaluated at global scope)\nv5 = zeros(SVector{3}) # defaults to Float64\nv6 = @SVector zeros(3)\nv7 = SVector{3}([1, 2, 3]) # Array conversions must specify size\n\n# Can get size() from instance or type\nsize(v1) == (3,)\nsize(typeof(v1)) == (3,)\n\n# Similar constructor syntax for matrices\nm1 = SMatrix{2,2}(1, 2, 3, 4) # flat, column-major storage, equal to m2:\nm2 = @SMatrix [ 1  3 ;\n                2  4 ]\nm3 = eye(SMatrix{3,3})\nm4 = @SMatrix randn(4,4)\nm5 = SMatrix{2,2}([1 3 ; 2 4]) # Array conversions must specify size\n\n# Higher-dimensional support\na = @SArray randn(2, 2, 2, 2, 2, 2)\n\n# Supports all the common operations of AbstractArray\nv7 = v1 + v2\nv8 = sin.(v3)\nv3 == m3 * v3 # recall that m3 = eye(SMatrix{3,3})\n# map, reduce, broadcast, map!, broadcast!, etc...\n\n# Indexing can also be done using static arrays of integers\nv1[1] === 1\nv1[SVector(3,2,1)] === @SVector [3, 2, 1]\nv1[:] === v1\ntypeof(v1[[1,2,3]]) <: Vector # Can't determine size from the type of [1,2,3]\n\n# Is (partially) hooked into BLAS, LAPACK, etc:\nrand(MMatrix{20,20}) * rand(MMatrix{20,20}) # large matrices can use BLAS\neig(m3) # eig(), etc uses specialized algorithms up to 3×3, or else LAPACK\n\n# Static arrays stay statically sized, even when used by Base functions, etc:\ntypeof(eig(m3)) == Tuple{SVector{3,Float64}, SMatrix{3,3,Float64,9}}\n\n# similar() returns a mutable container, while similar_type() returns a constructor:\ntypeof(similar(m3)) == MMatrix{3,3,Float64,9} # (final parameter is length = 9)\nsimilar_type(m3) == SMatrix{3,3,Float64,9}\n\n# The Size trait is a compile-time constant representing the size\nSize(m3) === Size(3,3)\n\n# A standard Array can be wrapped into a SizedArray\nm4 = Size(3,3)(rand(3,3))\ninv(m4) # Take advantage of specialized fast methods\n\n# reshape() uses Size() or types to specify size:\nreshape([1,2,3,4], Size(2,2)) == @SMatrix [ 1  3 ;\n                                            2  4 ]\ntypeof(reshape([1,2,3,4], Size(2,2))) === SizedArray{(2, 2),Int64,2,1}\n"
},

]}
