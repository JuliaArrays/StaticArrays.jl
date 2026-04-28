using Pkg
package = ENV["STATICARRAYS_DOWNSTREAM_TEST_PACKAGE"];
env = ENV["STATICARRAYS_DOWNSTREAM_TEST_ENV"];
Pkg.activate(joinpath(@__DIR__, package, env))
Pkg.develop(PackageSpec(path=joinpath(@__DIR__, "..", "..")))
include(joinpath(package, "test.jl"))
