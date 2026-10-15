using StaticArrays, ParallelTestRunner

# Tests from a single file can be run via Pkg.test. For example, to run only
# the MVector and SVector tests, use:
#
#   Pkg.test("StaticArrays", test_args=["MVector", "SVector"])
#

# Each test file runs in its own module, set up with this code
const init_code = quote
    using StaticArrays, Test
    include($(joinpath(@__DIR__, "testutil.jl")))
end

testsuite = find_tests(@__DIR__)
# testutil.jl only defines helpers, which are loaded by `init_code`
delete!(testsuite, "testutil")

# check_bounds_no.jl has to run in a Julia process with --check-bounds=no
test_worker(name) = name == "check_bounds_no" ? addworker(; exeflags=["--check-bounds=no"]) : nothing

runtests(StaticArrays, ARGS; testsuite, init_code, test_worker)
