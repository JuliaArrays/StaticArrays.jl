"""
    x ≊ y

Inexact equality comparison. Like `≈` this calls `isapprox`, but with a
tighter tolerance of `rtol=10*eps()`.  Input with "\\approxeq".
"""
≊(x,y) = isapprox(x, y, rtol=10*eps())

"""
    @testinf a op b

Test that the type of the first argument `a` is inferred, and that `a op b` is
true.  For example, the following are equivalent:

    @testinf SVector(1,2) + SVector(1,2) == SVector(2,4)
    @test @inferred(SVector(1,2) + SVector(1,2)) == SVector(2,4)
"""
macro testinf(ex)
    @assert ex.head == :call
    infarg = ex.args[2]
    if !(infarg isa Expr) || infarg.head != :call
        # Workaround for an oddity in @inferred
        infarg = :(identity($infarg))
    end
    ex.args[2] = :(@inferred($infarg))
    esc(:(@test $ex))
end

function test_expand_error(ex)
    @test_throws LoadError macroexpand(@__MODULE__, ex)
end

mutable struct ErrorCounterTestSet <: Test.AbstractTestSet
    passcount::Int
    errorcount::Int
    failcount::Int
end
ErrorCounterTestSet(args...; kws...) = ErrorCounterTestSet(0,0,0)
Test.finish(ts::ErrorCounterTestSet) = ts
Test.record(ts::ErrorCounterTestSet, ::Test.Pass)  = (ts.passcount += 1)
Test.record(ts::ErrorCounterTestSet, ::Test.Error) = (ts.errorcount += 1)
Test.record(ts::ErrorCounterTestSet, ::Test.Fail)  = (ts.failcount += 1)

"""
    @test_inlined f(x,y, ...)

Check that the (optimized) llvm code generated for the expression
`f(x,y,...)` contains no `call` instructions.

Note that LLVM IR can contain `call` instructions to intrinsics which don't
make it into the native code, so this can be overly eager in declaring a
a lack of complete inlining.
"""
macro test_inlined(ex, should_inline=true)
    ex_orig = ex
    ex = macroexpand(@__MODULE__, :(@code_llvm $ex))
    expr = quote
        @static if hasfield(Base.CodegenParams, :safepoint_on_entry)
            code_str = let params = Base.CodegenParams(safepoint_on_entry=false),
                f = $(esc(ex.args[2])),
                t = $(esc(ex.args[3]))
                d = InteractiveUtils._dump_function(f, t, false, false, true, false, :att, true, :none, false, params)
                sprint(print, d)
            end
        else
            code_str = sprint() do io
                code_llvm(io, $(map(esc, ex.args[2:end])...))
            end
        end
        # Crude detection of call instructions remaining within what should be
        # fully inlined code.
        #
        # TODO: Figure out some better pattern matching; LLVM IR can contain
        # calls to intrinsics, so this will sometimes/often fail even when the
        # native code has no call instructions.
        $(should_inline ?
          :(@test !occursin("call", code_str)) :
          :(@test occursin("call", code_str))
        )
    end
    @assert expr.args[4].head == :macrocall
    expr.args[4].args[2] = __source__
    expr
end

should_be_inlined(x) = x*x
@noinline _should_not_be_inlined(x) = x*x
should_not_be_inlined(x) = _should_not_be_inlined(x)

"""
    @test_was_once_broken good_version ex

Expands to `@test ex` if `VERSION ≥ good_version` and to `@test_broken ex` if 
`VERSION ≥ good_version`. Useful for tests that are broken on earlier versions of Julia
that are fixed on later versions.
"""
macro test_was_once_broken(good_version, ex)
    esc(quote
        if VERSION < $good_version
            @test_broken $ex
        else
            @test $ex
        end
    end)
end

@testset "test utils" begin
    @testset "@testinf" begin
        @testinf [1,2] == [1,2]
        x = [1,2]
        @testinf x == [1,2]
        @testinf (@SVector [1,2]) == (@SVector [1,2])
    end

    @testset "@test_inlined" begin
        @test_inlined should_be_inlined(1)
        @test_inlined should_not_be_inlined(1) false
        ts = @testset ErrorCounterTestSet "" begin
            @test_inlined should_be_inlined(1) false
            @test_inlined should_not_be_inlined(1)
        end
        @test ts.errorcount == 0 && ts.failcount == 2 && ts.passcount == 0
    end
end
