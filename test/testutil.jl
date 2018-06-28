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

@testset "@testinf" begin
    @testinf [1,2] == [1,2]
    x = [1,2]
    @testinf x == [1,2]
    @testinf (@SVector [1,2]) == (@SVector [1,2])
end

function test_expand_error(ex)
    @test_throws LoadError macroexpand(@__MODULE__, ex)
end
