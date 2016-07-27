function bc(x,i)
    (i > 0 && i <= length(x)) || error("bc failed")
end

@inline function mygetindex(x,i)
    #@boundscheck bc(x,i)
    @boundscheck error("Always an error")
    @inbounds return x[i]
end

foo(x,i) = @inbounds return mygetindex(x,i)
bar(x,i) = return mygetindex(x,i)


#=function getindex1(n, i)
    @boundscheck error("YOU SHALL NEVER CHECK MY BOUNDS!!")
    @inbounds return n[i]
end

#foo(x) = @inbounds return getindex1(x,1)
function foo(x)
    @inbounds out = getindex1(x,1)
    out
end

julia> getindex1((1,2,3),1)
ERROR: YOU SHALL NEVER CHECK MY BOUNDS!!
 in getindex1(::Tuple{Int64,Int64,Int64}, ::Int64) at /home/aferris/.julia/v0.5/StaticArrays/perf/boundscheck.jl:2

julia> f1((1,2,3))
ERROR: StackOverflowError:
# I have also got, by modifying the above in trivial ways:
#   * ERROR: ReadOnlyMemoryError()
#   * signal (11): Segmentation fault (... quits)


julia> @code_warntype foo((1,2,3))
Variables:
  #self#::#foo
  x::Tuple{Int64,Int64,Int64}
  out::Union{}

Body:
  begin
      $(Expr(:inbounds, true))
      $(Expr(:inbounds, false))
      # meta: location /home/aferris/.julia/v0.5/StaticArrays/perf/boundscheck.jl getindex1 2
      $(Expr(:boundscheck, true))
      (Base.throw)($(Expr(:new, :(Core.ErrorException), "YOU SHALL NEVER CHECK MY BOUNDS!!")))::Union{}
      $(Expr(:boundscheck, :pop))
      $(Expr(:inbounds, true))
      $(Expr(:inbounds, :pop))
      # meta: pop location
      $(Expr(:inbounds, :pop))
      out::Union{} = nothing
      $(Expr(:inbounds, :pop))
  end::Union{}

=#
#=immutable NeverInBounds
    x::Tuple{Int,Int,Int}
end

function getindex1(n::NeverInBounds, i)
    @boundscheck error("YOU SHALL NEVER CHECK MY BOUNDS!!")
    @inbounds return n.x[1]
end

n = NeverInBounds((1,2,3))

f0(x) = getindex1(x, 1)
#function f1(x)
#    @inbounds return getindex1(x, 1)
#end
f1(x) = @inbounds return getindex1(x,1)

try
    f0(n)
    error("There should have been an error")
catch
    println("f0 throws an error")
end

#println("f1 returns $(f1(n))")
=#
