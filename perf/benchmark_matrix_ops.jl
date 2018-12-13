using StaticArrays, BenchmarkTools, DataFrames, DataStructures

Nmax = 20
unary = (det, inv, exp)
binary = (\,)

data = OrderedDict{Symbol,Any}()
data[:SIZE] = vcat(([i, "", "", ""] for i in 1:Nmax)...)
data[:STAT] = [stat for sz in 1:Nmax for stat in ("compile time (s)", "StaticArrays (μs)", "Base (μs)", "max error")]

for f in unary
    f_data = Float64[]
    for N in 1:Nmax
        print("\r$((f,N))")
        SA = @SMatrix rand(N,N)
        A = Array(SA)
        push!(f_data, @elapsed f(SA))
        push!(f_data, 1e6*@belapsed $f($SA))
        push!(f_data, 1e6*@belapsed $f($A))
        push!(f_data, maximum([begin
                                   SA = @SMatrix rand(N,N)
                                   A = Array(SA)
                                   norm(f(A) - f(SA))
                               end for i in 1:1000]))
    end
    data[Symbol(f)] = f_data
end

for f in binary
    f_data = Float64[]
    for N in 1:Nmax
        print("\r$((f,N))")
        SA = @SMatrix rand(N,N)
        A = Array(SA)
        SB = @SMatrix rand(N,N)
        B = Array(SB)
        push!(f_data, @elapsed f(SA,SB))
        push!(f_data, 1e6*@belapsed $f($SA,$SB))
        push!(f_data, 1e6*@belapsed $f($A,$B))
        push!(f_data, maximum([begin
                                   SA = @SMatrix rand(N,N)
                                   A = Array(SA)
                                   SB = @SMatrix rand(N,N)
                                   B = Array(SB)
                                   norm(f(A,B) - f(SA,SB))
                               end for i in 1:1000]))
    end
    data[Symbol(f)] = f_data
end

df = DataFrame(data...)

open("bench_matrix_ops.txt", "w") do f
    versioninfo(f)
    showall(f, df)
end