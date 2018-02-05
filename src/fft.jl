using Primes


ωcos(N::Integer, i::Integer) = cospi(-2i//N)
ωsin(N::Integer, i::Integer) = sinpi(-2i//N)
ωcos(N, i) = cospi(-2i/N)
ωsin(N, i) = sinpi(-2i/N)

ω(N, i) = ωcos(N, i) + im*ωsin(N, i)

struct fft_meta1
    factors::Vector{Int}
    status::Vector{Int}
    steps::Vector{Int} 
    remaining::Vector{Int}
    cumprod::Vector{Int}
    inds::Vector{Int}
end
function fft_meta1(N::Integer)
    factors = factor(Vector, N)
    reverse!(factors)
    status = fill(1,4)#status order is last size, current size, cumulative, it_num
    steps = fill(1, length(factors))
    for i ∈ 2:length(factors) #cumprod, but with a 1 in front.
        steps[i] = factors[i-1] * steps[i-1]
    end
    remaining = N ./ steps
    cp = cumprod(factors)
    fft_meta1(factors, status, steps, remaining, cp, fill(1,length(factors)))

end

function shrink_tree!(x::fft_meta1)
    x.status[3] *= x.status[2]
    x.status[1] = x.status[2]
    x.status[2] = pop!(x.factors)
end

### The plan here is to traverse the remaining tree.
function Base.start(x::fft_meta1)
    shrink_tree!(x)
 #   if length(iter) > 0
    iter = x.inds
    resize!(iter, length(x.factors))
    fill!(iter, 1)
    iter[end] = 0
    iter
end
function Base.next(x::fft_meta1, iter)
    for j ∈ length(iter):-1:1
        if iter[j] == x.factors[j]
            iter[j] = 1
        else
            iter[j] += 1
            break
        end
    end
    iter, iter
end
function Base.done(x::fft_meta1, iter)
    done = true
    for j ∈ eachindex(iter)
        if iter[j] != x.factors[j]
            done = false
            break
        end
    end
    done
end
Base.eltype(::fft_meta1) = Vector{Int}
function Base.length(x::fft_meta1)
    out = 1
    @inbounds for i ∈ 1:length(x.factors)
        out *= x.factors[i]
    end
    out
end


function offset_gap(x::fft_meta1, iter)
    offset = iter[1]
    for i ∈ 2:length(x.factors)
        offset += (iter[i]-1)*x.cumprod[i-1]
    end
    offset, x.cumprod[length(x.factors)]
end

function initial_expr(x::fft_meta1, i, iter, o = :o_, input = :x_)
    expr = quote end
    initial_expr!(expr, x, i, iter, o, input)
end
function initial_expr!(expr, x::fft_meta1, i, iter, output = :o_, input = :x_, ::Type{T} = Float64) where T
#    @show iter
    off, gap = offset_gap(x, iter)
    N = x.status[2]
    initial_expr!(expr, off, gap, N, i, output, input, T)
end
function initial_expr!(expr, off, gap, N, i, output = :o_, input = :x_, ::Type{T} = Float64) where T
    for j ∈ 0:N-1
        push!(expr.args[2].args[2].args, 
                :( $(Symbol( output, 1+i*N+j )) = $(Symbol(input, off)) +
                    $(ω(N, j)) * $(Symbol(input, off+gap)) ) )
        for k ∈ 2:N-1 ##push to new line with += to avoid allocation.
            push!(expr.args[2].args[2].args, 
            :( $(Symbol( output, 1+i*N+j )) +=   $(ω(N, j*k)) * $(Symbol(input, off+gap*k) ) ) )
        end
    end
    expr
end
function initial_exprdefunct!(expr, off, gap, N, i, output = :o_, input = :x)
#    @show iter
    for j ∈ 0:N-1
        summation = :( $(Symbol( output, 1+i*N+j )) =
                ($input)[$(off)] + $(ω(N, j)) * ($input)[$(off+gap)] )  
        for k ∈ 2:N-1
            push!(summation.args[2].args, :( ( $(ω(N, j*k)) ) * ($input)[$(off+gap*k)] ))
        end
        push!(expr.args, summation)
    end
    expr
end
function gen_initial_expr(fm::fft_meta1, ::Type{T} = Float64) where T
    expr = quote @fastmath begin end end
    for (i,iter) ∈ enumerate(fm)
        initial_expr!(expr, fm, i-1, iter, :o_, :x_, T)
    end
    expr
end
function gie(n::Int)
    fm = fft_meta1(n)
    expr = quote end
    for (i,iter) ∈ enumerate(fm)
        initial_expr!(expr, fm, i-1, iter)
    end
    expr
end

function next_iter!(expr, fm::fft_meta1, ::Type{T} = Float64) where T
    fm.status[end] += 1
    shrink_tree!(fm)
    l, N, c, it_num = fm.status
    for i ∈ 1:length(fm)
        combine!(expr, l, N, c, i-1, (:o_, :u_)[1 + it_num%2], (:u_, :o_)[1 + it_num%2], T )
    end
end

function combine!(expr, last_N, current_N, cumulative_N, i, input, output, ::Type{T} = Float64) where T
    N = cumulative_N * current_N
    offset = i*N
    ind = 0
    for l ∈ 1:current_N, j ∈ 1:cumulative_N
        push!(expr.args[2].args[2].args, :( $(Symbol(output, 1+ind+offset )) =
                $(Symbol(input, offset+j)) +
                $(ω(N, ind)) * $(Symbol(input, offset+j+cumulative_N)) ) )
        for k ∈ 2:current_N-1 ##push to new line with += to avoid allocation.
            push!(expr.args[2].args[2].args,
                :( $(Symbol(output, 1+ind+offset )) += ( $(ω(N, ind*k)) ) * $(Symbol(input, offset+j+cumulative_N*k) ))) 

        end
        ind += 1
    end
    expr
end

function fft_expression(N::Int, ::Type{T}) where T
    fm = fft_meta1(N)
    expr = gen_initial_expr(fm, T)
    n = length(fm.factors)
    for i ∈ 1:n
        next_iter!(expr, fm, T)
    end
    last_out = (:o_,:u_)[2-fm.status[4]%2]
    out = :( SVector( ( $(Symbol(last_out, 1)), $(Symbol(last_out, 2))) ) )
    for i ∈ 3:N
        push!(out.args[2].args, Symbol( last_out, i ))
    end
    push!(expr.args, out)
    expr
end

@generated function Base.fft(x::A) where {N, T, A <: StaticArray{Tuple{N},T}}
    fft_expr = fft_expression(N,T)
    quote
    #    $(Expr(:meta, :inline))
        @inbounds begin
            Base.Cartesian.@nextract $N x x
        end
        $fft_expr
    end
end
