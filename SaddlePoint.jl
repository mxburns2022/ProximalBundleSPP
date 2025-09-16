using Random
using LinearAlgebra
using ProximalAlgorithms
using ProximalOperators
import SparseArrays as sp




# @doc """Definitions for the Saddle Point problem
# ```math
# \min_{x\in\Delta^N}\max_{y\in\Delta^M}\{y^\top Ax + \gamma_x\|x\|_\infty- \gamma_y\|y\|_\infty\}\triangleq f(x,y)
# ```
# """
struct BilinearSPP{TR,TA}
    A::TA
    N::Int
    M::Int
    γy::TR
    γx::TR
end



# Return the subgradient of f(x,y) with respect to x
function subgradient!(game::BilinearSPP, g::AbstractVector{T}, x::AbstractVector{T}, y::AbstractVector{T}) where T<:Real
    g .= game.A' * y
    maxargs = findall(x .== maximum(x))
    g[maxargs] .+= game.γx / size(maxargs, 1)
end

# Return the supergradient of f(x,y) with respect to y
function supergradient!(game::BilinearSPP, g::AbstractVector{T}, x::AbstractVector{T}, y::AbstractVector{T}) where T<:Real
    g .= game.A * x
    maxargs = findall(y .== maximum(y))
    g[maxargs] .-= game.γy / size(maxargs, 1)
end

# Evaluate the value of the SPP
function eval(game::BilinearSPP, x::AbstractVector{T}, y::AbstractVector{T}) where T<:Real
    return y' * game.A * x + game.γx * maximum(x) - game.γy * maximum(y)
end


# Generate a random zero-sum L∞ regularized game with the given parameters
function generate_random_zero_sum(n::Int, m::Int, density::Float64, stddev::Float64; seed::Int=0, sparse::Bool=true)
    generator = Xoshiro(seed)
    if !sparse
        A = ((randn(generator, m, n) .* stddev) .* convert.(Float64, rand(generator, m, n) .<= density))
    else
        rfn(gen, k) = randn(gen, k) .* stddev
        A = sp.sprand(generator, m, n, density, rfn)
    end

    return BilinearSPP(A, n, m, 1.0, 1.0)
end

# Compute the value of the primal function max_{y∈Δᴹ} f(x,y), see paper for derivation
function primalv(game::BilinearSPP, x::AbstractVector{T}; solution::Bool=false) where T<:Real
    Ax = -game.A * x
    indices = sortperm(Ax) # Sort the negated vector in ascending order
    sorted_Ax = Ax[indices]
    index = 1
    value = sorted_Ax[1] + game.γy
    running_sum = sorted_Ax[1]
    # Find the first index with a increasing increment
    for i in 2:game.M
        running_sum += sorted_Ax[i]
        newval = running_sum / i + game.γy / i
        if newval > value
            break
        end
        index = i
        value = newval
    end
    # If not returning the optimizer as well, then return the primal value
    if !solution
        return -value + maximum(x) * game.γx
    end
    # Return the optimizer along with the value
    optimizer = zeros(game.N)
    optimizer[indices[1:index]] .= 1 / index
    return -value + maximum(x) * game.γx, optimizer
end

# Compute the value of the dual function min_{x∈Δᴺ} f(x,y), see paper for derivation
function dualv(game::BilinearSPP, y::AbstractVector{T}; solution::Bool=false) where T<:Real
    Ay = game.A' * y
    indices = sortperm(Ay) # Sort the vector in ascending order
    sorted_Ay = Ay[indices]
    index = 1
    value = sorted_Ay[1] + game.γx
    running_sum = sorted_Ay[1]

    # Find the first index with a increasing increment
    for i in 2:game.N
        running_sum += sorted_Ay[i]
        newval = running_sum / i + game.γx / i
        if newval > value
            break
        end
        index = i
        value = newval
    end
    # If not returning the optimizer as well, then return the dual value
    if !solution
        return value - maximum(y) * game.γy
    end
    # Return the optimizer along with the value
    optimizer = zeros(game.N)
    optimizer[indices[1:index]] .= 1 / index
    return value - maximum(y) * game.γy, optimizer
end



# Implementation of the CS-SPP method for solving the SaddlePointProblem
function subgradient_method(game::BilinearSPP, target_accuracy::Float64; log_frequency::Int=1000, dynamic_stepsize::Bool=true)
    # Allocate sub/supergradient cache
    gx = zeros(game.N)
    gy = zeros(game.M)
    # Initialize both players to uniform random
    x = ones(game.N) / game.N
    y = ones(game.M) / game.M
    # Compute the upper estimate of the sub/supergradient bound M
    Mx = maximum([norm(game.A[i, :]) for i in 1:game.M]) + game.γx
    My = maximum([norm(game.A[:, i]) for i in 1:game.N]) + game.γy
    M = max(Mx, My)

    # Indicator function of the unit simplex
    h = IndSimplex()

    # Number of iterations (while allowing for early termination)
    niter = convert(Int, ceil(128 * M^2 * 1 / (target_accuracy^2)))

    # Set the initial stepsize
    h₁ = if dynamic_stepsize
        1 / (32 * M^2)
    else
        target_accuracy / (32 * M^2)
    end

    # Allocate space for the ergodic averages
    x̄ = copy(x)
    ȳ = copy(y)
    starttime = time_ns()
    # data = []
    println("elapsed_time(s),inner_steps,outer_steps,pd_gap")
    Tk = 0.0
    for k in 1:niter
        # Set the stepsize for this iteration
        hₖ = if dynamic_stepsize
            h₁ / sqrt(k)
        else
            h₁
        end
        Tk += hₖ
        # Log step
        if (k - 1) % log_frequency == 0
            println(round((time_ns() - starttime) / 1e9, sigdigits=6), ",", 2(k - 1), ",", k, ",", primalv(game, x̄) - dualv(game, ȳ))
            flush(stdout)
        end
        # Check for early termination
        if k % 500 == 0 && primalv(game, x̄) - dualv(game, ȳ) <= target_accuracy
            return x, y
        end

        # Compute the sub/super gradients (without performing either update)
        subgradient!(game, gx, x, y)
        supergradient!(game, gy, x, y)
        # Update x, y
        axpby!(-hₖ, gx, 1.0, x)
        axpby!(hₖ, gy, 1.0, y)

        # Perform the projection back to the unit simplex
        prox!(x, h, x)
        prox!(y, h, y)

        # Update the sample averages
        axpby!(hₖ / Tk, x, (Tk - hₖ) / Tk, x̄)
        axpby!(hₖ / Tk, y, (Tk - hₖ) / Tk, ȳ)
    end
    return x, y
end


