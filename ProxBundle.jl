using ProximalAlgorithms
include("SaddlePoint.jl")
import ProximalOperators: prox!
using Test


# Object to store the multicut subproblem max_{λ∈Δᵖ}\{-hθᶜ(-Aλ)+bᵀλ\} for the FISTA subsolver, where hθᶜ(y)=max_{x∈Δⁿ}{⟨y, Aᵀx+b⟩+h(x)+1/2θ||x-x_0||²}
Base.@kwdef struct ProxBundleSubproblem{R,Tx,Tb,TA}
    x0::Tx # Prox center
    A::TA  # Primal objective matrix (concatenated subgradients)
    b::Tb  # Primal objective vector (concatenated linearization constants)
    h::Any  # Simple closed convex function with available Prox Operator
    θ::R # stepsize
end


# Storage for Prox Bundle Solver State
Base.@kwdef mutable struct ProxBundleState{R,Tx}
    x::Tx                   # iterate
    y::Tx = copy(x)                   # null step iterate
    x0::Tx = copy(x)                   # null step iterate
    fmin::R = 0.0                 # best iterate
    g::Any
    α::Tx = [real(eltype(x))(1.0)]                  # qp dual values
    f_x::R = real(eltype(x))(0.0)                 # value of f at x
    f_y::R = real(eltype(x))(0.0)                 # value of f at y
    g_x::R = real(eltype(x))(0.0)                 # value of g at x
    sf_x::Tx = zero(x)                # subgradient of f at x
    sφ_x::Tx = zero(x)                # subgradient of model function φ at x
    rho::R = real(eltype(x))(1.0)                 # stepsize parameter of forward and backward steps
    sₖ::AbstractArray{Tx} = []   # subgradient history
    fₖ::AbstractArray{R} = []   # function value history
    eₖ::AbstractArray{R} = []    # error history
    δₖ::R = 1.0  # error history
    εₖ::R = 1.0  # error history
    subsolver::Any = ProximalAlgorithms.SFISTA(tol=1e-10)
    null_steps::Int = 0
    descent_steps::Int = 0
end

# Storage for the proximal subproblem for the L∞ regularized matrix game
Base.@kwdef mutable struct ProximalBilinearSPP
    game::BilinearSPP # Game struct
    stepsize::Float64 # Prox stepsize
    x::Vector{Float64} = zeros(game.N) # Current x iterate
    y::Vector{Float64} = zeros(game.M)  # Current y iterate
    x⁺::Vector{Float64} = zeros(game.N) # Cache for next x iterate
    y⁺::Vector{Float64} = zeros(game.M) # Cache for next y iterate
    x̃::Vector{Float64} = zeros(game.N) # Cache for the best x iterate seen so far (during this prox step)
    ỹ::Vector{Float64} = zeros(game.M) # Cache for the best y iterate seen so far (during this prox step)
    setting::Symbol = :Min # Symbol to denote the step type (minimization over x or maximization over y)
    gradient_cache_x::Vector{Float64} = zeros(game.N) # Subgradient cache for x
    gradient_cache_y::Vector{Float64} = zeros(game.M) # Supergradient cache for y
    h = IndSimplex() # Simplex indicator function
end

# Bundle Management, if cut limit is reached then aggregate using dual multipliers 
function bundle_management!(state::ProxBundleState, memory::Int)

    pnk = size(state.sₖ, 1)
    if pnk <= memory
        return 0
    end

    # Perform cut aggregation
    state.sₖ = [sum([αi * si for (αi, si) in zip(state.α, state.sₖ)])]
    state.fₖ = [sum([αi * fi for (αi, fi) in zip(state.α, state.fₖ)])]
end


# Compute the subgradient of the Proximal subproblem (negated for the supergradient)
function gradient(f::ProximalBilinearSPP, x)
    if f.setting == :Min
        subgradient!(f.game, f.gradient_cache_x, x, f.y)
        return f.gradient_cache_x
    elseif f.setting == :Max
        supergradient!(f.game, f.gradient_cache_y, f.x, x)
        return -f.gradient_cache_y
    else
        throw(ArgumentError("Unrecognized subproblem setting: The subproblem setting should either be :Min or :Max"))
    end
end

# initialize the Proximal struct for the next iteration
function init_prox_step!(pgame::ProximalBilinearSPP, setting::Symbol)
    @assert setting == :Min || setting == :Max
    fill!(pgame.gradient_cache_x, 0.0)
    fill!(pgame.gradient_cache_y, 0.0)
    pgame.setting = setting
end

# Easy evaluation function for the prox subproblem
function (a::ProximalBilinearSPP)(x)
    if a.setting == :Min
        return eval(a.game, x, a.y)
    else
        return -eval(a.game, a.x, x)
    end

end


# Bisection search for the univariate proximal problem
function _twocut_maximize(a₁::Tx, b₁::R, a₂::Tx, b₂::R, λ::R, x₀::Tx, h::Any; tol::R=1e-12) where {R,Tx}
    τ_low = 0.0
    τ_high = 1.0
    x⁺ = zeros(size(x₀))
    τ = -1.0
    dϕ = 1.0
    niters = 0
    while abs(dϕ) > tol
        if (τ_high - τ_low) <= tol || abs(dϕ) < tol
            break
        end
        τ = (τ_high + τ_low) / 2
        niters += 1
        dϕ = (a₁ - a₂)' * x₀ - λ * (τ * a₁' * a₁ + (1 - 2τ)a₁' * a₂ - (1 - τ) * a₂' * a₂)
        # If derivative is negative, then decrease (seach for maximum)
        if dϕ < 0
            τ_high = τ
        else
            τ_low = τ
        end

    end
    # Return the optimal x given τ
    prox!(x⁺, h, x₀ - λ * (τ * a₁ - (1 - τ) * a₂), λ)

    return τ, niters
end


# Solve the two cut univariate dual problem
function subproblem_2cut!(game::ProximalBilinearSPP, state::ProxBundleState{R,Tx}) where {R,Tx}

    τ, niters = _twocut_maximize(state.sₖ[1], state.fₖ[1], gradient(game, state.y), game(state.y), state.rho, state.x0, game.h)
    state.α = [τ]
    prox!(state.y, game.h, state.x0 - state.rho * (τ * state.sₖ[1] + (1 - τ) * state.sf_x))
    return niters
end


# (sub/super)Gradient functions for the multicut subproblem
function ProximalAlgorithms.gradient(prob::ProxBundleSubproblem, λ)
    y, _ = prox(prob.h, prob.x0 - prob.θ * prob.A * λ, prob.θ)
    value = (prob.A * λ)' * y + λ' * prob.b + prob.h(y) + 1 / (2prob.θ) * norm(y - prob.x0)^2
    return -(prob.A' * y + prob.b), -value
end
function ProximalAlgorithms.gradient!(z, prob::ProxBundleSubproblem, λ)
    y, _ = prox(prob.h, prob.x0 - prob.θ * prob.A * λ, prob.θ)
    # println("grad ", -(prob.A' * y + prob.b))
    z .= -(prob.A' * y + prob.b)
end
# Function evaluation for the multicut subproblem
function (prob::ProxBundleSubproblem)(λ)
    y, _ = prox(prob.h, prob.x0 - prob.θ * prob.A * λ, prob.θ)
    (prob.A * λ)' * y + λ' * prob.b + prob.h(y) + 1 / (2prob.θ) * norm(y - prob.x0)^2
end

# Solve the dual maximization problem for the multicut auxiliary step 
function solve_composite!(state::ProxBundleState)
    perm = randperm(size(state.sₖ, 1)) # Randomize the permutations to break ties as needed to avoid cycling
    S = hcat(state.sₖ[perm]...)
    h = IndSimplex()
    subprob = ProxBundleSubproblem(x0=state.x0, A=S, b=state.fₖ[perm], h=state.g, θ=state.rho)
    state.α, iters = state.subsolver(f=subprob, Lf=norm(S)^2 * state.rho, g=h, x0=ones(size(S, 2)) ./ size(S, 2))
    state.y, _ = prox(state.g, state.x0 - state.rho * S * state.α)

    state.α = state.α[sortperm(perm)]
    indices = findall(<(1e-6), state.α)
    deleteat!(state.fₖ, indices)
    deleteat!(state.sₖ, indices)
    deleteat!(state.α, indices)
    return iters
end

# Evaluate the bundle model
function eval_bundle(
    state::ProxBundleState
)
    value = maximum(
        [
        fk + sk' * state.y for (fk, sk) in zip(state.fₖ, state.sₖ)
    ]
    )
    return value
end


# Implementation of the Primal-Dual Cutting Plane (PDCP) subroutine
function PDCP!(game::ProximalBilinearSPP, ε::Float64, memory::Int=20)
    # Start by building the model
    if game.setting == :Min
        state = ProxBundleState(x=copy(game.x), g=game.h, rho=game.stepsize)
    else
        state = ProxBundleState(x=copy(game.y), g=game.h, rho=game.stepsize)
    end

    # Initialize state parameters
    state.sf_x = gradient(game, state.x)
    state.f_x = game(state.x)
    fcomp = game(state.y)
    state.fmin = fcomp
    state.fₖ = [state.f_x - state.x' * state.sf_x]
    state.sₖ = [copy(state.sf_x)]
    nsteps = 0
    tj = 1.0

    # While IPP condition not satisfied
    while tj > ε

        # Find the exact minimizer to the bundle model
        if memory > 2 # multicut model
            nsteps += solve_composite!(state)
            # elseif memory == 2
        elseif memory == 2 # 2-cut
            nsteps += subproblem_2cut!(game, state)
        elseif memory == 1 # 1-cut
            prox!(state.y, state.g, state.x0 - state.rho * state.sₖ[1])
            nsteps += 1
        end
        # Compute the loop termination sequence tj
        fcomp = game(state.x)
        fy = eval_bundle(state)
        normx = 1 / (2 * state.rho) * norm(state.x - state.x0)^2
        normy = 1 / (2 * state.rho) * norm(state.y - state.x0)^2
        tj = fcomp + normx - (fy + normy)

        # Update the best point if we have a decrease in function value
        ϕy = game(state.y)
        sf_y = gradient(game, state.y)
        if ϕy <= fcomp - 1e-10
            copy!(state.x, state.y)
        end

        # Update the bundle model
        τⱼ = (nsteps - 1) / nsteps
        if memory == 1
            state.sₖ = [state.sₖ[1] * τⱼ + sf_y * (1 - τⱼ)]
            state.fₖ = [state.fₖ[1] * τⱼ + (ϕy - sf_y' * state.y) * (1 - τⱼ)]
        else
            push!(state.sₖ, copy(sf_y))
            push!(state.fₖ, ϕy - sf_y' * state.y)
            bundle_management!(state, memory)
        end
    end
    # Copy back the final iterates
    if game.setting == :Min
        copy!(game.x⁺, state.y)
        copy!(game.x̃, state.x)
    else
        copy!(game.y⁺, state.y)
        copy!(game.ỹ, state.x)
    end
    return nsteps
end

# Implementation of PB-SPP
function bundle_saddle_point(game::BilinearSPP, target_accuracy::Float64; memory::Int=typemax(Int), log_frequency=10)
    # Compute the upper estimate on the subgradient bound
    Mx = maximum([norm(game.A[i, :]) for i in 1:game.M]) + game.γx
    My = maximum([norm(game.A[:, i]) for i in 1:game.N]) + game.γy
    M = max(Mx, My)

    # Use dynamic stepsize sequence
    λ₁ = 1 / 4M
    # Allocate storage for proximal subproblems
    proxgame = ProximalBilinearSPP(game=game, stepsize=λ₁)

    # Initialize to uniform random distributions
    proxgame.x .= ones(game.N) / game.N
    proxgame.y .= ones(game.M) / game.M
    copy!(proxgame.x̃, proxgame.x)
    copy!(proxgame.ỹ, proxgame.y)
    x̄ = zeros(game.N)
    ȳ = zeros(game.M)
    inner_steps = 0
    starttime = time_ns()
    # data = []
    println("elapsed_time(s),inner_steps,outer_steps,pd_gap")
    for k in 1:300000
        proxgame.stepsize = λ₁ / sqrt(k)
        # Update the ergodic averages
        axpby!(1 / k, proxgame.x̃, (k - 1) / k, x̄)
        axpby!(1 / k, proxgame.ỹ, (k - 1) / k, ȳ)

        # Log current iterate status
        if (k - 1) % log_frequency == 0
            println(round((time_ns() - starttime) / 1e9, sigdigits=6), ",", inner_steps, ",", k, ",", primalv(game, x̄) - dualv(game, ȳ))
            flush(stdout)
        end
        # If target_accuracy is reached, terminate
        if k % 100 == 0 && primalv(game, x̄) - dualv(game, ȳ) <= target_accuracy
            return
        end

        # Solve the proximal subproblems
        init_prox_step!(proxgame, :Min)
        inner_steps += PDCP!(proxgame, target_accuracy / 4, memory)
        init_prox_step!(proxgame, :Max)
        inner_steps += PDCP!(proxgame, target_accuracy / 4, memory)

        # Update the main iterates
        copy!(proxgame.x, proxgame.x⁺)
        copy!(proxgame.y, proxgame.y⁺)

    end
end





# Test functions
function test_solve_composite()
    gen = Xoshiro(0)
    game = generate_random_zero_sum(5, 5, 0.4, 10.0; seed=1)
    stepsize = 0.1

    proxgame = ProximalBilinearSPP(game=game, stepsize=stepsize)
    x0 = normalize(rand(gen, game.N), 1)
    proxgame.y = normalize(rand(gen, game.M), 1)
    proxgame.x = x0
    init_prox_step!(proxgame, :Min)
    state = ProxBundleState(x=copy(proxgame.x), rho=proxgame.stepsize, g=proxgame.h)
    p = 40
    for i in 1:p
        y = normalize(rand(gen, game.N), 1)
        ∇f = gradient(proxgame, y)
        fp = proxgame(state.x)
        push!(state.fₖ, fp - y' * ∇f)
        push!(state.sₖ, copy(∇f))
    end
    solve_composite!(state)
    strength = 1e-2
    f_y = eval_bundle(state) + 1 / (2 * state.rho) * norm(state.y - state.x0)^2
    cache = copy(state.y)
    println(state.y)
    for i in 1:100000
        state.y, _ = prox(proxgame.h, state.y + (strength .* rand(gen, size(state.y, 1))))
        f_other = eval_bundle(state) + 1 / (2 * state.rho) * norm(state.y - state.x0)^2
        @test f_other >= f_y
        state.y .= cache
    end


end

function test_solve_2cut()
    gen = Xoshiro(0)
    game = generate_random_zero_sum(20, 20, 0.4, 10.0; seed=1)
    stepsize = 0.1

    proxgame = ProximalBilinearSPP(game=game, stepsize=stepsize)
    x0 = normalize(rand(gen, game.N), 1)
    proxgame.y = normalize(rand(gen, game.M), 1)
    proxgame.x = x0
    init_prox_step!(proxgame, :Min)
    state = ProxBundleState(x=copy(proxgame.x), rho=proxgame.stepsize, g=proxgame.h)
    p = 1
    for i in 1:p
        y = normalize(rand(gen, game.N), 1)
        ∇f = gradient(proxgame, y)
        fp = proxgame(state.x)
        push!(state.fₖ, fp - y' * ∇f)
        push!(state.sₖ, copy(∇f))
    end
    subproblem_2cut!(proxgame, state)
    strength = 1e-3
    f_y = eval_bundle(state) + 1 / (2 * state.rho) * norm(state.y - state.x0)^2
    cache = copy(state.y)
    for i in 1:1000000
        state.y, _ = prox(proxgame.h, state.y + (strength .* rand(gen, size(state.y, 1))))
        f_other = eval_bundle(state) + 1 / (2 * state.rho) * norm(state.y - state.x0)^2
        @test f_other >= f_y
        state.y .= cache
    end


end