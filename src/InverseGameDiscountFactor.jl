module InverseGameDiscountFactor

using DifferentiableTrajectoryOptimization:
    ParametricTrajectoryOptimizationProblem, get_constraints_from_box_bounds, _coo_from_sparse!
using TrajectoryGamesExamples:
    TrajectoryGamesExamples,
    PolygonEnvironment,
    two_player_meta_tag,
    animate_sim_steps,
    planar_double_integrator,
    UnicycleDynamics,
    create_environment_axis
    # BicycleDynamics
using TrajectoryGamesBase:
    TrajectoryGamesBase,
    TrajectoryGame,
    get_constraints,
    num_players,
    state_dim,
    control_dim,
    horizon,
    state_bounds,
    control_bounds,
    solve_trajectory_game!,
    JointStrategy,
    RecedingHorizonStrategy,
    rollout,
    ProductDynamics
using LiftedTrajectoryGames: LiftedTrajectoryStrategy
using BlockArrays: Block, BlockVector, mortar, blocksizes
using SparseArrays: sparse, blockdiag, findnz, spzeros
using PATHSolver: PATHSolver
using LinearAlgebra: I, norm_sqr, pinv, ColumnNorm, qr, norm
using Random: Random
using ProgressMeter: ProgressMeter
# using GLMakie: GLMakie
using CairoMakie
using Symbolics: Symbolics, @variables, scalarize
using ChainRulesCore: ChainRulesCore
using ForwardDiff: ForwardDiff
using Zygote: Zygote
using Flux: Flux, glorot_uniform, Dense, Optimise.Adam, NNlib.relu, Chain
using ParametricMCPs: ParametricMCPs
using Makie: Makie
using Colors: @colorant_str
using JLD2: JLD2
using Statistics
using ParametricMCPs

using Infiltrator

include("utils/ExampleProblems.jl")
using .ExampleProblems: n_player_collision_avoidance, two_player_guidance_game, CollisionAvoidanceGame, HighwayGame


include("utils/utils.jl")
include("problem_formulation.jl")
include("solve.jl")
include("baseline/inverse_MCP_solver.jl")


function main(;
    initial_state = mortar([
        [-1, 2.5, 0.1, -0.2],
        [1, 2.8, 0.0, 0.0],
    ]),
    goal = mortar([[0.0, -2.7, 0.9], [2.7, 1, 0.9]]),
    plot_please = true,
    simulate_please = true
)

    """
    Player Colors:
    1: Red
    2: Magenta
    3: Purple
    4: Blue
    """
    """
    An example of the MCP game solver
    """
    CairoMakie.activate!();
    environment = PolygonEnvironment(6, 8)
    game = n_player_collision_avoidance(2; environment, min_distance = 1.2)

    if min(goal[Block(1)][3], goal[Block(2)][3]) == 1
        horizon = 75
    else
        horizon = convert(Int64, ceil(log(1e-4)/(log(min(goal[Block(1)][3], goal[Block(2)][3])))))
    end

    turn_length = 2
    solver = MCPCoupledOptimizationSolver(game.game, horizon, blocksizes(goal, 1))
    mcp_game = solver.mcp_game

    forward_solution = solve_mcp_game(mcp_game, initial_state, goal; verbose = true)

    initial_state_guess, context_state_guess = sample_initial_states_and_context(game, horizon, Random.MersenneTwister(1), 0.08)

    context_state_guess[1] = 0
    context_state_guess[2] = -2.7

    context_state_guess[4] = 2.7
    context_state_guess[5] = 1

    println("Initial State Guess: ", initial_state_guess)
    println("Context State Guess: ", context_state_guess)

    observation_model_noisy = (; σ = 0.1, expected_observation = x -> x[1:2] .+ observation_model_noisy.σ * randn(length(x[1:2])))
    observation_model_inverse = (; σ = 0.0, expected_observation = x -> x[1:2])

    for_sol = reconstruct_solution(forward_solution, game.game, horizon)

    for_sol = observation_model_noisy.expected_observation(for_sol)

    # @infiltrate

    context_state_estimation, last_solution, i_, solving_info, time_exec = solve_inverse_mcp_game(mcp_game, initial_state, for_sol, context_state_guess, horizon;
                                                                            observation_model = observation_model_inverse,         
                                                                            max_grad_steps = 500)

    println("Context State Estimation: ", context_state_estimation)
    sol_error = norm_sqr(reconstruct_solution(forward_solution, game.game, horizon) - reconstruct_solution(last_solution, game.game, horizon))
    # @infiltrate
    println("Solution Error: ", sol_error)


    if plot_please
        for_solution = reconstruct_solution(forward_solution, game.game, horizon)
        inv_solution = reconstruct_solution(last_solution, game.game, horizon)
        # @infiltrate

        fig1 = CairoMakie.Figure()
        ax1 = CairoMakie.Axis(fig1[1, 1])

        for ii in 1:horizon
            CairoMakie.scatter!(ax1, for_solution[Block(ii)][1], for_solution[Block(ii)][2], color = :red)
            CairoMakie.scatter!(ax1, for_solution[Block(ii)][5], for_solution[Block(ii)][6], color = :blue)
            CairoMakie.scatter!(ax1, inv_solution[Block(ii)][1], inv_solution[Block(ii)][2], color = :purple)
            CairoMakie.scatter!(ax1, inv_solution[Block(ii)][5], inv_solution[Block(ii)][6], color = :magenta)
            # # GLMakie.scatter!(ax, x[9],x[10], color = :purple)
            # GLMakie.scatter!(ax, x[13],x[14], color = :magenta)
        end

        CairoMakie.save("SolutionPlot.png", fig1)
    end


    if simulate_please

        sim_steps1 = let
            n_sim_steps = 150
            progress = ProgressMeter.Progress(n_sim_steps)
            receding_horizon_strategy =
                WarmStartRecedingHorizonStrategy(; solver, game.game, turn_length, context_state = goal)
            rollout(
                game.game.dynamics,
                receding_horizon_strategy,
                initial_state,
                n_sim_steps;
                get_info = (γ, x, t) ->
                    (ProgressMeter.next!(progress); γ.receding_horizon_strategy),
            )
        end
        if context_state_estimation[3] > 1
            context_state_estimation[3] = 1
        end
        if context_state_estimation[6] > 1
            context_state_estimation[6] = 1
        end
        
        context_state_estimation = mortar([context_state_estimation[1:3], context_state_estimation[4:6]])

        sim_steps2 = let
            n_sim_steps = 150
            progress = ProgressMeter.Progress(n_sim_steps)
            receding_horizon_strategy =
                WarmStartRecedingHorizonStrategy(; solver, game.game, turn_length, context_state = context_state_estimation)
            rollout(
                game.game.dynamics,
                receding_horizon_strategy,
                initial_state,
                n_sim_steps;
                get_info = (γ, x, t) ->
                    (ProgressMeter.next!(progress); γ.receding_horizon_strategy),
            )
        end
        
        animate_sim_steps(game.game, sim_steps1; live = false, framerate = 20, show_turn = true, heading = "Forward Solution", filename = "ForwardSolution")
        animate_sim_steps(game.game, sim_steps2; live = false, framerate = 20, show_turn = true, heading = "Inverse Solution", filename = "InverseSolution")
        
    end
    (; sol_error, context_state_estimation)
end

function GenerateNoiseGraph(
    initial_state = mortar([
        [-1, 2.5, 0.1, -0.2],
        [1, 2.8, 0.0, 0.0],
    ]),
    goal = mortar([[0.0, -2.7, 0.9], [2.7, 1, 0.9]]),
    rng = Random.MersenneTwister(1),
)
    num_trials = 10;
    CairoMakie.activate!();

    environment = PolygonEnvironment(6, 8)
    game = n_player_collision_avoidance(2; environment, min_distance = 1.2)

    if min(goal[Block(1)][3], goal[Block(2)][3]) == 1
        horizon = 75
    else
        horizon = convert(Int64, ceil(log(1e-4)/(log(min(goal[Block(1)][3], goal[Block(2)][3])))))
    end

    turn_length = 2
    solver = MCPCoupledOptimizationSolver(game.game, horizon, blocksizes(goal, 1))
    mcp_game = solver.mcp_game

    forward_solution = solve_mcp_game(mcp_game, initial_state, goal; verbose = true)
    for_sol = reconstruct_solution(forward_solution, game.game, horizon)

    σs = [0.01*i for i in 0:10]
    # σs = [0.01]

    context_state_guess = sample_initial_states_and_context(game, horizon, rng, 0.08)[2]
    context_state_guess[1] = 0.0
    context_state_guess[2] = -2.7
    context_state_guess[4] = 2.7
    context_state_guess[5] = 1.0

    # TODO: check percentage of non-converging solver attempts ✓
    # try perturbing initial state ✓
    # check if noise makes initial state infeasible 
    # different random seeds

    # errors = [[
    #     norm_sqr(
    #         for_sol - 
    #         reconstruct_solution(
    #             solve_inverse_mcp_game(
    #                 mcp_game,
    #                 initial_state,
    #                 for_sol .+ σ * randn(rng, length(for_sol)),
    #                 context_state_guess,
    #                 horizon;
    #                 max_grad_steps = 150)[2],
    #             game.game,
    #             horizon)) for _ in num_trials] for σ in eachindex(σs)]
    errors = Array{Float64}(undef, length(σs), num_trials)
    failure_counter = 0
    num_attempts_if_failed = 3

    parametric_observation_function = (σ) -> (x) -> x[1:2] .+ σ * randn(rng, length(x[1:2]))
    max_likelihood_observation = (x) -> x[1:2]
    f = (x) -> norm_sqr(max_likelihood_observation(x) - raw_observation)


    # Ls = map(zip(1:N, fs, gs, hs)) do (i, f, g, h)
    #     f - λs[Block(i)]' * g - μs[Block(i)]' * h - λ̃' * g̃ - μ̃' * h̃
    # end
    # # Build F = [∇ₓLs, gs, hs, g̃, h̃]'.
    # ∇ₓLs = map(zip(Ls, blocks(xs))) do (L, x)
    #     Symbolics.gradient(L, x)
    # end
    # F = [reduce(vcat, ∇ₓLs); reduce(vcat, gs); reduce(vcat, hs);g̃;h̃]


    warm_start_mcp = ParametricMCPs.ParametricMCP(
                        f, # gotta be wrong
                        [-Inf for _ in 1:size(forwards_solution.primals, 1)],
                        [Inf for _ in 1:size(forwards_solution.primals, 1)],
                        length(context_state_guess),
                    )
    
    for (idx, σ) in enumerate(σs)
        observation_function = parametric_observation_function(σ)
        for i in 1:num_trials
            println("std: ", σ, " trial: ", i)
            attempts = 1
            println("\tattempt: ", attempts)

            while (attempts < num_attempts_if_failed)
                try
                    raw_observation = observation_function(for_sol)
                    
                    warm_start_sol = ParametricMCPs.solve( # where did feasibility constraints go?
                        warm_start_mcp, # need to instantiate, probably min statement
                        context_state_guess; # parameter_value
                        initial_guess = zeros(size(forwards_solution.primals)), # initial_guess?
                        verbose = false, # verbose
                        cumulative_iteration_limit = 100000,
                        proximal_perturbation = 1e-2,
                        use_basics = true,
                        use_start = true,
                    )
                    error = norm_sqr(
                        for_sol - 
                        reconstruct_solution(
                            solve_inverse_mcp_game(
                                mcp_game,
                                initial_state,
                                warm_start_sol,
                                context_state_guess,
                                horizon;
                                observation_model = (; expected_observation = x -> x[1:2]),
                                max_grad_steps = 150)[2],
                            game.game,
                            horizon))
                    errors[idx, i] = error
                    attempts = num_attempts_if_failed + 1
                    break
                catch
                    println("\tfailed")
                    attempts += 1
                end
            end
            if attempts == num_attempts_if_failed
                failure_counter += 1
            end
        end
        # push!(errors, [])
    end

    println("Failure Counter: ", failure_counter, " / ", num_trials * length(σs))

    fig1 = CairoMakie.Figure()
    ax1 = CairoMakie.Axis(fig1[1, 1])

    mean_errors = [Statistics.mean(errors[i, :]) for i in 1:size(errors, 1)]
    stds = [Statistics.std(errors[i, :]) for i in 1:size(errors, 1)]
    # variances = [std / sqrt(length(error)) for (std, error) in zip(stds, errors)]
    # Originally, in errorbars, we are plotting variance? but std seems more likely?
    CairoMakie.scatter!(ax1, σs, mean_errors, color = :red)
    CairoMakie.errorbars!(ax1, σs, mean_errors, stds, color = :red)
    
    CairoMakie.save("NoiseGraph_warm_start.png", fig1)

end
end
