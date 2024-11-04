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
include("utils/WarmStart.jl")


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
    horizon = 5

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
    observation_model_inverse = (; σ = 0.0, expected_observation = x -> x)

    for_sol = reconstruct_solution(forward_solution, game.game, horizon)

    for_sol = observation_model_inverse.expected_observation(for_sol)

    warm_start_sol = 
        expand_warm_start(
            warm_start(
                for_sol,
                initial_state,
                horizon;
                observation_model = observation_model_inverse),
            mcp_game)

    context_state_estimation, last_solution, i_, solving_info, time_exec = 
        solve_inverse_mcp_game(
            mcp_game,
            initial_state,
            for_sol,
            context_state_guess,
            horizon;
            observation_model = observation_model_inverse,         
            max_grad_steps = 500,
            last_solution = warm_start_sol
        )

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
    rng = Random.MersenneTwister(42),
)
    CairoMakie.activate!();

    environment = PolygonEnvironment(6, 8)
    game = n_player_collision_avoidance(2; environment, min_distance = 1.2)
    # baseline_game = n_player_collision_avoidance(2; environment, min_distance = 1.2, myopic = false)

    if min(goal[Block(1)][3], goal[Block(2)][3]) == 1
        horizon = 75
    else
        horizon = convert(Int64, ceil(log(1e-4)/(log(min(goal[Block(1)][3], goal[Block(2)][3])))))
    end
    # horizon = 5
    num_trials = 3

    turn_length = 2
    solver = MCPCoupledOptimizationSolver(game.game, horizon, blocksizes(goal, 1))
    # baseline_solver = MCPCoupledOptimizationSolver(baseline_game.game, horizon, blocksizes(goal, 1))
    mcp_game = solver.mcp_game
    # baseline_mcp_game = baseline_solver.mcp_game
    player_state_dimension = convert(Int64, state_dim(game.game.dynamics)/2)

    forward_solution = solve_mcp_game(mcp_game, initial_state, goal; verbose = true)
    # baseline_forward_solution = solve_mcp_game(baseline_mcp_game, initial_state, goal; verbose = true)
    for_sol = reconstruct_solution(forward_solution, game.game, horizon)
    # baseline_for_sol = reconstruct_solution(baseline_forward_solution, baseline_game.game, horizon)
    # Just holds states, [ [[] = p1 state [] = p2 state] ... horizon ]

    σs = [0.005*i for i in 0:10]
    # σs = [0.01]

    context_state_guess = sample_initial_states_and_context(game, horizon, rng, 0.08)[2]
    context_state_guess[1] = 0.0
    context_state_guess[2] = -2.7
    context_state_guess[4] = 2.7
    context_state_guess[5] = 1.0

    baseline_context_state_guess = sample_initial_states_and_context(game, horizon, rng, 0.08)[2]
    baseline_context_state_guess[1] = 0.0
    baseline_context_state_guess[2] = -2.7
    baseline_context_state_guess[3] = 1
    baseline_context_state_guess[4] = 2.7
    baseline_context_state_guess[5] = 1.0
    
    errors = Array{Float64}(undef, length(σs), num_trials)
    baseline_errors = Array{Float64}(undef, length(σs), num_trials)
    failure_counter = 0
    num_attempts_if_failed = 3

    parameter_error = Array{Float64}(undef, length(σs), num_trials)
    baseline_parameter_error = Array{Float64}(undef, length(σs), num_trials)

    partial_observation_function_generator = (σ) -> (x) -> vcat(x[1:2], x[5:6]) + σ * randn(rng, 4)
    per_player_expected_partial_observation_model = (; expected_observation = x -> x[1:2])
    expected_partial_observation_model = (; expected_observation = (x) -> vcat(x[1:2], x[5:6]) )

    # σ_indices = enumerate(σs)

    for idx in eachindex(σs)
        σ = σs[idx]
        for i in 1:num_trials
            # println("std: ", σ, " trial: ", i, " thread: ", Threads.threadid())
            attempts = 1
            while (attempts <= num_attempts_if_failed)
                # println("\tattempt: ", attempts)
                observed_trajectory = draw_observations(
                    for_sol,
                    partial_observation_function_generator(σ))
                try
                    inv_game_sol = solve_inverse_mcp_game(
                        mcp_game,
                        initial_state,
                        observed_trajectory,
                        context_state_guess,
                        horizon;
                        observation_model = expected_partial_observation_model,
                        max_grad_steps = 150,
                        last_solution = expand_warm_start(
                                            warm_start(
                                                observed_trajectory,
                                                initial_state,
                                                horizon;
                                                observation_model = per_player_expected_partial_observation_model,
                                                partial_observation_state_size = 2),
                                            mcp_game))
                    reconstructed_sol = reconstruct_solution(
                            inv_game_sol[2],
                            game.game,
                            horizon)
                    error = norm_sqr(for_sol - reconstructed_sol)
                    # println("context state: ", context_state_guess) # .951916339835734 as gamma every time
                    errors[idx, i] = error
                    # println("\tmyopic solver done")
                    baseline_inv_game_sol = solve_inverse_mcp_game(
                        mcp_game,
                        initial_state,
                        observed_trajectory,
                        baseline_context_state_guess,
                        horizon;
                        observation_model = expected_partial_observation_model,
                        max_grad_steps = 150,
                        last_solution = expand_warm_start(
                                            warm_start(
                                                observed_trajectory,
                                                initial_state,
                                                horizon;
                                                observation_model = per_player_expected_partial_observation_model,
                                                partial_observation_state_size = 2),
                                            mcp_game
                                        ))
                    baseline_reconstructed_sol = reconstruct_solution(
                            baseline_inv_game_sol[2],
                            game.game,
                            horizon)
                    baseline_error = norm_sqr(for_sol - baseline_reconstructed_sol)
                    baseline_errors[idx, i] = baseline_error
                    # println("\tnon-myopic solver done")

                    # TODO: not sure how to get actual context state...
                    parameter_error[idx, i] = norm_sqr(inv_game_sol[1] - context_state_guess)
                    baseline_parameter_error[idx, i] = norm_sqr(baseline_inv_game_sol[1] - context_state_guess)
                    break
                catch e
                    # rethrow(e)
                    println("\tfailed")
                    attempts += 1
                end
            end
            if attempts > num_attempts_if_failed
                failure_counter += 1
            end
        end
        # push!(errors, [])
    end

    println("Failure Counter: ", failure_counter, " / ", num_trials * length(σs))

    open("experiments.tmp.txt", "w") do file
        write(file, "our errors, then baseline errors. the rows are different noise levels, and the columns are different trials\n")
        for i in 1:size(errors, 1)
            write(file, string(σs[i]) * ": ")
            # write(file, "\t")
            write(file, string(errors[i, :]))
            write(file, "\n")
        end
        write(file, "\n")
        for i in 1:size(baseline_errors, 1)
            write(file, string(σs[i]) * ": ")
            # write(file, "\t")
            write(file, string(baseline_errors[i, :]))
            write(file, "\n")
        end

        write(file, "parameter errors:")
        for i in 1:size(parameter_error, 1)
            write(file, string(σs[i]) * ": ")
            # write(file, "\t")
            write(file, string(parameter_error[i, :]))
            write(file, "\n")
        end
        write(file, "\n")
        for i in 1:size(baseline_parameter_error, 1)
            write(file, string(σs[i]) * ": ")
            # write(file, "\t")
            write(file, string(baseline_parameter_error[i, :]))
            write(file, "\n")
        end
    end

    fig1 = CairoMakie.Figure()
    ax1 = CairoMakie.Axis(fig1[1, 1])

    mean_errors = [Statistics.mean(errors[i, :]) for i in 1:size(errors, 1)]
    stds = [Statistics.std(errors[i, :]) for i in 1:size(errors, 1)]

    baseline_mean_errors = [Statistics.mean(baseline_errors[i, :]) for i in 1:size(baseline_errors, 1)]
    baseline_stds = [Statistics.std(baseline_errors[i, :]) for i in 1:size(baseline_errors, 1)]
    # variances = [std / sqrt(length(error)) for (std, error) in zip(stds, errors)]
    # Originally, in errorbars, we are plotting variance? but std seems more likely?
    CairoMakie.scatter!(ax1, σs, mean_errors, color = (:red, 0.5))
    CairoMakie.errorbars!(ax1, σs, mean_errors, stds, color = (:red, 0.5))

    CairoMakie.scatter!(ax1, σs, baseline_mean_errors, color = (:blue, 0.5))
    CairoMakie.errorbars!(ax1, σs, baseline_mean_errors, baseline_stds, color = (:blue, 0.5))
    
    CairoMakie.save("NoiseGraph_warm_start.png", fig1)

end

#observation_model should  be full_state observation, indexed by time not by player
function draw_observations(full_state_trajectory, observation_model; num_players = 2)
    observation_length = length(observation_model(full_state_trajectory[Block(1)]))
    # observations = []
    # for full_state_time_slice in full_state_trajectory.blocks
    #     push!(observations, observation_model(full_state_time_slice))
    # end
    BlockVector(
        vcat(
            [observation_model(state) for state in full_state_trajectory.blocks]...),
        [observation_length for _ in 1:length(full_state_trajectory.blocks)])
end

function get_observed_trajectory(trajectory, observation_model)
    observations = []
    observation_length = length(observation_model(trajectory[Block(1)]))
    for full_state_time_slice in trajectory.blocks
        push!(observations, observation_model(full_state_time_slice))
    end
    BlockVector(vcat(observations...),
        [observation_length for _ in eachindex(trajectory.blocks)])
end


function thread_test()
    print("using ", Threads.nthreads(), " threads\n")
    jjj = [0.005*i for i in 0:10]

    # @infiltrate
    Threads.@threads for i in eachindex(jjj)
        println(i, "^2, ", jjj[i], " by thread ", Threads.threadid())
    end
end
end
