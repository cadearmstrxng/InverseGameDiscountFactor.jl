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
        [0, 2, 0.1, -0.2],
        [2.5, 2, 0.0, 0.0],
    ]),
    hidden_params = mortar([[2, 0, 0.95], [0, 0, 0.9]]),
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
    game = n_player_collision_avoidance(2; environment, min_distance = 0.5, collision_avoidance_coefficient = 5.0)

    if max(hidden_params[Block(1)][3], hidden_params[Block(2)][3]) == 1
        horizon = 75
    else
        horizon = convert(Int64, ceil(log(1e-4)/(log(max(hidden_params[Block(1)][3], hidden_params[Block(2)][3]))) * 4/3))
    end
    horizon = 5
    println("horizon: ", horizon)

    turn_length = 2
    solver = MCPCoupledOptimizationSolver(game.game, horizon, blocksizes(hidden_params, 1))
    mcp_game = solver.mcp_game

    forward_solution = solve_mcp_game(mcp_game, initial_state, hidden_params; verbose = true)

    initial_state_guess, context_state_guess = sample_initial_states_and_context(game, horizon, Random.MersenneTwister(1), 0.08)

    # context_state_guess[1] = hidden_params[1]
    # context_state_guess[2] = hidden_params[2]

    # context_state_guess[4] = hidden_params[4]
    # context_state_guess[5] = hidden_params[5]

    println("Initial State Guess: ", initial_state_guess)
    println("Context State Guess: ", context_state_guess)

    observation_model_noisy = (; σ = 0.1, expected_observation = x -> vcat([x[4*i-3:4*i-2] .+ observation_model_noisy.σ * randn(length(x[i:i+1])) for i in 1:num_players(game.game)]...))
    # observation_model_inverse = (; σ = 0.0, expected_observation = x -> x)
    observation_model_warm_start = (; σ = 0.0, expected_observation = x -> vcat([x[4*i-3:4*i-2] for i in 1:num_players(game.game)]...))

    function inverse_expected_observation(x)
        # new_x = [x[i].value for i in 1:length(x)]
        # @infiltrate
        vcat([x[4*i-3:4*i-2] for i in 1:num_players(game.game)]...)
    end

    observation_model_inverse = (; σ = 0.0, expected_observation = x -> inverse_expected_observation(x))

    for_sol = reconstruct_solution(forward_solution, game.game, horizon)
    # @infiltrate

    for_sol = vcat([observation_model_noisy.expected_observation(state_t) for state_t in for_sol.blocks]...)
    for_sol = BlockVector(for_sol, [4 for _ in 1:horizon])
    # @infiltrate

    warm_start_sol = 
        expand_warm_start(
            warm_start(
                for_sol,
                initial_state,
                horizon;
                observation_model = observation_model_warm_start,
                partial_observation_state_size = 2),
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
        #TODO: do legend
    end


    if simulate_please

        sim_steps1 = let
            n_sim_steps = 150
            progress = ProgressMeter.Progress(n_sim_steps)
            receding_horizon_strategy =
                WarmStartRecedingHorizonStrategy(; solver, game.game, turn_length, context_state = hidden_params)
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
        [0, 2, 0.1, -0.2],
        [2.5, 2, 0.0, 0.0],
    ]),
    hidden_params = mortar([[2, 0, 0.95], [0, 0, 0.9]]),
    rng = Random.MersenneTwister(1),
)
    CairoMakie.activate!();

    environment = PolygonEnvironment(6, 8)
    game = n_player_collision_avoidance(2; environment, min_distance = 0.5, collision_avoidance_coefficient = 5.0)
    baseline_game = n_player_collision_avoidance(2; environment, min_distance = 0.5, collision_avoidance_coefficient = 5.0, myopic = false)

    if min(hidden_params[Block(1)][3], hidden_params[Block(2)][3]) == 1
        horizon = 75
    else
        horizon = convert(Int64, ceil(log(1e-4)/(log(min(hidden_params[Block(1)][3], hidden_params[Block(2)][3])))))
    end
    horizon = 50
    num_trials = 1

    turn_length = 2
    solver = MCPCoupledOptimizationSolver(game.game, horizon, blocksizes(hidden_params, 1))
    baseline_solver = MCPCoupledOptimizationSolver(baseline_game.game, horizon, [2, 2])

    mcp_game = solver.mcp_game
    baseline_mcp_game = baseline_solver.mcp_game

    player_state_dimension = convert(Int64, state_dim(game.game.dynamics)/2)

    forward_solution = solve_mcp_game(mcp_game, initial_state, hidden_params; verbose = true)
    # baseline_forward_solution = solve_mcp_game(baseline_mcp_game, initial_state, hidden_params; verbose = true)
    for_sol = reconstruct_solution(forward_solution, game.game, horizon)
    # baseline_for_sol = reconstruct_solution(baseline_forward_solution, baseline_game.game, horizon)
    # Just holds states, [ [[] = p1 state [] = p2 state] ... horizon ]

    # σs = [0.01*i for i in 0:1]
    σs = [0.0]

    context_state_guess = sample_initial_states_and_context(game, horizon, rng, 0.08)[2]
    context_state_guess[3] = 1.0
    context_state_guess[6] = 1.0
    println("context_state_guess: ", context_state_guess)

    baseline_context_state_guess = sample_initial_states_and_context(game, horizon, rng, 0.08, myopic=false)[2]
    baseline_context_state_guess[1:2] = context_state_guess[1:2]
    baseline_context_state_guess[3:4] = context_state_guess[4:5]
    println("baseline_context_state_guess: ", baseline_context_state_guess)

    errors = Array{Float64}(undef, length(σs), num_trials)
    baseline_errors = Array{Float64}(undef, length(σs), num_trials)
    

    parameter_error = Array{Float64}(undef, length(σs), num_trials)
    baseline_parameter_error = Array{Float64}(undef, length(σs), num_trials)

    parameter_cosine_error = Array{Float64}(undef, length(σs), num_trials)
    baseline_parameter_cosine_error = Array{Float64}(undef, length(σs), num_trials)

    observed_trajectories = []
    
    recovered_traj = []
    baseline_recovered_traj = []

    recovered_params = []
    baseline_recovered_params = []

    partial_observation_function_generator = (σ) -> (x) -> vcat(x[1:2], x[5:6]) + σ * randn(rng, 4)
    per_player_expected_partial_observation_model = (; expected_observation = x -> x[1:2])
    expected_partial_observation_model = (; expected_observation = (x) -> vcat(x[1:2], x[5:6]) )
    observation_model_inverse = (; σ = 0.0, expected_observation = x -> x)
    
    failure_counter = 0
    num_attempts_if_failed = 3
    # σ_indices = enumerate(σs)

    for idx in eachindex(σs)
        σ = σs[idx]
        for i in 1:num_trials
            println("std: ", σ, " trial: ", i, " thread: ", Threads.threadid())
            attempts = 1
            while (attempts <= num_attempts_if_failed)
                println("\tattempt: ", attempts)
                # observed_trajectory = draw_observations(
                #     for_sol,
                #     partial_observation_function_generator(σ))
                observed_trajectory = for_sol
                try
                    inv_game_sol = solve_inverse_mcp_game(
                        mcp_game,
                        initial_state,
                        observed_trajectory,
                        context_state_guess,
                        horizon;
                        # observation_model = expected_partial_observation_model,
                        observation_model = observation_model_inverse,
                        max_grad_steps = 500,
                        last_solution = expand_warm_start(
                                            warm_start(
                                                observed_trajectory,
                                                initial_state,
                                                horizon;
                                                # observation_model = per_player_expected_partial_observation_model,
                                                observation_model = observation_model_inverse,
                                                # partial_observation_state_size = 2),
                                            ),
                                            mcp_game))
                    reconstructed_sol = reconstruct_solution(
                            inv_game_sol[2],
                            game.game,
                            horizon)
                    println("\tmyopic solver done")
                    baseline_inv_game_sol = solve_inverse_mcp_game(
                        baseline_mcp_game,
                        initial_state,
                        observed_trajectory,
                        baseline_context_state_guess,
                        horizon;
                        # observation_model = expected_partial_observation_model,
                        observation_model = observation_model_inverse,
                        max_grad_steps = 150,
                        last_solution = expand_warm_start(
                                            warm_start(
                                                observed_trajectory,
                                                initial_state,
                                                horizon;
                                                # observation_model = per_player_expected_partial_observation_model,
                                                observation_model = observation_model_inverse,
                                                # partial_observation_state_size = 2),
                                            ),
                                                baseline_mcp_game))
                    baseline_reconstructed_sol = reconstruct_solution(
                            baseline_inv_game_sol[2],
                            baseline_game.game,
                            horizon)
                    baseline_error = norm_sqr(for_sol - baseline_reconstructed_sol)
                    baseline_errors[idx, i] = baseline_error
                    println("\tbaseline solver done")

                    # parameter_error[idx, i] = norm_sqr(inv_game_sol[1] - hidden_params)
                    # baseline_parameter_error[idx, i] = norm_sqr(baseline_inv_game_sol[1] - hidden_params)

                    # parameter_cosine_error[idx, i] = sum(inv_game_sol[1] .* hidden_params) / (norm(inv_game_sol[1]) * norm(hidden_params))
                    baseline_params = expand_baseline_params(baseline_inv_game_sol[1])

                    recovered_params = push!(recovered_params, inv_game_sol[1])
                    baseline_recovered_params = push!(baseline_recovered_params, baseline_params)

                    # Graphing updates
                    errors[idx, i] = norm_sqr(for_sol - reconstructed_sol)
                    baseline_errors[idx, i] = norm_sqr(for_sol - baseline_reconstructed_sol)

                    parameter_error[idx, i] = norm_sqr(inv_game_sol[1] - hidden_params)
                    baseline_parameter_error[idx, i] = norm_sqr(baseline_params - hidden_params)

                    parameter_cosine_error[idx, i] = sum(inv_game_sol[1] .* hidden_params) / (norm(inv_game_sol[1]) * norm(hidden_params))
                    baseline_parameter_cosine_error[idx, i] = sum(baseline_params .* hidden_params) / (norm(baseline_params) * norm(hidden_params))

                    push!(observed_trajectories, observed_trajectory)
                    push!(recovered_traj, reconstructed_sol)
                    push!(baseline_recovered_traj, baseline_reconstructed_sol)
                    
                    # for_solution = reconstruct_solution(forward_solution, game.game, horizon)
                    # inv_solution = reconstruct_solution(last_solution, game.game, horizon)
                    # @infiltrate

                    fig1 = CairoMakie.Figure()
                    ax1 = CairoMakie.Axis(fig1[1, 1])

                    for ii in 1:horizon
                        CairoMakie.scatter!(ax1, for_sol[Block(ii)][1], for_sol[Block(ii)][2], color = :red)
                        CairoMakie.scatter!(ax1, for_sol[Block(ii)][5], for_sol[Block(ii)][6], color = :blue)
                        CairoMakie.scatter!(ax1, reconstructed_sol[Block(ii)][1], reconstructed_sol[Block(ii)][2], color = :purple)
                        CairoMakie.scatter!(ax1, reconstructed_sol[Block(ii)][5], reconstructed_sol[Block(ii)][6], color = :magenta)
                        # # GLMakie.scatter!(ax, x[9],x[10], color = :purple)
                        # GLMakie.scatter!(ax, x[13],x[14], color = :magenta)
                    end

                    CairoMakie.save("SolutionPlot"* string(i) *".png", fig1)
                    break
                catch e
                    rethrow(e)
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

    open("experiments.tmp2.txt", "w+") do file
        write(file, "\nground truth:\n")
        write(file, string(hidden_params))

        write(file, "\nour method's errors: \n")
        write(file, string(errors))

        write(file, "\nbaseline errors: \n")
        write(file, string(baseline_errors))

        write(file, "\nrecovered parameters:\n")
        write(file, string(recovered_params))

        write(file, "\nbaseline recovered parameters:\n")
        write(file, string(baseline_recovered_params))

        write(file, "\nparameter errors:\n")
        write(file, string(parameter_error))

        write(file, "\nbaseline parameter errors:\n")
        write(file, string(baseline_parameter_error))

        write(file, "\nparameter cosine errors:\n")
        write(file, string(parameter_cosine_error))

        write(file, "\nbaseline parameter cosine errors:\n")
        write(file, string(baseline_parameter_cosine_error))

        write(file, "\nobserved trajectories:\n")
        write(file, string(observed_trajectories))

        write(file, "\nrecovered trajectories\n")
        write(file, string(recovered_traj))

        write(file, "\nbaseline recovered trajectories\n")
        write(file, string(baseline_recovered_traj))
    end

    graph(
    errors,
    baseline_errors, parameter_error,
    baseline_parameter_error,
    parameter_cosine_error,
    baseline_parameter_cosine_error,
    σs)
end

function expand_baseline_params(baseline_params)
    mortar([vcat(baseline_params[Block(1)], [1.0]), vcat(baseline_params[Block(2)], [1.0])])
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

function graph(
    errors,
    baseline_errors, parameter_error,
    baseline_parameter_error,
    parameter_cosine_error,
    baseline_parameter_cosine_error,
    σs)
    fig1 = CairoMakie.Figure()
    ax1 = CairoMakie.Axis(fig1[1, 1])

    mean_errors = [Statistics.mean(errors[i, :]) for i in 1:size(errors, 1)]
    stds = [Statistics.std(errors[i, :]) for i in 1:size(errors, 1)]

    baseline_mean_errors = [Statistics.mean(baseline_errors[i, :]) for i in 1:size(baseline_errors, 1)]
    baseline_stds = [Statistics.std(baseline_errors[i, :]) for i in 1:size(baseline_errors, 1)]

    our_method = CairoMakie.scatter!(ax1, σs, mean_errors, color = (:blue, 0.75))
    CairoMakie.errorbars!(ax1, σs, mean_errors, stds, color = (:blue, 0.75))

    baseline = CairoMakie.scatter!(ax1, σs, baseline_mean_errors, color = (:red, 0.75))
    CairoMakie.errorbars!(ax1, σs, baseline_mean_errors, baseline_stds, color = (:red, 0.75))

    CairoMakie.axislegend(ax1, [our_method, baseline], ["Our Method", "Baseline"], position = :lt)
    CairoMakie.save("NoiseGraph.png", fig1)


    fig2 = CairoMakie.Figure()
    ax2 = CairoMakie.Axis(fig2[1, 1])

    mean_parameter_errors = [Statistics.mean(parameter_error[i, :]) for i in 1:size(parameter_error, 1)]
    parameter_stds = [Statistics.std(parameter_error[i, :]) for i in 1:size(parameter_error, 1)]

    baseline_mean_parameter_errors = [Statistics.mean(baseline_parameter_error[i, :]) for i in 1:size(baseline_parameter_error, 1)]
    baseline_parameter_stds = [Statistics.std(baseline_parameter_error[i, :]) for i in 1:size(baseline_parameter_error, 1)]

    our_method = CairoMakie.scatter!(ax2, σs, mean_parameter_errors, color = (:blue, 0.75))
    CairoMakie.errorbars!(ax2, σs, mean_parameter_errors, parameter_stds, color = (:blue, 0.75))

    baseline = CairoMakie.scatter!(ax2, σs, baseline_mean_parameter_errors, color = (:red, 0.75))
    CairoMakie.errorbars!(ax2, σs, baseline_mean_parameter_errors, baseline_parameter_stds, color = (:red, 0.75))

    CairoMakie.axislegend(ax2, [our_method, baseline], ["Our Method", "Baseline"], position = :lt)
    CairoMakie.save("ParameterErrorGraph.png", fig2)


    fig3 = CairoMakie.Figure()
    ax3 = CairoMakie.Axis(fig3[1, 1])

    mean_parameter_cosine_errors = [Statistics.mean(parameter_cosine_error[i, :]) for i in 1:size(parameter_cosine_error, 1)]
    parameter_cosine_stds = [Statistics.std(parameter_cosine_error[i, :]) for i in 1:size(parameter_cosine_error, 1)]

    baseline_mean_parameter_cosine_errors = [Statistics.mean(baseline_parameter_cosine_error[i, :]) for i in 1:size(baseline_parameter_cosine_error, 1)]
    baseline_parameter_cosine_stds = [Statistics.std(baseline_parameter_cosine_error[i, :]) for i in 1:size(baseline_parameter_cosine_error, 1)]

    our_method = CairoMakie.scatter!(ax3, σs, mean_parameter_cosine_errors, color = (:blue, 0.75))
    CairoMakie.errorbars!(ax3, σs, mean_parameter_cosine_errors, parameter_cosine_stds, color = (:blue, 0.75))

    baseline = CairoMakie.scatter!(ax3, σs, baseline_mean_parameter_cosine_errors, color = (:red, 0.75))
    CairoMakie.errorbars!(ax3, σs, baseline_mean_parameter_cosine_errors, baseline_parameter_cosine_stds, color = (:red, 0.75))

    CairoMakie.axislegend(ax3, [our_method, baseline], ["Our Method", "Baseline"], position = :lt)
    CairoMakie.save("ParameterCosineErrorGraph.png", fig3)
end

end
