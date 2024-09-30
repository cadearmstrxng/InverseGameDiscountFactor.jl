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
using GLMakie: GLMakie
using Symbolics: Symbolics, @variables, scalarize
using ChainRulesCore: ChainRulesCore
using ForwardDiff: ForwardDiff
using Zygote: Zygote
using Flux: Flux, glorot_uniform, Dense, Optimise.Adam, NNlib.relu, Chain
using ParametricMCPs: ParametricMCPs
using Makie: Makie
using Colors: @colorant_str
using JLD2: JLD2
using Statistics: std

using Infiltrator

include("utils/ExampleProblems.jl")
using .ExampleProblems: n_player_collision_avoidance, two_player_guidance_game, CollisionAvoidanceGame, HighwayGame


include("utils/utils.jl")
include("problem_formulation.jl")
include("solve.jl")
include("baseline/inverse_MCP_solver.jl")


function main(; 
    # initial_state = mortar([
    #     [-1, 2.5, 0.1, -0.2],
    #     [1, 2.8, 0.0, 0.0],
    #     [-2.8, 1, 0.2, 0.1],
    #     [-2.8, -1, -0.27, 0.1],
    # ]),
    # goal = mortar([[0.0, -2.7, 1], [2, -2.8, 1], [2.7, 1, 1], [2.7, -1.1, 1]])

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

    observation_model = (; σ = 0.1, expected_observation = x -> x .+ observation_model.σ * randn(length(x)))

    for_sol = reconstruct_solution(forward_solution, game.game, horizon)

    for_sol = observation_model.expected_observation(for_sol)

    # @infiltrate

    context_state_estimation, last_solution, i_, solving_info, time_exec = solve_inverse_mcp_game(mcp_game, initial_state, for_sol, context_state_guess, horizon; 
                                                                            max_grad_steps = 500)

    println("Context State Estimation: ", context_state_estimation)
    sol_error = norm_sqr(reconstruct_solution(forward_solution, game.game, horizon) - reconstruct_solution(last_solution, game.game, horizon))
    # @infiltrate
    println("Solution Error: ", sol_error)


    if plot_please
        for_solution = reconstruct_solution(forward_solution, game.game, horizon)
        inv_solution = reconstruct_solution(last_solution, game.game, horizon)
        # @infiltrate

        fig1 = GLMakie.Figure()
        ax1 = GLMakie.Axis(fig1[1, 1])

        for ii in 1:horizon
            GLMakie.scatter!(ax1, for_solution[Block(ii)][1], for_solution[Block(ii)][2], color = :red)
            GLMakie.scatter!(ax1, for_solution[Block(ii)][5], for_solution[Block(ii)][6], color = :blue)
            GLMakie.scatter!(ax1, inv_solution[Block(ii)][1], inv_solution[Block(ii)][2], color = :purple)
            GLMakie.scatter!(ax1, inv_solution[Block(ii)][5], inv_solution[Block(ii)][6], color = :magenta)
            # # GLMakie.scatter!(ax, x[9],x[10], color = :purple)
            # GLMakie.scatter!(ax, x[13],x[14], color = :magenta)
        end

        GLMakie.save("SolutionPlot.png", fig1)
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

    σs = [0.01*i for i in 0:10]

    errors = []

    for σ in σs

        observation_model = (; σ = σ, expected_observation = x -> x .+ observation_model.σ * randn(rng, length(x)))

        error = []

        for i in 0:10
            for_sol = reconstruct_solution(forward_solution, game.game, horizon)
            for_sol = observation_model.expected_observation(for_sol)

            initial_state_guess, context_state_guess = sample_initial_states_and_context(game, horizon, rng, 0.08)

            context_state_guess[1] = 0
            context_state_guess[2] = -2.7

            context_state_guess[4] = 2.7
            context_state_guess[5] = 1

            context_state_estimation, last_solution, i_, solving_info, time_exec = solve_inverse_mcp_game(mcp_game, initial_state, for_sol, context_state_guess, horizon; 
                                                                            max_grad_steps = 150)

            sol_error = norm_sqr(reconstruct_solution(forward_solution, game.game, horizon) - reconstruct_solution(last_solution, game.game, horizon))
        
            push!(error, sol_error)
        end
        push!(errors, error)
    end

    fig1 = GLMakie.Figure()
    ax1 = GLMakie.Axis(fig1[1, 1])

    for i in 1:11
        GLMakie.scatter!(ax1, σs[i], mean(errors[i]), color = :red)

        stdev = Statistics.std(errors[i])

        errbar = stdev/sqrt(11)

        GLMakie.errorbar!(ax1, σs[i], mean(errors[i]), yerrors = errbar, color = :red)
    end

    GLMakie.save("NoiseGraph.png", fig1)

end


end
