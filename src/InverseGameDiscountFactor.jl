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
    goal = mortar([[0.0, -2.7, 1], [2.7, 1, 1]]),
    plot_please = false
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
    horizon = 75
    turn_length = 2
    solver = MCPCoupledOptimizationSolver(game.game, horizon, blocksizes(goal, 1))
    mcp_game = solver.mcp_game
    # sim_steps = let
    #     n_sim_steps = 150
    #     progress = ProgressMeter.Progress(n_sim_steps)
    #     receding_horizon_strategy =
    #         WarmStartRecedingHorizonStrategy(; solver, game.game, turn_length, context_state = goal)
    #     rollout(
    #         game.game.dynamics,
    #         receding_horizon_strategy,
    #         initial_state,
    #         n_sim_steps;
    #         get_info = (γ, x, t) ->
    #             (ProgressMeter.next!(progress); γ.receding_horizon_strategy),
    #     )
    # end
    # animate_sim_steps(game.game, sim_steps; live = false, framerate = 20, show_turn = true)
    # (; sim_steps, game.game)

    strat = WarmStartRecedingHorizonStrategy(; solver, game.game, turn_length, context_state = goal)

    # forward_solution = rollout(game.game.dynamics, strat, initial_state, 150)
    forward_solution = solve_mcp_game(mcp_game, initial_state, goal; verbose = true)
    
    # @infiltrate

    if plot_please
        player_state_dimension = convert(Int64, state_dim(game.game.dynamics)/2)
        player_control_dimension = convert(Int64, control_dim(game.game.dynamics)/2)
        
        # forward_solution = [forward_solution.primals[1][1:state_dimension*horizon], forward_solution.primals[2][1:state_dimension*horizon],
        #                     forward_solution.primals[1][state_dimension*horizon+1:end], forward_solution.primals[2][state_dimension*horizon+1:end]]

        # @infiltrate
            
        player1state = forward_solution.primals[1][1:player_state_dimension*horizon]
        player2state = forward_solution.primals[2][1:player_state_dimension*horizon]
        player1control = forward_solution.primals[1][player_state_dimension*horizon+1:end]
        player2control = forward_solution.primals[2][player_state_dimension*horizon+1:end]

        solution = []

        for i in 1:horizon
            push!(solution, player1state[(i-1) * player_state_dimension + 1: i * player_state_dimension])
            push!(solution, player2state[(i-1) * player_state_dimension + 1: i * player_state_dimension])
            push!(solution, player1control[(i-1) * player_control_dimension + 1: i * player_control_dimension])
            push!(solution, player2control[(i-1) * player_control_dimension + 1: i * player_control_dimension])
        end
        
        # @infiltrate

        solution = vcat(solution...)

        solution = BlockVector(solution, [2*player_state_dimension + 2*player_control_dimension for i in 1:horizon])

        # @infiltrate

        fig1 = GLMakie.Figure()
        ax = GLMakie.Axis(fig1[1, 1])

        for ii in 1:horizon
            GLMakie.scatter!(ax, solution[Block(ii)][1],solution[Block(ii)][2], color = :red)
            GLMakie.scatter!(ax, solution[Block(ii)][5],solution[Block(ii)][6], color = :blue)
            # GLMakie.scatter!(ax, x[9],x[10], color = :purple)
            # GLMakie.scatter!(ax, x[13],x[14], color = :magenta)
        end

        GLMakie.display(fig1)
    end

    
    initial_state_guess, context_state_guess = sample_initial_states_and_context(game, horizon, Random.MersenneTwister(1), 0.08)

    println("Initial State Guess: ", initial_state_guess)
    println("Context State Guess: ", context_state_guess)

    context_state_estimation, last_solution, i_, solving_info, time_exec = solve_inverse_mcp_game(mcp_game,initial_state,forward_solution.variables,context_state_guess)

    println("Context State Estimation: ", context_state_estimation)
    # @infiltrate
    println("Solution Error: ", norm_sqr(restruct_solution(forward_solution, game, horizon) - restruct_solution(last_solution, game, horizon)))

    # inv_game = InverseMCPProblem(game,horizon,blocksizes(goal, 1))

    # @infiltrate

    # inverse_solution = solve_inverse_mcp_game(inv_game, game.game, forward_solution.xs, initial_state; horizon = horizon, dim_params = sum(blocksizes(goal, 1)))


    # @infiltrate
    # # fig2 = GLMakie.Figure()
    # # ax = GLMakie.Axis(fig1[1, 1])

    # # for x in inverse_solution.xs
    # #     GLMakie.scatter!(ax, x[1],x[2], color = :red)
    # #     GLMakie.scatter!(ax, x[5],x[6], color = :blue)
    # #     GLMakie.scatter!(ax, x[9],x[10], color = :purple)
    # #     GLMakie.scatter!(ax, x[13],x[14], color = :magenta)
    # # end

    # # fig2
end

function restruct_solution(solution, game, horizon)
    player_state_dimension = convert(Int64, state_dim(game.game.dynamics)/2)
    player_control_dimension = convert(Int64, control_dim(game.game.dynamics)/2)
        
    # forward_solution = [forward_solution.primals[1][1:state_dimension*horizon], forward_solution.primals[2][1:state_dimension*horizon],
    #                     forward_solution.primals[1][state_dimension*horizon+1:end], forward_solution.primals[2][state_dimension*horizon+1:end]]

    # @infiltrate

    if typeof(solution) == NamedTuple{(:primals, :variables, :status), Tuple{Vector{Vector{ForwardDiff.Dual{Nothing, Float64, 14}}}, Vector{Float64}, PATHSolver.MCP_Termination}}
        primals = solution.primals

        solution = []

        states = []
        actions = []

        for primal in primals
            for i in eachindex(primal)
                primal[i] = primal[i].value
            end

            push!(states, primal[1:player_state_dimension*horizon])
            push!(actions, primal[player_state_dimension*horizon+1:end])
        end

        solution = vcat(states...)
        solution = vcat(solution, vcat(actions...))
        @infiltrate

        vars = []

        for i in solution
            push!(vars, i.value)
        end

        return vars
    end
        
    player1state = solution.primals[1][1:player_state_dimension*horizon]
    player2state = solution.primals[2][1:player_state_dimension*horizon]
    player1control = solution.primals[1][player_state_dimension*horizon+1:end]
    player2control = solution.primals[2][player_state_dimension*horizon+1:end]

    solution = []

    for i in 1:horizon
        push!(solution, player1state[(i-1) * player_state_dimension + 1: i * player_state_dimension])
        push!(solution, player2state[(i-1) * player_state_dimension + 1: i * player_state_dimension])
        push!(solution, player1control[(i-1) * player_control_dimension + 1: i * player_control_dimension])
        push!(solution, player2control[(i-1) * player_control_dimension + 1: i * player_control_dimension])
    end
    
    # @infiltrate

    solution = vcat(solution...)

    solution = BlockVector(solution, [2*player_state_dimension + 2*player_control_dimension for i in 1:horizon])

    solution

end

end # module InverseGameDiscountFactor
