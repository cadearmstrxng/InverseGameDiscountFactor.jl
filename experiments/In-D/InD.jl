module InD

using BlockArrays: blocksizes, Block, mortar, BlockVector
using Infiltrator
using CairoMakie
using TrajectoryGamesBase:
    num_players, state_dim
using LinearAlgebra: norm_sqr

using ImageTransformations
using Rotations
using OffsetArrays:Origin
using TrajectoryGamesExamples: BicycleDynamics, PolygonEnvironment

include("../GameUtils.jl")
include("../graphing/ExperimentGraphingUtils.jl")
include("../../src/InverseGameDiscountFactor.jl")
include("Environment.jl")

export run_bicycle_sim
function run_bicycle_sim(;full_state=true, graph=true, verbose = true)

    # InD_observations = GameUtils.observe_trajectory(forward_solution, init)
    # frames = [620, 1001]
    # frames = [646, 1103]
    # frames = [780, 806]
    tracks = [19, 20]
    downsample_rate = 16
    InD_observations, InD_player_time_intervals = GameUtils.pull_trajectory("07";
        track = tracks, downsample_rate = downsample_rate)
    open("InD_observations.tmp.txt", "w") do f
        for i in eachindex(InD_observations)
            write(f, string(round.(InD_observations[i]; digits = 4)), "\n")
        end
        write(f, "observation times: ", string(InD_player_time_intervals))
    end
    # 780 -> 916 = 136
    # Take 1:
    # 17: 530- 806
    # 19: 620-1001
    # 22: 780- 916
    # 780 -> 806 = 26

    # potentially use receding horizon - https://arxiv.org/pdf/2302.01999
    # downsample the trajectory by 5

    # @infiltrate
    dynamics = BicycleDynamics(;
        dt = 0.04*downsample_rate, # needs to become framerate
        l = 2.0,
        state_bounds = (; 
            lb = [-35.0, 20.0, 0.0, -2*pi],  # [x, y, velocity, heading]
            ub = [10.0, 100.0, 15.0, 2*pi]
        ),
        control_bounds = (; lb = [-5, -pi/4], ub = [5, pi/4]),
        integration_scheme = :forward_euler
    )
    # @infiltrate
    init = GameUtils.init_bicycle_test_game(
        full_state;
        # initial_state = mortar( 
        #     [InD_observations[InD_player_time_intervals[i][1]][Block(i)][1:end]
        #     for i in 1:length(tracks)]),
        initial_state = copy(InD_observations[1]),
        game_params = mortar([
            [[InD_observations[InD_player_time_intervals[i][end]][Block(i)][1:2]..., 1.0, 1.0] for i in 1:length(tracks)]...]),
        game_environment = PolygonEnvironment(4, 200),
        horizon = max(last.(InD_player_time_intervals)...),
        n = length(tracks),
        dt = 0.04*downsample_rate,
        myopic=true,
        verbose = verbose,
        dynamics = dynamics,
        observation_times = InD_player_time_intervals
    )
    !verbose || println("initial state: ", init.initial_state)
    !verbose || println("initial game parameters: ", init.game_parameters)
    !verbose || println("min horizon: ", min(first.(InD_player_time_intervals)...))
    !verbose || println("max horizon: ", max(last.(InD_player_time_intervals)...))
    !verbose || println("initial horizon: ", init.horizon)
    !verbose || println("observation model: ", init.observation_model)
    !verbose || println("observation dim: ", init.observation_dim)
    
    InD_observations = (full_state) ? InD_observations : [BlockVector(GameUtils.observe_trajectory(InD_observations[t], init;blocked_by_time = false),
        [init.observation_dim for _ in 1:length(tracks)]) for t in 1:init.horizon]
    
    !verbose || println("game initialized\ninitializing mcp coupled optimization solver")
    mcp_solver = InverseGameDiscountFactor.MCPCoupledOptimizationSolver(
        init.game_structure.game,
        init.horizon,
        blocksizes(init.game_parameters, 1),
        InD_player_time_intervals
    )
    !verbose || println("mcp coupled optimization solver initialized")

    # !verbose || println("solving forward game")
    # fs_temp = InverseGameDiscountFactor.solve_mcp_game(
    #     mcp_solver.mcp_game,
    #     init.initial_state,
    #     init.game_parameters;
    #     verbose = false
    #     )
    # forward_solution = InverseGameDiscountFactor.reconstruct_solution(
    #     fs_temp,
    #     init.game_structure.game,
    #     init.horizon
    # )
    # xs = [BlockVector(forward_solution[Block(t)], [4 for _ in 1:length(tracks)]) for t in 1:init.horizon]
    # xs = vcat([init.initial_state], xs)
    # us = [fs_temp.primals[i][4*init.horizon+1:end] for i in 1:length(tracks)]
    # us = [vcat([us[i][2*t-1:2*t] for i in 1:length(tracks)]...) for t in 1:init.horizon]
    # us = [BlockVector(us[t], [2 for _ in 1:length(tracks)]) for t in 1:init.horizon]

    # cost_val = mcp_solver.mcp_game.game.cost(xs, us, init.game_parameters)
    # !verbose || println("forward game solved, cost: ", cost_val)
    # # @infiltrate

    # # return
    # !verbose||println("forward game solved, status: ", fs_temp.status)
    # forward_game_observations = GameUtils.observe_trajectory(forward_solution, init)

    # Add graph comparing forward solution to observations
    # if graph
    #     ExperimentGraphingUtils.graph_trajectories(
    #         "Observed v. Forward Solution",
    #         [InD_observations, BlockVector(vcat(forward_game_observations...), [length(tracks)*4 for _ in 1:init.horizon])],
    #         init.game_structure,
    #         init.horizon;
    #         colors = [
    #             [(:red, 1.0), (:blue, 1.0), (:green, 1.0)],
    #             [(:red, 0.5), (:blue, 0.5), (:green, 0.5)]
    #         ],
    #         constraints = get_constraints(init.environment)
    #     )
    # end
    # return

    !verbose || println("solving inverse game")
    method_sol = InverseGameDiscountFactor.solve_myopic_inverse_game(
        mcp_solver.mcp_game,
        InD_observations,
        # forward_game_observations,
        init.observation_model,
        Tuple(blocksizes(init.game_parameters, 1)),;
        initial_state = init.initial_state,
        hidden_state_guess = init.game_parameters,
        max_grad_steps = 200,
        retries_on_divergence = 3,
        verbose = verbose,
        dynamics = dynamics,
    )
    !verbose || println("finished inverse game")
    !verbose || println("recovered pararms: ", method_sol.recovered_params)

    if graph
        ExperimentGraphingUtils.graph_trajectories(
            "Observed v. Recovered v. Warm Start",
            [InD_observations, method_sol.recovered_trajectory, method_sol.warm_start_trajectory],
            init.game_structure,
            init.horizon,
            InD_player_time_intervals;
            colors = [
                [(:red, 1.0), (:blue, 1.0), (:green, 1.0)],
                [(:red, 0.5), (:blue, 0.5), (:green, 0.5)],
                [(:red, 0.2 ), (:blue, 0.2), (:green, 0.2)]
            ],
            constraints = get_constraints(init.environment)
        )
        ExperimentGraphingUtils.graph_trajectories(
            "Observed v. Recovered",
            [InD_observations, method_sol.recovered_trajectory],
            init.game_structure,
            init.horizon,
            InD_player_time_intervals;
            colors = [
                [(:red, 1.0), (:blue, 1.0), (:green, 1.0)],
                [(:red, 0.5), (:blue, 0.5), (:green, 0.5)],
                [(:red, 0.2 ), (:blue, 0.2), (:green, 0.2)]
            ],
            constraints = get_constraints(init.environment)
        )
        ExperimentGraphingUtils.graph_trajectories(
            "Observed v. Warm Start",
            [InD_observations, method_sol.warm_start_trajectory],
            init.game_structure,
            init.horizon,
            InD_player_time_intervals;
            colors = [
                [(:red, 1.0), (:blue, 1.0), (:green, 1.0)],
                [(:red, 0.5), (:blue, 0.5), (:green, 0.5)]
            ],
            constraints = get_constraints(init.environment)
        )
        ExperimentGraphingUtils.graph_trajectories(
            "Recovered v. Warm Start",
            [InD_observations, method_sol.recovered_trajectory, method_sol.warm_start_trajectory],
            init.game_structure,
            init.horizon,
            InD_player_time_intervals;
            colors = [
                [(:red, 0.0), (:blue, 0.0), (:green, 0.0)],
                [(:red, 1.0), (:blue, 1.0), (:green, 1.0)], 
                [(:red, 0.5), (:blue, 0.5), (:green, 0.5)]
            ],
            constraints = get_constraints(init.environment)
        )
    end

    sol_error = method_sol.sol_error
    !verbose || println("inverse sol error: ", sol_error)
    # warm_sol_error = norm_sqr(method_sol.warm_start_trajectory - vcat(InD_observations...))
    !verbose || println("warm start sol error: ", method_sol.sol_error)
    # !verbose || println("% improvement on warm start: ", (warm_sol_error - sol_error) / warm_sol_error * 100)
end

end