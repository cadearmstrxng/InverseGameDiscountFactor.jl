module InD

using BlockArrays: blocksizes, Block, mortar
using Infiltrator
using CairoMakie
using TrajectoryGamesBase:
    num_players, state_dim
using LinearAlgebra: norm_sqr

include("../GameUtils.jl")
include("../graphing/ExperimentGraphingUtils.jl")
include("../../src/InverseGameDiscountFactor.jl")

export run_bicycle_sim
function run_bicycle_sim(full_state=true, graph=true)

    # observed_forward_solution = GameUtils.observe_trajectory(forward_solution, init)
    frames = [780, 916]
    downsample_rate = 5
    observed_forward_solution = GameUtils.pull_trajectory("07";
        track = [20, 19, 22], downsample_rate = downsample_rate, all = false, frames = frames)
    # TODO need to time-synch each trajectory
    # 20: 646 - 1103
    # 19: 620 - 1001
    # 22: 780 - 916
    # 780 -> 916 = 136

    # potentially use receding horizon - https://arxiv.org/pdf/2302.01999
    # downsample the trajectory by 5

    # @infiltrate

    init = GameUtils.init_bicycle_test_game(
        full_state;
        initial_state = observed_forward_solution[1],
        game_params = mortar([
            [observed_forward_solution[end][Block(1)][1:2]..., 0.75, [1.0 for _ in 4:9]...],
            [observed_forward_solution[end][Block(2)][1:2]..., 0.75, [1.0 for _ in 4:9]...],
            [observed_forward_solution[end][Block(3)][1:2]..., 0.75, [1.0 for _ in 4:9]...]]),
        horizon = length(frames[1]:downsample_rate:frames[2]),
        dt = 0.04*downsample_rate,
        myopic=true
    )
    
    mcp_game = InverseGameDiscountFactor.MCPCoupledOptimizationSolver(
        init.game_structure.game,
        init.horizon,
        blocksizes(init.game_parameters, 1)
    ).mcp_game

    # forward_solution = InverseGameDiscountFactor.reconstruct_solution(
    #     InverseGameDiscountFactor.solve_mcp_game(
    #         mcp_game,
    #         init.initial_state,
    #         init.game_parameters;
    #         verbose = false
    #         ),
    #     init.game_structure.game,
    #     init.horizon
    # )

    method_sol = InverseGameDiscountFactor.solve_myopic_inverse_game(
        mcp_game,
        observed_forward_solution,
        init.observation_model,
        (9, 9, 9);
        hidden_state_guess = init.game_parameters,
        max_grad_steps = 200,
        retries_on_divergence = 3,
        verbose = false
    )

    if graph
        ExperimentGraphicUtils.graph_trajectories(
            "Our Method",
            [observed_forward_solution, method_sol.recovered_trajectory],
            init.game_structure,
            init.horizon;
            colors = [
                [(:red, 1.0), (:blue, 1.0), (:green, 1.0)],
                [(:red, 0.5), (:blue, 0.5), (:green, 0.5)]
            ]
        )
    end

    sol_error = norm_sqr(method_sol.recovered_trajectory - vcat(observed_forward_solution...))

    println("inverse sol error: ", sol_error)

    sol_error = norm_sqr(InverseGameDiscountFactor.reconstruct_solution(method_sol.warm_start_trajectory, mcp_game.game, init.horizon) - vcat(observed_forward_solution...))

    println("warm start sol error: ", sol_error)


end

end