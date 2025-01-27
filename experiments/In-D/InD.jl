module InD

using BlockArrays: blocksizes, Block, mortar
using Infiltrator
using CairoMakie
using TrajectoryGamesBase:
    num_players, state_dim

include("../GameUtils.jl")
include("../graphing/ExperimentGraphingUtils.jl")
include("../../src/InverseGameDiscountFactor.jl")

export run_bicycle_sim
function run_bicycle_sim(full_state=true, graph=true)

    # observed_forward_solution = GameUtils.observe_trajectory(forward_solution, init)
    frames = [780, 806]
    observed_forward_solution = GameUtils.pull_trajectory("07";
        track = [17, 19, 22], all = false, frames = frames)
    # TODO need to time-synch each trajectory
    # 17: 530- 806
    # 19: 620-1001
    # 22: 780- 916
    # 780 -> 806 = 26

    # @infiltrate

    init = GameUtils.init_bicycle_test_game(
        full_state;
        initial_state = observed_forward_solution[1],
        game_params = mortar([
            [observed_forward_solution[end][Block(1)][1:2]..., 0.6],
            [observed_forward_solution[end][Block(2)][1:2]..., 0.6],
            [observed_forward_solution[end][Block(3)][1:2]..., 0.6]]),
        horizon = frames[2] - frames[1] + 1,
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
        (3, 3, 3);
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
end

end