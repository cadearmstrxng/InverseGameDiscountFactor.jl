module Crosswalk

using BlockArrays: blocksizes, mortar, BlockVector
using TrajectoryGamesExamples: BicycleDynamics
include("../GameUtils.jl")
include("../graphing/ExperimentGraphingUtils.jl")
include("../../src/InverseGameDiscountFactor.jl")

export run_myopic_crosswalk_sim

function run_myopic_crosswalk_sim(full_state = true, graph = true, verbose = true)
    init = GameUtils.init_crosswalk_game(
        full_state;
        myopic = true
    )
    
    mcp_game = InverseGameDiscountFactor.MCPCoupledOptimizationSolver(
        init.game_structure.game,
        init.horizon,
        blocksizes(init.game_parameters, 1)
    ).mcp_game

    forward_solution = InverseGameDiscountFactor.reconstruct_solution(
        InverseGameDiscountFactor.solve_mcp_game(
            mcp_game,
            init.initial_state,
            init.game_parameters;
            verbose = false
            ),
        init.game_structure.game,
        init.horizon
    )

    observed_forward_solution = GameUtils.observe_trajectory(forward_solution, init)

    method_sol = InverseGameDiscountFactor.solve_myopic_inverse_game(
        mcp_game,
        observed_forward_solution,
        init.observation_model,
        (3, 3);
        hidden_state_guess = init.game_parameters,
        max_grad_steps = 200,
        retries_on_divergence = 3,
        verbose = false
    )

    if graph
        ExperimentGraphicUtils.graph_crosswalk_trajectories(
            "Our Method",
            [forward_solution, method_sol.recovered_trajectory],
            init.game_structure,
            init.horizon
        )
        # observations = GameUtils.pull_trajectory("07";
        #     track = [20,22], downsample_rate = 10, all = false, frames = [780, 916])
        # observations = BlockVector(vcat(observations...), [init.state_dim[1]*2 for _ in eachindex(observations)])
        # ExperimentGraphicUtils.graph_crosswalk_trajectories(
        #     "Observations",
        #     [forward_solution, observations],
        #     init.game_structure,
        #     init.horizon;
        #     colors = [[(:red, 1.0), (:blue, 1.0)], [(:orange, 0.25), (:purple, 0.25)]]
        # )
    end
end

end