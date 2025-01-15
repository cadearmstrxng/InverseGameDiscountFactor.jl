module InD

using BlockArrays: blocksizes

include("../GameUtils.jl")
include("../graphing/ExperimentGraphingUtils.jl")
include("../../src/InverseGameDiscountFactor.jl")

export run_bicycle_sim
function run_bicycle_sim(full_state=true, graph=true)
    init = GameUtils.init_bicycle_test_game(
        full_state;
        myopic=true
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
        ExperimentGraphicUtils.graph_trajectories(
            "Our Method",
            [forward_solution, method_sol.recovered_trajectory],
            init.game_structure,
            init.horizon
        )
    end
end

end