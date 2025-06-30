module Crosswalk
using LinearAlgebra: norm
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
        ExperimentGraphingUtils.graph_crosswalk_trajectories(
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

function run_bicycle_crosswalk_sim(full_state = true, graph = true, verbose = true)
    # recovers: [-20.603578775451442, 60.97433529761103, 0.9731234554485012, -7.242849841954976, 93.02474410616205, 0.899999999999994]
    mode = :forward_euler
    dr = 7
    # observations = GameUtils.pull_trajectory("07";
    #         track = [20,22], downsample_rate = dr, all = false, frames = [780, 916])
    init = GameUtils.init_crosswalk_game(
        full_state;
        myopic = true,
        initial_state = mortar([
            # [-20.613224922805635, 60.9594987927257, 0.012641427925673587, 3.1467872410871696],
            [-20, 10, 3.0, 0],
            [20, 6, -7, 1.57]]),
            # [-14.100017007522755, 26.38296417815802, 11.62063796039615, 4.591809731462673]]),
        dynamics = BicycleDynamics(;
            dt = 0.04*dr,
            state_bounds = (; lb = [-Inf, -Inf, -Inf, -Inf], ub = [Inf, Inf, Inf, Inf]),
            control_bounds = (; lb = [-Inf, -pi/2], ub = [Inf, pi/2]),
            integration_scheme = mode
        ),
        horizon = length(780:dr:916),
        game_params = mortar([
            # [-20.60372487399995, 60.97425094757472, 0.95],
            [7, 2, 0.95],
            # [-20.60372487399995, 60.97425094757472, 0.95]
            [-5, 15, 0.95]
        ])
    )
    !verbose || println("initial state: ", init.initial_state)
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
        # observations,
        observed_forward_solution,
        init.observation_model,
        (3, 3);
        initial_state = init.initial_state,
        hidden_state_guess = init.game_parameters,
        max_grad_steps = 200,
        retries_on_divergence = 3,
        verbose = false,
        dynamics = BicycleDynamics(;
            dt = 0.04*dr, # needs to become framerate
            l = 1.0,
            state_bounds = (; lb = [-Inf, -Inf, -Inf, -Inf], ub = [Inf, Inf, Inf, Inf]),
            control_bounds = (; lb = [-Inf, -pi/2], ub = [Inf, pi/2]),
            integration_scheme = mode
        )
    )
    !verbose || println("recovered params: ", method_sol.recovered_params)
    !verbose || println("param error: ", norm(method_sol.recovered_params - init.game_parameters))

    if graph
        # ExperimentGraphicUtils.graph_crosswalk_trajectories(
        #     "Our Method",
        #     [forward_solution, method_sol.recovered_trajectory],
        #     init.game_structure,
        #     init.horizon
        # )
        # observed_forward_solution = BlockVector(vcat(observations...), [init.state_dim[1]*2 for _ in eachindex(observations)])
        observed_forward_solution = BlockVector(vcat(observed_forward_solution...), [init.state_dim[1]*2 for _ in eachindex(observed_forward_solution)])

        
        ExperimentGraphicUtils.graph_crosswalk_trajectories(
            "Observations",
            [observed_forward_solution, method_sol.recovered_trajectory],
            init.game_structure,
            init.horizon;
            colors = [[(:red, 1.0), (:blue, 1.0)], [(:orange, 0.25), (:purple, 0.25)]]
        )
    end
end


end