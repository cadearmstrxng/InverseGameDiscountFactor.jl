module baseline_crosswalk_sim
#TODO mirror in Crosswalk.jl
using TrajectoryGamesExamples:
    PolygonEnvironment
using TrajectoryGamesBase:
    TrajectoryGame
using BlockArrays

include("../GameUtils.jl")
#TODO would be nice to just include one thing
include("../../src/solver/ProblemFormulation.jl")
include("../../src/solver/solve.jl")

export run_baseline_crosswalk_sim

function run_baseline_crosswalk_sim(full_state = true, graph = true)
    baseline_init = init_crosswalk_game(
        full_state;
        myopic = false
    )
    
    init = init_crosswalk_game(
        full_state
    )
    
    mcp_game = MCPCoupledOptimizationSolver(
        init.game_structure.game,
        init.horizon,
        blocksizes(init.game_parameters, 1)
    ).game

    forward_solution = solve_mcp_game(
        mcp_game,
        init.initial_state,
        init.game_parameters;
        verbose = false
    )

    if noisy
        observed_forward_solution = init.observation_model(forward_solution)
    else
        observed_forward_solution = init.observation_model(forward_solution, σ = 0.0)
    end

    method_sol = nothing #TODO fill in once set, probably need to include some file 

    if graph
        graph_trajectories(
            "Baseline",
            [forward_solution, method_sol],
            init.game_structure,
            init.horizon
        )
    end
end

end