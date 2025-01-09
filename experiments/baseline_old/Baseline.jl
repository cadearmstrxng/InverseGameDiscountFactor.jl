module baseline_crosswalk_sim
#TODO mirror in Crosswalk.jl
using TrajectoryGamesExamples:
    PolygonEnvironment
using BlockArrays

include("../GameUtils.jl")
#TODO would be nice to just include one thing
include("../../src/solver/ProblemFormulation.jl")
include("../../src/solver/solve.jl")

export run_baseline_crosswalk_sim, init_baseline_crosswalk_game

function run_baseline_crosswalk_sim(full_state = true, graph = true)
    init = init_crosswalk_game(full_state)
    
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

    observed_forward_solution = init.observation_model(forward_solution)

    # method_sol = #TODO fill in once set, probably need to include some file 

    if graph
    # TODO graph both forward sol w ground truth and method sol
    end
end

function init_baseline_crosswalk_game(
    full_state;
    state_dim = (4, 4),
    action_dim = (2, 2),
    σ = 0.0,
    game_environment = PolygonEnvironment(6, 8)
)
    if full_state # TODO
        observation_model = (;
            σ = σ,
            observation_model = 
                (x, σ = σ) -> 
                vcat(
                    [ x[state_dim[1] * (i - 1):state_dim[1]*i] .+ σ * randn(state_dim[1] * (i - 1):state_dim[1]*i)
                        for i in 1:num_players(init.game_structure) ]...
                ),
        )
    else
        observation_model = (;
            σ = σ,
            observation_model = 
                (x, σ = σ) -> 
                vcat(
                    [ x[state_dim[1]*(i-1)-(state_dim[1]-1):state_dim[1]*i-(state_dim[1]-2)] .+ σ * randn(state_dim[1] * (i - 1):state_dim[1]*i)
                        for i in 1:num_players(init.game_structure) ]...
                )
        )
    end

    (;
    initial_state = mortar([
            [0, 2, 0.1, -0.2], # initial x, y, initial velocity in x, y direction (player 1)
            [2.5, 2, 0.0, 0.0],# player 2
        ]),
    game_parameters = mortar([
            [2, 0, 0.6], # starting position x, y, discount factor (player 1)
            [0, 0, 0.6]  # player 2
        ]),
    environment = game_environment,
    horizon = 25,
    state_dim = state_dim,
    action_dim = action_dim,
    σ = σ,
    game_structure = n_player_collision_avoidance(
        2;
        game_environment,
        min_distance = 0.5,
        collision_avoidance_coefficient = 5.0,
        myopic = false # TODO don't love
    ), # TODO fill in args
    )
end

end