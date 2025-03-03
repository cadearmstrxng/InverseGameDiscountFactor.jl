module GameUtils

using BlockArrays: Block, blocksize, blocksizes, mortar, BlockVector, blocks
using TrajectoryGamesBase:
    GeneralSumCostStructure,
    ProductDynamics,
    TrajectoryGame,
    TrajectoryGameCost,
    PolygonEnvironment,
    num_players
using TrajectoryGamesExamples:planar_double_integrator, BicycleDynamics
using LinearAlgebra: norm_sqr, norm
using Statistics: mean
using Infiltrator

include("./In-D/Environment.jl")

export init_crosswalk_game, init_bicycle_test_game, observe_trajectory, pull_trajectory

struct CollisionAvoidanceGame
    game::TrajectoryGame
end

function my_norm_sqr(x)
    x'*x
end

function shared_collision_avoidance_coupling_constraints(num_players, min_distance)
    function coupling_constraint(xs, us)
        # mapreduce(vcat, 1:(num_players - 1)) do player_i
        #     mapreduce(vcat, (player_i + 1):num_players) do paired_player
        #         map(xs) do x
        #             my_norm_sqr(x[Block(player_i)][1:2] - x[Block(paired_player)][1:2]) -
        #             min_distance^2
        #         end
        #     end
        # end
        mapreduce(vcat, 1:(num_players - 1)) do player_i
            mapreduce(vcat, (player_i + 1):num_players) do paired_player
                map(xs) do x
                    if (player_i == 1 && paired_player == 2)
                        return 1000.0
                    else
                        my_norm_sqr(x[Block(player_i)][1:2] - x[Block(paired_player)][1:2]) -
                        min_distance^2
                    end
                end
            end
        end
    end
end

function n_player_collision_avoidance(
    num_players;
    environment,
    min_distance = 1.0, # context state 5
    collision_avoidance_coefficient = 20.0,
    dynamics = planar_double_integrator(;
        state_bounds = (; lb = [-Inf, -Inf, -0.8, -0.8], ub = [Inf, Inf, 0.8, 0.8]),
        control_bounds = (; lb = [-10, -10], ub = [10, 10]),
    ),
    myopic = true
)
    cost = let
        function target_cost(x, context_state, t)
            (myopic ? context_state[3] ^ t : 1) * norm_sqr(x[1:2] - context_state[1:2])
        end
        function control_cost(u, context_state, t)
            norm_sqr(u) * (myopic ? context_state[3] ^ t : 1)
        end
        function collision_cost(x, i, context_state, t)
            cost = [max(0.0, 1.0 + 0.2 * 1.0 - norm(x[Block(i)][1:2] - x[Block(paired_player)][1:2]))^2 for paired_player in [1:(i - 1); (i + 1):num_players]]
            sum(cost) 
        end
        function cost_for_player(i, xs, us, context_state, T)
            early_target = target_cost(xs[1][Block(i)], context_state[Block(i)], 1)
            mean_target = mean([target_cost(xs[t + 1][Block(i)], context_state[Block(i)], t) for t in 1:T])
            minimum_target = minimum([target_cost(xs[t][Block(i)], context_state[Block(i)],t) for t in 1:T])
            control = mean([control_cost(us[t][Block(i)], context_state[Block(i)], t) for t in 1:T])
            safe_distance_violation = mean([collision_cost(xs[t], i, context_state[Block(i)], t) for t in 1:T])

            #contex states 6-10

            # 0.2 * early_target + 
            1.0 * mean_target + 
            # 0.2 * minimum_target + 
            0.1 * control + 
            20 * safe_distance_violation
        end
        function cost_function(xs, us, context_state)
            num_players = blocksize(xs[1], 1)
            T = size(us,1)
            [cost_for_player(i, xs, us, context_state, T) for i in 1:num_players]
        end
        TrajectoryGameCost(cost_function, GeneralSumCostStructure())
    end
    dynamics = ProductDynamics([dynamics for _ in 1:num_players])
    game = TrajectoryGame(
        dynamics,
        cost,
        environment,
        shared_collision_avoidance_coupling_constraints(num_players, min_distance),
    )
    CollisionAvoidanceGame(game)
end

function InD_collision_avoidance(
    num_players,
    lane_centers;
    environment,
    min_distance = 1.0, # context state 5
    collision_avoidance_coefficient = 20.0,
    dynamics = planar_double_integrator(;
        state_bounds = (; lb = [-Inf, -Inf, -0.8, -0.8], ub = [Inf, Inf, 0.8, 0.8]),
        control_bounds = (; lb = [-10, -10], ub = [10, 10]),
    ),
    myopic = true)

    cost = let
        function target_cost(x, context_state, t)
            (myopic ? context_state[3] ^ t : 1) * norm_sqr(x[1:2] - context_state[1:2])
        end
        function control_cost(u, context_state, t)
            norm_sqr(u) * (myopic ? context_state[3] ^ t : 1)
        end
        function lane_center_cost(x, i, context_state, t)

            (myopic ? context_state[3] ^ t : 1) * norm_sqr(x[2] - lane_centers[i](x[1]))

            #  * cost # not sure how to really go about this. The lane center vector is not necessarily the same length as the state vector, how to resolve?
        end
        function collision_cost(x, i, context_state, t)
            if (i == 1 || i == 2)
                cost = [max(0.0, context_state[4] + 0.02 * context_state[4] - norm(x[Block(i)][1:2] - x[Block(paired_player)][1:2]))^2 for paired_player in 3:4]
            else
                cost = [max(0.0, context_state[4] + 0.02 * context_state[4] - norm(x[Block(i)][1:2] - x[Block(paired_player)][1:2]))^2 for paired_player in [1:(i - 1); (i + 1):num_players]]
            end
            sum(cost)
        end
        function cost_for_player(i, xs, us, context_state, T)
            early_target = target_cost(xs[1][Block(i)], context_state[Block(i)], 1)
            mean_target = mean([target_cost(xs[t + 1][Block(i)], context_state[Block(i)], t) for t in 1:T])
            minimum_target = minimum([target_cost(xs[t][Block(i)], context_state[Block(i)],t) for t in 1:T])
            control = mean([control_cost(us[t][Block(i)], context_state[Block(i)], t) for t in 1:T])
            safe_distance_violation = mean([collision_cost(xs[t], i, context_state[Block(i)], t) for t in 1:T])
            lane_center = mean([lane_center_cost(xs[t+1][Block(i)], i, context_state[Block(i)], t) for t in 1:T])

            #contex states 6-10

            # context_state[Block(i)][5] * early_target + 
            context_state[Block(i)][myopic ? 5 : 4] * mean_target + 
            # context_state[Block(i)][7] * minimum_target + 
            context_state[Block(i)][myopic ? 6 : 5] * control +
            # context_state[Block(i)][myopic ? 7 : 6] * safe_distance_violation +
            (i > 2 ? context_state[Block(i)][myopic ? 8 : 7] * lane_center : 0.0)

            # # 1.0 * early_target + 
            # 3.0 * mean_target +
            # # 1.0 * minimum_target + 
            # 0.01 * control +
            # 2.0 * safe_distance_violation
        end
        function cost_function(xs, us, context_state)
            num_players = blocksize(xs[1], 1)
            T = size(us,1)
            [cost_for_player(i, xs, us, context_state, T) for i in 1:num_players]
        end
        TrajectoryGameCost(cost_function, GeneralSumCostStructure())
    end
    dynamics = ProductDynamics([dynamics for _ in 1:num_players])
    game = TrajectoryGame(
        dynamics,
        cost,
        environment,
        shared_collision_avoidance_coupling_constraints(num_players, min_distance),
    )
    CollisionAvoidanceGame(game)


end

function init_crosswalk_game(
    full_state;
    state_dim = (4, 4),
    action_dim = (2, 2),
    σ_ = 0.0,
    game_environment = PolygonEnvironment(6, 8),
    initial_state = mortar([
        [0, 2, 0.1, -0.2], # initial x, y, initial velocity in x, y direction (player 1)
        [2.5, 2, 0.0, 0.0],# player 2
    ]),
    # FOR KINEMATIC BICYCLE MODEL: initial state = mortar([[0, 2, 0.2236, -1.10715], [2.5, 2, 0.0, 0.0]]) NOTE: heading is in radians and can be found using atan2
    game_params = mortar([
        [2, 0, 0.6], # starting position x, y, discount factor (player 1)
        [0, 0, 0.6]  # player 2
    ]),
    horizon = 25,
    num_players = 2,
    myopic = false,
    dynamics = planar_double_integrator(;
        state_bounds = (; lb = [-Inf, -Inf, -0.8, -0.8], ub = [Inf, Inf, 0.8, 0.8]),
        control_bounds = (; lb = [-10, -10], ub = [10, 10]),
    )
)
    game_structure = n_player_collision_avoidance(
        num_players;
        environment = game_environment,
        min_distance = 0.5,
        collision_avoidance_coefficient = 5.0,
        myopic = myopic, # TODO don't love
        dynamics = dynamics
    )
    if full_state
        observation_dim = 4
        observation_model = 
            (x; σ = σ_) -> 
            vcat(
                [ x[state_dim[1] * (i - 1)+1:state_dim[1]*i] .+ σ * randn(state_dim[1])
                    for i in 1:num_players]...
            )
    else
        observation_dim = 2
        observation_model = 
            (x; σ = σ_) -> 
            vcat(
                [ x[state_dim[1]*i-(state_dim[1]-1):state_dim[1]*i-(state_dim[1]-2)] .+ σ * randn(2)
                    for i in 1:num_players ]...
            )
    end

    (;
    initial_state = initial_state,
    game_parameters = game_params,
    environment = game_environment,
    observation_model = observation_model,
    observation_dim = observation_dim,
    horizon = horizon,
    state_dim = state_dim,
    action_dim = action_dim,
    σ = σ_,
    game_structure = game_structure,
    )
end

function init_bicycle_test_game(
    full_state;
    state_dim = (4, 4, 4),
    action_dim = (2, 2),
    σ_ = 0.0,
    game_environment = create_env(), 
    initial_state = mortar([
        [0, 2, 0.2236, 2*pi-1.10715], # initial x, y, initial velocity magnitude, heading angle (player 1)
        [2.5, 2, 0.0, 0.0],# player 2
        [2.5, 0, 0.0, 0.0] # player 3
    ]),
    game_params = mortar([
        [0.0, 0.0, 0.6, 1.0], # target x,y, discount, min_dist
        [0.0, 0.0, 0.6, 1.0],
        [0.0, 0.0, 0.6, 1.0]
    ]),
    horizon = 25,
    n = 3,
    dt = 0.04,
    myopic = false,
    verbose = false,
    dynamics = BicycleDynamics(;
        dt = dt, # needs to become framerate
        l = 1.0,
        state_bounds = (; lb = [-Inf, -Inf, -Inf, -Inf], ub = [Inf, Inf, Inf, Inf]),
        control_bounds = (; lb = [-5, -pi/2], ub = [5, pi/2]),
        integration_scheme = :forward_euler
    ),
    lane_centers = nothing,
)
    !verbose || print("initializing game ... ")
    game_structure = InD_collision_avoidance(
        n,
        lane_centers; # lane centers
        environment = game_environment,
        min_distance = 0.5,
        collision_avoidance_coefficient = 5.0,
        dynamics = dynamics,
        myopic = myopic
    )
    !verbose || print(" game structure initialized\n")
    if full_state
        observation_dim = state_dim[1]
        observation_model = 
            (x; σ = σ_) -> 
            vcat(
                [ x[state_dim[1] * (i - 1)+1:state_dim[1]*i] .+ σ * randn(state_dim[1])
                    for i in 1:n]...
            )
    else
        observation_dim = 2
        observation_model = 
            (x; σ = σ_) -> 
            vcat(
                [ x[state_dim[1]*i-(state_dim[1]-1):state_dim[1]*i-(state_dim[1]-2)]
                    for i in 1:n]...
            )
    end
    !verbose || println("observation model initialized")

    (;
    initial_state = initial_state,
    game_parameters = game_params,
    environment = game_environment,
    observation_model = observation_model,
    observation_dim = observation_dim,
    horizon = horizon,
    state_dim = state_dim,
    action_dim = action_dim,
    σ = σ_,
    game_structure = game_structure,
    )
end

function observe_trajectory(trajectory, game_init; blocked_by_time = true)
    if blocked_by_time
        map(blocks(trajectory)) do x
            BlockVector(game_init.observation_model(x), [game_init.observation_dim for _ in 1:num_players(game_init.game_structure.game)])
        end
    else # observe by state at time t
        BlockVector(game_init.observation_model(trajectory[1:end]), [game_init.observation_dim for _ in 1:num_players(game_init.game_structure.game)])
    end
end

end