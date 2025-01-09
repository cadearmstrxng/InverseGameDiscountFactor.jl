module GameUtils

using BlockArrays: Block, blocksize
using TrajectoryGamesBase:
    GeneralSumCostStructure, ProductDynamics, TrajectoryGame, TrajectoryGameCost
using TrajectoryGamesExamples: planar_double_integrator
using LinearAlgebra: norm_sqr, norm
using Statistics: mean

export n_player_collision_avoidance, CollisionAvoidanceGame

struct CollisionAvoidanceGame
    game::TrajectoryGame
end

function my_norm_sqr(x)
    x'*x
end

function shared_collision_avoidance_coupling_constraints(num_players, min_distance)
    function coupling_constraint(xs, us)
        mapreduce(vcat, 1:(num_players - 1)) do player_i
            mapreduce(vcat, (player_i + 1):num_players) do paired_player
                map(xs) do x
                    my_norm_sqr(x[Block(player_i)][1:2] - x[Block(paired_player)][1:2]) -
                    min_distance^2
                end
            end
        end
    end
end

function n_player_collision_avoidance(
    num_players;
    environment,
    min_distance = 1.0,
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
            cost = [max(0.0, min_distance + 0.2 * min_distance - norm(x[Block(i)][1:2] - x[Block(paired_player)][1:2]))^2 for paired_player in [1:(i - 1); (i + 1):num_players]]
            sum(cost) 
        end
        function cost_for_player(i, xs, us, context_state, T)
            early_target = target_cost(xs[1][Block(i)], context_state[Block(i)], 1)
            mean_target = mean([target_cost(xs[t + 1][Block(i)], context_state[Block(i)], t) for t in 1:T])
            minimum_target = minimum([target_cost(xs[t][Block(i)], context_state[Block(i)],t) for t in 1:T])
            control = mean([control_cost(us[t][Block(i)], context_state[Block(i)], t) for t in 1:T])
            safe_distance_violation = mean([collision_cost(xs[t], i, context_state[Block(i)], t) for t in 1:T])

            0.0 * early_target + 1.0 * mean_target + 0.0 * minimum_target + 0.1 * control + collision_avoidance_coefficient * safe_distance_violation
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
    σ = 0.0,
    game_environment = PolygonEnvironment(6, 8),
    initial_state = mortar([
        [0, 2, 0.1, -0.2], # initial x, y, initial velocity in x, y direction (player 1)
        [2.5, 2, 0.0, 0.0],# player 2
    ]),
    game_params = mortar([
        [2, 0, 0.6], # starting position x, y, discount factor (player 1)
        [0, 0, 0.6]  # player 2
    ]),
    horizon = 25,
    num_players = 2,
    myopic = false
)
    if full_state
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
    initial_state = initial_state,
    game_parameters = game_params,
    environment = game_environment,
    horizon = horizon,
    state_dim = state_dim,
    action_dim = action_dim,
    σ = σ,
    game_structure = n_player_collision_avoidance(
        num_players;
        game_environment,
        min_distance = 0.5,
        collision_avoidance_coefficient = 5.0,
        myopic = myopic # TODO don't love
    ),
    )
end

end