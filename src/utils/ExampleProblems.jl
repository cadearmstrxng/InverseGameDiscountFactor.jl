module ExampleProblems

using BlockArrays: Block, blocksize
using TrajectoryGamesBase:
    GeneralSumCostStructure, ProductDynamics, TrajectoryGame, TrajectoryGameCost
using TrajectoryGamesExamples: planar_double_integrator
using LinearAlgebra: norm_sqr, norm
using Statistics: mean
using Infiltrator

export n_player_collision_avoidance, CollisionAvoidanceGame, HighwayGame

struct CollisionAvoidanceGame
    game::TrajectoryGame
end

struct HighwayGame
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

"""
Collision avoidance in multiplayer scenario, all trying to reach a given individual target

State layout:

point-mass states per player stacked into game state
x = [x₁ ... xᵢ ... xₙ]
where
xᵢ = [px py vx vy]
with p positional state and v spatial velocity

context state contains targets per player
c = [c₁ ... cᵢ ... cₙ]
where
cᵢ = [tx ty]
with (tx, ty) describing the position of the target

Collision avoidance bounds are modeled via n × (n-1) coupling constraints enforcing a minimal distance between all players
"""
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

end
