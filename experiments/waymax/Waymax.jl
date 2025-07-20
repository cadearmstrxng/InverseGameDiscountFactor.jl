module Waymax

using BlockArrays
using TrajectoryGamesBase
using TrajectoryGamesExamples: BicycleDynamics
using LinearAlgebra
using Statistics: mean,median
using Random
using Infiltrator
using Optim

using InverseGameDiscountFactor

export init_solver, get_action

struct CollisionAvoidanceGame
    game::TrajectoryGame
end

struct WaymaxEnv
end

function TrajectoryGamesBase.get_constraints(env::WaymaxEnv, i)
    # [2546.63, -5539.43,
    n=2
    function (state)
        return [
            # norm_sqr(state[1:2] - [2549, -5547]) - 5^2
            norm_sqr(state[1:2] - [2549, -5548]) - 5^2
            norm_sqr(state[1:2] - [2548, -5544]) - 3^2
            # ((1/3(state[1] - 2548))^n + (state[2] + 5541)^n)^1/n - 1
            # (1/1.15(state[1] - 2550))^2 + (state[2] + 5547)^2 - 1
            ]
    end
end

function init_solver(;
    horizon = 11,
    myopic = false,
    verbose = false,
)
    goal_init_guess = [2543.9316778018742, -5584.346918384463] # mean of all roadgraph points, pulled out.
    if myopic
        game_params = mortar([[vcat(goal_init_guess, [1.0, 5, 0.1, 0.75, 7, -4, 2]) for i in 1:4]...])
        game_params[1:9] = [2555, -5600, 1.0, 5, 0.1, 0.75, 7, -4, 2] # Ego agent knows its goal
    else
        game_params = mortar([[vcat(goal_init_guess, [5, 0.1, 1.0, 7, -4, 2]) for i in 1:4]...])
        game_params[1:8] = [2555, -5600, 5, 0.1, 1.0, 7, -4, 2] # Ego agent knows its goal
    end
    
    init = init_waymax_test_game(;
        game_params = game_params,
        horizon = horizon,
        n = 4,
        myopic = myopic,
        verbose = verbose,
        # lane_centers = lane_centers,
    )
    mcp_solver = InverseGameDiscountFactor.MCPCoupledOptimizationSolver(
        init.game_structure.game,
        init.horizon,
        blocksizes(init.game_parameters, 1)
    )
    return init, mcp_solver
end

function get_action(init, mcp_solver, current_agent_states;verbose = false)
    agent_states_float = [BlockVector([Float64(x) for x in line], [4, 4, 4, 4]) for line in current_agent_states]
    initial_state = agent_states_float[end]
    !verbose || println("solving inverse game")
    method_sol = InverseGameDiscountFactor.solve_myopic_inverse_game(
        mcp_solver.mcp_game,
        agent_states_float,
        init.observation_model,
        Tuple(blocksizes(init.game_parameters, 1));
        initial_state = initial_state,
        hidden_state_guess = init.game_parameters[1:end],
        max_grad_steps = 2,
        verbose = verbose,
        total_horizon = init.horizon,
        lr = 1e-4,
        frozen_ego = true,
        freeze_initial_state = true
    )
    recovered_params = method_sol.recovered_params
    current_state = agent_states_float[end]
    !verbose || println("current state: ", round.(current_state[1:4], digits=2))
    forward_solution = solve_mcp_game(
        mcp_solver.mcp_game,
        current_state,
        recovered_params
    )
    primals = forward_solution.primals
    actions = BlockVector(vcat([primals[ii][4*init.horizon+1:4*init.horizon+2] for ii in 1:4]...), [2 for _ in 1:4])
    projected_actions = BlockVector(vcat([[min(max(action[1], -5), 3), max(min(action[2], pi/7), -pi/7)] for action in actions.blocks]...), [2 for _ in 1:4])
    raw_state = init.game_structure.game.dynamics(current_state, projected_actions).blocks[1]
    return [raw_state[1], raw_state[2], raw_state[4], raw_state[3]*sin(raw_state[4]), raw_state[3]*cos(raw_state[4])], projected_actions, recovered_params
end

function calculate_road_func(roadpoints; display_plot = false)
    xs = map(p -> p[1], roadpoints)
    ys = map(p -> p[2], roadpoints)

    model(x, p) = p[1] * exp(p[2] * (x - p[3])) + p[4]
    
    loss(p) = sum(((x->model(x,p)).(xs) .- ys).^2)

    p0 = [1.0, 1.0, median(xs), median(ys)]

    lower = [-Inf, -100.0, -Inf, -Inf]
    upper = [Inf, 100.0, Inf, Inf]

    result = optimize(loss, lower, upper, p0, Fminbox(LBFGS()))
    best_params = Optim.minimizer(result)
    println("Fit complete. Best parameters [c1, c2, c3]: ", best_params)

    return model, best_params
end

function pull_roadpoints(filename)
    data = readlines(filename)
    roadpoints = []
    for line in data
        x, y = split(line, " ")
        push!(roadpoints, [parse(Float64, x), parse(Float64, y)])
    end
    return roadpoints
end

function init_waymax_test_game(;
    state_dim = (4, 4, 4, 4),
    action_dim = (2, 2, 2, 2),
    σ_ = 0.0,
    game_environment = WaymaxEnv(),
    game_params = nothing,
    horizon = 10,
    n = 4,
    myopic = false,
    verbose = false,
    lane_centers = nothing,
    dynamics = BicycleDynamics(;
        dt = 0.1,
        l = 1.0,
        state_bounds = (; lb = [-Inf, -Inf, -Inf, -Inf], ub = [Inf, Inf, Inf, Inf]),
        control_bounds = (; lb = [-12, -pi/7], ub = [5, pi/7]),
        integration_scheme = :forward_euler
    ),
)
    !verbose || println("initializing game ... ")
    game_structure = Waymax_collision_avoidance(
        n,
        lane_centers;
        environment = game_environment,
        myopic = myopic,
        dynamics = dynamics
    )
    observation_dim = state_dim[1]
    observation_model = 
        (x; σ = σ_) -> 
        BlockVector(
            vcat(
                [ x[state_dim[1] * (i - 1)+1:state_dim[1]*i] .+ σ * randn(state_dim[1])
                    for i in 1:n]...
            ),
            [state_dim[1] for _ in 1:n]
        )
    (;
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

function Waymax_collision_avoidance(
    num_players,
    lane_centers;
    environment,
    dynamics = nothing,
    myopic = true)

    cost = let
        function target_cost(x, context_state, t)
            (myopic ? context_state[3] ^ t : 1) * norm_sqr(x[1:2] - context_state[1:2])
        end
        function control_cost(u, context_state, t)
            norm_sqr(u) * (myopic ? context_state[3] ^ t : 1)
        end
        function collision_cost(x, i, context_state, t)
            mapreduce(+, [1:(i-1); (i+1):num_players]) do paired_player
                # 10 * log(1 + exp(-9*norm_sqr(x[Block(i)][1:2] - x[Block(paired_player)][1:2])))
                -1 * norm_sqr(x[Block(i)][1:2] - x[Block(paired_player)][1:2])
            end
        end
        function lane_center_cost(x, i, context_state, t)
            (x[Block(i)][1] - 2555)^2
        end
        function road_boundary_cost(x, i, context_state, t)
            (x[Block(i)][1] - 2552)^2
        end
        function velocity_cost(x, i, context_state, t)
            norm_sqr(x[Block(i)][3] - 15)
        end
        function cost_for_player(i, xs, us, context_state, T)
            mean_target = mean([target_cost(xs[t + 1][Block(i)], context_state[Block(i)], t) for t in 1:T])
            control = mean([control_cost(us[t][Block(i)], context_state[Block(i)], t) for t in 1:T])
            safe_distance_violation = mean([collision_cost(xs[t], i, context_state[Block(i)], t) for t in 1:T])
            lane_center = mean([lane_center_cost(xs[t], i, context_state[Block(i)], t) for t in 1:T])
            road_boundary = mean([road_boundary_cost(xs[t], i, context_state[Block(i)], t) for t in 1:T])
            velocity = mean([velocity_cost(xs[t], i, context_state[Block(i)], t) for t in 1:T])
            
            context_state[Block(i)][myopic ? 4 : 3] * mean_target + # TODO basic screenshots at some times.could annotate with gamma that ego is inferring.
            context_state[Block(i)][myopic ? 5 : 4] * control +
            context_state[Block(i)][myopic ? 6 : 5] * safe_distance_violation + 
            context_state[Block(i)][myopic ? 7 : 6] * lane_center + 
            context_state[Block(i)][myopic ? 8 : 7] * road_boundary +
            context_state[Block(i)][myopic ? 9 : 8] * velocity
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
        nothing
    )
    InverseGameDiscountFactor.CollisionAvoidanceGame(game)
end

function norm_sqr(x)
    return sum(x.^2)
end

end 