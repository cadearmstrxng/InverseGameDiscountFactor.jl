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

export run_waymax_sim, test_waymax_sim

struct CollisionAvoidanceGame
    game::TrajectoryGame
end

function run_waymax_sim(initial_agent_states;full_state=false, graph=false, verbose = false, myopic = true)

    rng = MersenneTwister(1234)
    Random.seed!(rng)

    # Find Lane Center Funcs
    # ego_roadgraph_idx = 3001:3340
    # agent_3_roadgraph_idx = vcat(2000:2109, 2676:2900)
    # agent_8_roadgraph_idx = [4441]
    # agent_2_roadgraph_idx = agent_8_roadgraph_idx

    roadgraph_points = pull_roadpoints("experiments/waymax/data/roadgraph_points.txt")
    # ego_roadgraph_points = roadgraph_points[ego_roadgraph_idx]
    # agent_3_roadgraph_points = roadgraph_points[agent_3_roadgraph_idx]
    # agent_8_roadgraph_points = roadgraph_points[agent_8_roadgraph_idx]
    # agent_2_roadgraph_points = roadgraph_points[agent_2_roadgraph_idx]

    # ego_model, ego_params = calculate_road_func(ego_roadgraph_points)
    # ego_params[4] -= 2
    # agent_3_model, agent_3_params = calculate_road_func(agent_3_roadgraph_points)
    # agent_8_point = roadgraph_points[agent_8_roadgraph_idx][1]
    # agent_2_point = roadgraph_points[agent_2_roadgraph_idx][1]

    # ego_lane_center = x -> ego_model(x, ego_params)
    # agent_3_lane_center = x -> agent_3_model(x, agent_3_params)
    # agent_8_lane_center = x -> agent_8_point[2]
    # agent_2_lane_center = x -> agent_2_point[2]

    # lane_centers = [ego_lane_center, agent_3_lane_center, agent_8_lane_center, agent_2_lane_center]

    
    agent_states_float = [[Float64(x) for x in line] for line in initial_agent_states]
    agent_states = [BlockArray(state, [4, 4, 4, 4]) for state in agent_states_float]
    
    initial_state = agent_states[1]
    goal_init_guess = mean(roadgraph_points)
    if myopic
        game_params = mortar([[vcat(goal_init_guess, [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]) for i in 1:4]...])
    else
        game_params = mortar([[vcat(goal_init_guess, [1.0, 1.0, 1.0, 1.0, 1.0]) for i in 1:4]...])
    end
    game_params[1:2] = [2557, -5800] # Ego agent knows its goal
    
    horizon = length(agent_states)

    init = init_waymax_test_game(;
        initial_state = initial_state,
        game_params = game_params,
        horizon = horizon,
        n = 4,
        myopic = myopic,
        verbose = verbose,
        # lane_centers = lane_centers,
    )

    !verbose || println("initializing mcp coupled optimization solver...")
    mcp_solver = InverseGameDiscountFactor.MCPCoupledOptimizationSolver(
        init.game_structure.game,
        init.horizon,
        blocksizes(init.game_parameters, 1)
    )
    return function get_next_action_inverse(current_agent_states;)
        agent_states_float = [BlockVector([Float64(x) for x in line], [4, 4, 4, 4]) for line in current_agent_states]

        !verbose || println("solving inverse game")
        method_sol = InverseGameDiscountFactor.solve_myopic_inverse_game(
            mcp_solver.mcp_game,
            agent_states_float,
            init.observation_model,
            Tuple(blocksizes(init.game_parameters, 1));
            initial_state = init.initial_state,
            hidden_state_guess = init.game_parameters[1:end],
            max_grad_steps = 20,
            verbose = verbose,
            total_horizon = horizon,
            lr = 1e-4,
        )
        recovered_params = method_sol.recovered_params
        current_state = agent_states_float[end]
        !verbose || println("params: ", recovered_params[1:2])
        !verbose || println("current state: ", round.(current_state[1:4], digits=2))
        # recovered_params[1:2] = [2557, -5650]
        forward_solution = solve_mcp_game(
            mcp_solver.mcp_game,
            current_state,
            recovered_params
        )
        primals = forward_solution.primals
        player_state_dim = 4
        player_control_dim = 2
        ego_action_t1 = primals[1][player_state_dim*horizon+1:player_state_dim*horizon+player_control_dim]
        return ego_action_t1
    end
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
    game_environment = InverseGameDiscountFactor.NullEnv(), 
    initial_state = nothing,
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
        control_bounds = (; lb = [-5, -pi/5], ub = [3, pi/5]),
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

function Waymax_collision_avoidance(
    num_players,
    lane_centers;
    environment,
    dynamics = nothing,
    myopic = true)

    cost = let
        function target_cost(x, context_state, t)
            (myopic ? context_state[3] ^ t : 1) * 2 * norm_sqr(x[1:2] - context_state[1:2])
        end
        function control_cost(u, context_state, t)
            0.1 * norm_sqr(u) * (myopic ? context_state[3] ^ t : 1)
        end
        function collision_cost(x, i, context_state, t)
            mapreduce(+, [1:(i-1); (i+1):num_players]) do paired_player
                -0.15 * norm_sqr(x[Block(i)][1:2] - x[Block(paired_player)][1:2])
            end
        end
        function cost_for_player(i, xs, us, context_state, T)
            mean_target = mean([target_cost(xs[t + 1][Block(i)], context_state[Block(i)], t) for t in 1:T])
            control = mean([control_cost(us[t][Block(i)], context_state[Block(i)], t) for t in 1:T])
            safe_distance_violation = mean([collision_cost(xs[t], i, context_state[Block(i)], t) for t in 1:T])

            context_state[Block(i)][myopic ? 5 : 4] * mean_target + 
            context_state[Block(i)][myopic ? 6 : 5] * control +
            context_state[Block(i)][myopic ? 7 : 6] * safe_distance_violation 
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