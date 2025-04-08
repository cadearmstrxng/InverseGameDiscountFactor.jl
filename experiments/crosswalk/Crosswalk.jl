module Crosswalk
using LinearAlgebra: norm, norm_sqr
using BlockArrays: blocksizes, mortar, BlockVector, blocks
using TrajectoryGamesExamples: BicycleDynamics, planar_double_integrator
using TrajectoryGamesBase: num_players
using PATHSolver: PATHSolver
using Infiltrator
using TrajectoryGamesBase:
    GeneralSumCostStructure,
    ProductDynamics,
    TrajectoryGame,
    TrajectoryGameCost,
    PolygonEnvironment,
    num_players
using TrajectoryGamesExamples:planar_double_integrator
using CairoMakie
include("../GameUtils.jl")
include("../graphing/ExperimentGraphingUtils.jl")
include("../../src/InverseGameDiscountFactor.jl")

export run_myopic_crosswalk_sim

function run_myopic_crosswalk_sim(full_state = true, graph = true, verbose = true)
    coeffs = [1.0, 0.1, 5.0]
    init = GameUtils.init_crosswalk_game(
        full_state;
        myopic = true,
        initial_state = mortar([
            [2.0, 2.0, 0, 0],
            [0, 2.0, 0, 0]]),
        game_params = mortar([
            [0, 0, 0.95],
            [2, 0, 0.9]
        ]),
        coeffs = coeffs,
        horizon = 25
    )

    init_baseline = GameUtils.init_crosswalk_game(
        full_state;
        myopic = false,
        initial_state = init.initial_state,
        game_params = mortar([
            [0, 0],
            [2, 0]
        ]),
        coeffs = coeffs,
        horizon = init.horizon
    )
    println("initialized game structure")
    
    mcp_game = InverseGameDiscountFactor.MCPCoupledOptimizationSolver(
        init.game_structure.game,
        init.horizon,
        blocksizes(init.game_parameters, 1)
    ).mcp_game


    mcp_game_baseline = InverseGameDiscountFactor.MCPCoupledOptimizationSolver(
        init_baseline.game_structure.game,
        init_baseline.horizon,
        blocksizes(init_baseline.game_parameters, 1)
    ).mcp_game

    println("solving forward game ... ")
    for i in 1:num_players(mcp_game.game)
        println("\tgame_parameters_i: ", init.game_parameters[(i-1)*3+1:i*3])
    end
    for i in 1:num_players(mcp_game.game)
        println("\tinitial_state_i: ", init.initial_state[(i-1)*4+1:i*4])
    end
    f_sol = InverseGameDiscountFactor.solve_mcp_game(
                mcp_game,
                init.initial_state,
                init.game_parameters;
                verbose = false
            )
    forward_solution = InverseGameDiscountFactor.reconstruct_solution(
        f_sol,
        init.game_structure.game,
        init.horizon
    )
        observed_forward_solution = GameUtils.observe_trajectory(forward_solution, init)
    println("solving inverse game ... ")

    method_sol = InverseGameDiscountFactor.solve_myopic_inverse_game(
        mcp_game,
        observed_forward_solution,
        init.observation_model,
        (3, 3);
        hidden_state_guess = init.game_parameters,
        max_grad_steps = 200,
        retries_on_divergence = 3,
        verbose = false,
        warm_start = false,
        lr = 1e-3
    )

    baseline_sol = InverseGameDiscountFactor.solve_myopic_inverse_game(
        mcp_game_baseline,
        observed_forward_solution,
        init.observation_model,
        (2, 2);
        hidden_state_guess = init_baseline.game_parameters,
        max_grad_steps = 200,
        retries_on_divergence = 3,
        verbose = false,
        warm_start = false,
        lr = 1e-3
    )
    println("solved inverse game")
    println("convergence percent ours: ", count(==(PATHSolver.MCP_Solved), method_sol.solving_status) / length(method_sol.solving_status))
    error = norm(method_sol.recovered_trajectory - forward_solution)
    println("ours: ", error)
    error_baseline = norm(baseline_sol.recovered_trajectory - forward_solution)
    println("baseline: ", error_baseline)
    println("improvement: ", (error_baseline - error) / error_baseline)

    method_traj = BlockVector(method_sol.recovered_trajectory, [init.state_dim[1] *2 for _ in 1:init.horizon])
    baseline_traj = BlockVector(baseline_sol.recovered_trajectory, [init.state_dim[1] *2 for _ in 1:init.horizon])
    if graph
        ExperimentGraphingUtils.graph_crosswalk_trajectories(
            "Our Method",
            [forward_solution, method_traj, baseline_traj],
            init.game_structure,
            init.horizon;
            observations = observed_forward_solution,
            colors = [
                [(:red, 1.0), (:blue, 1.0), (:green, 1.0)],
                [(:red, 0.5), (:blue, 0.5), (:green, 0.5)],
                [(:pink, 0.25), (:purple, 0.25), (:orange, 0.25)]
            ]
        )
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

function problemLandscape(
    initial_state = mortar([
        [0, 2, 0.1, -0.2],
        [2.5, 2, 0.0, 0.0],
    ]),
    hidden_params = mortar([[2, 0, 0.6], [0, 0, 0.6]]),
)
    horizon = 25
    environment = PolygonEnvironment(6, 8)
    game = GameUtils.n_player_collision_avoidance(2; environment, min_distance = 0.5, collision_avoidance_coefficient = 5.0)
    solver = InverseGameDiscountFactor.MCPCoupledOptimizationSolver(game.game, horizon, blocksizes(hidden_params, 1))
    mcp_game = solver.mcp_game

    forward_solution = InverseGameDiscountFactor.solve_mcp_game(mcp_game, initial_state, hidden_params; verbose = false)
    for_sol = InverseGameDiscountFactor.reconstruct_solution(forward_solution, game.game, horizon)

    context_state_guess = copy(hidden_params)

    granularity = 100
    left_point1 = 0.0
    right_point1 = 1.2

    left_point2 = 0.0
    right_point2 = 1.2

    # gammas1 = [round(i*((right_point1 - left_point1)/granularity) + left_point1, sigdigits=7) for i in left_point1:(right_point1 - left_point1)/granularity:right_point1]
    # gammas2 = [round(i*((right_point2 - left_point2)/granularity) + left_point2, sigdigits=7) for i in left_point2:(right_point2 - left_point2)/granularity:right_point2]
    gammas1 = LinRange(left_point1, right_point1, granularity)
    gammas2 = LinRange(left_point2, right_point2, granularity)
    println("Gammas1: ", gammas1)
    println("Gammas2: ", gammas2)
    costs = Array{Float64}(undef, length(gammas1), length(gammas2))

    function likelihood_cost(τs_observed, context_state_estimation, initial_state)
        solution = InverseGameDiscountFactor.solve_mcp_game(mcp_game, initial_state, 
            context_state_estimation;verbose=false)

        if solution.status != PATHSolver.MCP_Solved
            @info "Inner solve did not converge properly, re-initializing..."
            solution = InverseGameDiscountFactor.solve_mcp_game(mcp_game, initial_state, 
                context_state_estimation; initial_guess = nothing)
        end
        # push!(solving_info, solution.info)
        # last_solution = solution.status == PATHSolver.MCP_Solved ? (; primals = ForwardDiff.value.(solution.primals),
        # variables = ForwardDiff.value.(solution.variables), status = solution.status) : nothing
        τs_solution = InverseGameDiscountFactor.reconstruct_solution(solution, mcp_game.game, horizon)
        observed_τs_solution = τs_solution
    
        # @infiltrate
        
        # if solution.status == PATHSolver.MCP_Solved
        #     infeasible_counter = 0
        # else
        #     infeasible_counter += 1
        # end
        # @infiltrate
        norm_sqr(τs_observed - observed_τs_solution)
    end

    for i in eachindex(gammas1)
        context_state_guess[3] = gammas1[i]
        for j in eachindex(gammas2)
            context_state_guess[6] = gammas2[j]
            # @infiltrate
            costs[i, j] = likelihood_cost(for_sol, context_state_guess, initial_state)
        end        
    end

    fig1 = CairoMakie.Figure()
    ax1 = CairoMakie.Axis(fig1[1, 1],
    # xticks = (1:length(gammas), gammas),
    xlabel = "Gamma P1",
    # yticks = (1:length(gammas), gammas),
    ylabel = "Gamma P2")

    CairoMakie.heatmap!(ax1,gammas1,gammas2, costs, colormap = :viridis)
    # CairoMakie.Colorbar(fig1[1, 2], ax1, label = "Cost")
    # Colorbar(fig1[1, 2], limits = (min(costs...), max(costs...)), colormap = :viridis)
    # CairoMakie.xlabel!(ax1, "Gamma P1")
    # CairoMakie.ylabel!(ax1, "Gamma P2")
    # CairoMakie.Legend(fig1[2, 1], [ax1], ["Cost"])

    # println(costs)

    CairoMakie.save("ProblemLandscape.png", fig1)

end


end