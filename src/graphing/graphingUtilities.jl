using CairoMakie
using Statistics

using Infiltrator

using PATHSolver: PATHSolver
using LinearAlgebra: I, norm_sqr, pinv, ColumnNorm, qr, norm
using ParametricMCPs

using DifferentiableTrajectoryOptimization:
    ParametricTrajectoryOptimizationProblem, get_constraints_from_box_bounds, _coo_from_sparse!
using TrajectoryGamesExamples:
    TrajectoryGamesExamples,
    PolygonEnvironment,
    two_player_meta_tag,
    animate_sim_steps,
    planar_double_integrator,
    UnicycleDynamics,
    create_environment_axis
using TrajectoryGamesBase:
    TrajectoryGamesBase,
    TrajectoryGame,
    get_constraints,
    num_players,
    state_dim,
    control_dim,
    horizon,
    state_bounds,
    control_bounds,
    solve_trajectory_game!,
    JointStrategy,
    RecedingHorizonStrategy,
    rollout,
    ProductDynamics

export GenerateFrontPageFigure

function GenerateFrontPageFigure(
    initial_state = mortar([
        [0.5, 1.65, 0.1, -0.2],
        [1.4, 1.6, 0.0, 0.0],
    ]),
    hidden_params = mortar([[1.45, 0.3, 0.95], [0.55, 0.25, 0.9]]),
)

    CairoMakie.activate!();
    environment = PolygonEnvironment(6, 8)
    game = n_player_collision_avoidance(2; environment, min_distance = 0.5, collision_avoidance_coefficient = 5.0)
    baseline_game = n_player_collision_avoidance(2; environment, min_distance = 0.5, collision_avoidance_coefficient = 5.0, myopic = false)

    if max(hidden_params[Block(1)][3], hidden_params[Block(2)][3]) == 1
        horizon = 75
    else
        horizon = convert(Int64, ceil(log(1e-4)/(log(max(hidden_params[Block(1)][3], hidden_params[Block(2)][3])))))
    end
    horizon = 50

    turn_length = 2
    solver = MCPCoupledOptimizationSolver(game.game, horizon, blocksizes(hidden_params, 1))
    baseline_solver = MCPCoupledOptimizationSolver(baseline_game.game, horizon, [2, 2])
    mcp_game = solver.mcp_game
    baseline_mcp_game = baseline_solver.mcp_game

    forward_solution = solve_mcp_game(mcp_game, initial_state, hidden_params; verbose = false)
    for_solution = reconstruct_solution(forward_solution, game.game, horizon)

    context_state_estimation = mortar([[1.480559515058807, 0.33095477218790573, 0.9487914269199791], [0.518935331080876, 0.2737633025260197, 0.8973704964094965]])

    baseline_context_state_estimation = mortar([[1.4802915250244733, 0.2814143611765102, 1.0], [0.5327075874529416, 0.1794895383095173, 1.0]])

    inv_sol = solve_mcp_game(mcp_game, initial_state, context_state_estimation; verbose = false)
    inv_solution = reconstruct_solution(inv_sol, game.game, horizon)

    baseline_sol = solve_mcp_game(mcp_game, initial_state, baseline_context_state_estimation; verbose = false)
    baseline_solution = reconstruct_solution(baseline_sol, game.game, horizon)

    observation_model_noisy = (; σ = 0.03, expected_observation = x -> x .+ observation_model_noisy.σ * randn(length(x)))
    noisy_solution = vcat([observation_model_noisy.expected_observation(state_t) for state_t in for_solution.blocks]...)
    noisy_solution = BlockVector(noisy_solution, [8 for _ in 1:horizon])

    fig = CairoMakie.Figure(resolution = (620,641))
    image_data = CairoMakie.load("C:\\Users\\Owner\\Documents\\Research Summer 2024\\InverseGameDiscountFactor.jl\\src\\dean_keeton_whitis.png")
    ax1 = Axis(fig[1,1], aspect = DataAspect())
    x = 2
    y = x*641/620
    image!(ax1, 0..x, 0..y, rotr90(image_data); interpolation = false)
    snapshot_time = 10

    player1xForward = Float64[]
    player1yForward = Float64[]
    player2xForward = Float64[]
    player2yForward = Float64[]
    player1xInverse = Float64[]
    player1yInverse = Float64[]
    player2xInverse = Float64[]
    player2yInverse = Float64[]
    player1xBaseline = Float64[]
    player1yBaseline = Float64[]
    player2xBaseline = Float64[]
    player2yBaseline = Float64[]

    for ii in 1:30
        push!(player1xForward, for_solution[Block(ii)][1])
        push!(player1yForward, for_solution[Block(ii)][2])
        push!(player2xForward, for_solution[Block(ii)][5])
        push!(player2yForward, for_solution[Block(ii)][6])
        push!(player1xInverse, inv_solution[Block(ii)][1])
        push!(player1yInverse, inv_solution[Block(ii)][2])
        push!(player2xInverse, inv_solution[Block(ii)][5])
        push!(player2yInverse, inv_solution[Block(ii)][6])
        push!(player1xBaseline, baseline_solution[Block(ii)][1])
        push!(player1yBaseline, baseline_solution[Block(ii)][2])
        push!(player2xBaseline, baseline_solution[Block(ii)][5])
        push!(player2yBaseline, baseline_solution[Block(ii)][6])
    end

    for ii in 1:30
        scatter!(ax1, [noisy_solution[Block(ii)][1]], [noisy_solution[Block(ii)][2]], color = :crimson, alpha = 0.5)
        scatter!(ax1, [noisy_solution[Block(ii)][5]], [noisy_solution[Block(ii)][6]], color = :dodgerblue, alpha = 0.5)   
    end

    lines!(ax1, player1xBaseline, player1yBaseline, color = :tomato4, linewidth = 3)
    lines!(ax1, player2xBaseline, player2yBaseline, color = :cyan4, linewidth = 3)
    lines!(ax1, player1xInverse, player1yInverse, color = :gold, linewidth = 3)
    lines!(ax1, player2xInverse, player2yInverse, color = :cyan, linewidth = 3)
    
    p1_noisy = scatter!(ax1, [noisy_solution[Block(snapshot_time)][1]], [noisy_solution[Block(snapshot_time)][2]], color = :crimson, alpha = 0.5)
    p2_noisy = scatter!(ax1, [noisy_solution[Block(snapshot_time)][5]], [noisy_solution[Block(snapshot_time)][6]], color = :dodgerblue, alpha = 0.5)
    p1_baseline = scatter!(ax1, [baseline_solution[Block(snapshot_time)][1]], [baseline_solution[Block(snapshot_time)][2]], color = :tomato4, alpha = 0)
    p2_baseline = scatter!(ax1, [baseline_solution[Block(snapshot_time)][5]], [baseline_solution[Block(snapshot_time)][6]], color = :cyan4, alpha = 0)
    p1_inv = scatter!(ax1, [inv_solution[Block(snapshot_time)][1]], [inv_solution[Block(snapshot_time)][2]], color = :gold, alpha = 0)
    p2_inv = scatter!(ax1, [inv_solution[Block(snapshot_time)][5]], [inv_solution[Block(snapshot_time)][6]], color = :cyan, alpha = 0)
    p1_goal = scatter!(ax1, [hidden_params[Block(1)][1]], [hidden_params[Block(1)][2]], color = :red, marker = :star5, markersize = 23, strokewidth = 2)
    p2_goal = scatter!(ax1, [hidden_params[Block(2)][1]], [hidden_params[Block(2)][2]], color = :deepskyblue1, marker = :star5, markersize = 23, strokewidth = 2)

    player1rotation = atan(inv_solution[Block(snapshot_time+2)][2] - inv_solution[Block(snapshot_time+1)][2], inv_solution[Block(snapshot_time+2)][1] - inv_solution[Block(snapshot_time+1)][1]) - pi/2
    player2rotation = atan(inv_solution[Block(snapshot_time+2)][6] - inv_solution[Block(snapshot_time+1)][6], inv_solution[Block(snapshot_time+2)][5] - inv_solution[Block(snapshot_time+1)][5]) - pi/2
    
    arrow_path = BezierPath([
        MoveTo(Point(0, 0)),
        LineTo(Point(0.3, -0.3)),
        LineTo(Point(0.15, -0.3)),
        LineTo(Point(0.3, -1)),
        LineTo(Point(0, -0.9)),
        LineTo(Point(-0.3, -1)),
        LineTo(Point(-0.15, -0.3)),
        LineTo(Point(-0.3, -0.3)),
        ClosePath()
    ])


    scatter!(ax1, [inv_solution[Block(snapshot_time+2)][1]], [inv_solution[Block(snapshot_time+2)][2]], color = :orangered, marker = arrow_path, markersize = 30, strokewidth = 2, rotations = player1rotation)
    scatter!(ax1, [inv_solution[Block(snapshot_time+2)][5]], [inv_solution[Block(snapshot_time+2)][6]], color = :skyblue1, marker = arrow_path, markersize = 30, strokewidth = 2, rotations = player2rotation)
    scatter!(ax1, [inv_solution[Block(snapshot_time+1)][1]], [inv_solution[Block(snapshot_time+1)][2]], color = :black, marker = '1', markersize = 13)
    scatter!(ax1, [inv_solution[Block(snapshot_time+1)][5] + 0.1/4], [inv_solution[Block(snapshot_time+1)][6]+0.1/3], color = :black, marker = '2', markersize = 13)
    

    Legend(fig[1,2], 
    [p1_noisy, p1_inv, p1_baseline, p2_noisy, p2_inv, p2_baseline], 
    ["Agent 1 Observations", "Agent 1 Recovered", "Agent 1 Baseline", "Agent 2 Observations", "Agent 2 Recovered", "Agent 2 Baseline"])

    CairoMakie.save("front_page_figure.png", fig)
end



function GeneratePartialStateGraphs(;
    noise_level_increment = 0.002,
    num_trials = 50,
    overlap_shift = 0.0005,
    directory = "./Graphs/"
    )
    experiment_results_file = "./experiments/partial state/partial_state_exp.result.txt"
    @assert isfile(experiment_results_file)
    
    noise_levels = [noise_level_increment*i for i in 0:50]

    file_content = open(experiment_results_file, "r") do f
        println("reading file")
        readlines(f)
    end

    ground_truth_parameters = [parse(Float64, strip(x)) for x in split(file_content[3][2:end-1],',')]
    ground_truth_goals = vcat(ground_truth_parameters[1:2], ground_truth_parameters[3:4])

    errors = Dict{Float64, Vector{Float64}}()
    for noise_level in noise_levels
        errors[noise_level] = Float64[]
    end
    num_semicolons = 0
    # Split and parse the line
    for e in split(file_content[5][2:end-1])
        e = replace(e, "," => "")
        # Check if semicolon is present
        if occursin(';', e)
            # Remove semicolon
            e = replace(e, ";" => "")
            errors[noise_level_increment * num_semicolons] = push!(errors[noise_level_increment * num_semicolons], parse(Float64, e))
            num_semicolons += 1
        else
            errors[noise_level_increment * num_semicolons] = push!(errors[noise_level_increment * num_semicolons], parse(Float64, e))
        end
    end
    baseline_errors = Dict{Float64, Vector{Float64}}()
    for noise_level in noise_levels
        baseline_errors[noise_level] = Float64[]
    end
    num_semicolons = 0
    # Split and parse the line
    for e in split(file_content[7][2:end-1])
        e = replace(e, "," => "")
        # Check if semicolon is present
        if occursin(';', e)
            # Remove semicolon
            e = replace(e, ";" => "")
            baseline_errors[noise_level_increment * num_semicolons] = push!(baseline_errors[noise_level_increment * num_semicolons], parse(Float64, e))
            num_semicolons += 1
        else
            baseline_errors[noise_level_increment * num_semicolons] = push!(baseline_errors[noise_level_increment * num_semicolons], parse(Float64, e))
        end
    end


    # Initialize dictionary
    recovered_parameters = Dict{Float64, Vector{Vector{Float64}}}()
    for noise_level in noise_levels
        recovered_parameters[noise_level] = []
    end

    noise_level = 1  # Start at first index
    num_trials_in_bucket = 0

    # Split and process the line
    for e in split(file_content[9][2:end-1], "], ")
        if num_trials_in_bucket == num_trials
            noise_level += 1
            num_trials_in_bucket = 0
        end
        params_raw = split(e[2:end], ',')
        params = [parse(Float64, strip(p, ['[', ']', ' '])) for p in params_raw]
        push!(recovered_parameters[noise_levels[noise_level]], params)
        num_trials_in_bucket += 1
    end

    baseline_recovered_parameters = Dict{Float64, Vector{Vector{Float64}}}()
    for noise_level in noise_levels
        baseline_recovered_parameters[noise_level] = []
    end
    noise_level = 1
    num_trials_in_bucket = 0
    for e in split(file_content[11][2:end-1], "], ")
        if num_trials_in_bucket == num_trials
            noise_level += 1
            num_trials_in_bucket = 0
        end
        params_raw = split(e[2:end], ',')
        params = [parse(Float64, strip(p, ['[', ']', ' '])) for p in params_raw]
        push!(baseline_recovered_parameters[noise_levels[noise_level]], params)
        num_trials_in_bucket += 1
    end

    fig1 = CairoMakie.Figure()
    ax1 = CairoMakie.Axis(fig1[1, 1],
    xlabel = "Max Absolute Postion Observation Error",
    ylabel = "Mean Aboslute Position Prediction Error",
    yticks = 3:1:8,
    yminorticks = IntervalsBetween(4),
    xticks = 0.0:0.01:0.1,
    xminorticks = IntervalsBetween(2),
    )

    mean_errors = [Statistics.mean(errors[i]) for i in noise_levels]
    stds = [Statistics.std(errors[i]) for i in noise_levels]
    baseline_mean_errors = [Statistics.mean(baseline_errors[i]) for i in noise_levels]
    baseline_stds = [Statistics.std(baseline_errors[i]) for i in noise_levels]
    our_method = CairoMakie.scatter!(ax1, noise_levels, mean_errors, color = (:blue, 0.75))
    CairoMakie.errorbars!(ax1, noise_levels, mean_errors, stds, color = (:blue, 0.75))
    baseline = CairoMakie.scatter!(ax1, noise_levels, baseline_mean_errors, color = (:red, 0.75))
    CairoMakie.errorbars!(ax1, noise_levels, baseline_mean_errors, baseline_stds, color = (:red, 0.75))

    CairoMakie.axislegend(ax1, [our_method, baseline], ["Our Method", "Baseline"], position = :lt)
    CairoMakie.save(directory*"NoiseGraph.png", fig1)


    fig2 = CairoMakie.Figure()
    ax2 = CairoMakie.Axis(fig2[1, 1],
    xlabel = "Max Absolute Postion Observation Error",
    ylabel = "Mean Absolute Parameter Prediction Error",
    xticks = 0.0:0.01:0.1,
    xminorticks = IntervalsBetween(2),
    )

    goal_indices = [1, 2, 4, 5]

    full_param_mean_errors = [mean([norm(ground_truth_parameters - recovered_parameters[i][j]) for j in 1:num_trials]) for i in noise_levels]
    full_param_stds = [std([norm(ground_truth_parameters - recovered_parameters[i][j]) for j in 1:num_trials]) for i in noise_levels]
    baseline_full_param_mean_errors = [mean([norm(ground_truth_parameters - baseline_recovered_parameters[i][j]) for j in 1:num_trials]) for i in noise_levels]
    baseline_full_param_stds = [std([norm(ground_truth_parameters - baseline_recovered_parameters[i][j]) for j in 1:num_trials]) for i in noise_levels]
    our_method = CairoMakie.scatter!(ax2, noise_levels, full_param_mean_errors, color = (:blue, 0.75))
    CairoMakie.errorbars!(ax2, noise_levels, full_param_mean_errors, full_param_stds, color = (:blue, 0.75))
    baseline = CairoMakie.scatter!(ax2, noise_levels, baseline_full_param_mean_errors, color = (:red, 0.75))
    CairoMakie.errorbars!(ax2, noise_levels, baseline_full_param_mean_errors, baseline_full_param_stds, color = (:red, 0.75))

    CairoMakie.axislegend(ax2, [our_method, baseline], ["Our Method", "Baseline"], position = :lt)
    CairoMakie.save(directory*"ParamErrorWDiscount.png", fig2)


    fig3 = CairoMakie.Figure()
    ax3 = CairoMakie.Axis(fig3[1, 1],
    xlabel = "Max Absolute Postion Observation Error",
    ylabel = "Mean Absolute Goal Prediction Error",
    xticks = 0.0:0.01:0.1,
    xminorticks = IntervalsBetween(2),
    )

    goal_mean_errors = [mean([norm(ground_truth_goals - recovered_parameters[i][j][goal_indices]) for j in 1:num_trials]) for i in noise_levels]
    goal_stds = [std([norm(ground_truth_goals - recovered_parameters[i][j][goal_indices]) for j in 1:num_trials]) for i in noise_levels]
    baseline_goal_mean_errors = [mean([norm(ground_truth_goals - baseline_recovered_parameters[i][j][goal_indices]) for j in 1:num_trials]) for i in noise_levels]
    baseline_goal_stds = [std([norm(ground_truth_goals - baseline_recovered_parameters[i][j][goal_indices]) for j in 1:num_trials]) for i in noise_levels]
    our_method = CairoMakie.scatter!(ax3, noise_levels .+ overlap_shift, goal_mean_errors, color = (:blue, 0.75))
    CairoMakie.errorbars!(ax3, noise_levels .+ overlap_shift, goal_mean_errors, goal_stds, color = (:blue, 0.75))
    baseline = CairoMakie.scatter!(ax3, noise_levels, baseline_goal_mean_errors, color = (:red, 0.75))
    CairoMakie.errorbars!(ax3, noise_levels, baseline_goal_mean_errors, baseline_goal_stds, color = (:red, 0.75))

    CairoMakie.axislegend(ax3, [our_method, baseline], ["Our Method", "Baseline"], position = :lt)
    CairoMakie.save(directory*"GoalError.png", fig3)

    fig4 = CairoMakie.Figure()
    ax4 = CairoMakie.Axis(fig4[1, 1],
    xlabel = "Max Absolute Postion Observation Error",
    ylabel = "Mean Absolute Goal Prediction Error",
    xticks = 0.0:0.01:0.1,
    xminorticks = IntervalsBetween(2),
    )

    goal_p1_mean_errors = [mean([norm(ground_truth_goals[1:2] - recovered_parameters[i][j][1:2]) for j in 1:num_trials]) for i in noise_levels]
    goal_p1_stds = [std([norm(ground_truth_goals[1:2] - recovered_parameters[i][j][1:2]) for j in 1:num_trials]) for i in noise_levels]
    baseline_goal_p1_mean_errors = [mean([norm(ground_truth_goals[1:2] - baseline_recovered_parameters[i][j][1:2]) for j in 1:num_trials]) for i in noise_levels]
    baseline_goal_p1_stds = [std([norm(ground_truth_goals[1:2] - baseline_recovered_parameters[i][j][1:2]) for j in 1:num_trials]) for i in noise_levels]
    goal_p2_mean_errors = [mean([norm(ground_truth_goals[3:4] - recovered_parameters[i][j][3:4]) for j in 1:num_trials]) for i in noise_levels]
    goal_p2_stds = [std([norm(ground_truth_goals[3:4] - recovered_parameters[i][j][3:4]) for j in 1:num_trials]) for i in noise_levels]
    baseline_goal_p2_mean_errors = [mean([norm(ground_truth_goals[3:4] - baseline_recovered_parameters[i][j][3:4]) for j in 1:num_trials]) for i in noise_levels]
    baseline_goal_p2_stds = [std([norm(ground_truth_goals[3:4] - baseline_recovered_parameters[i][j][3:4]) for j in 1:num_trials]) for i in noise_levels]

    our_p1_method = CairoMakie.scatter!(ax4, noise_levels .+ overlap_shift, goal_p1_mean_errors, color = (:blue, 0.75))
    CairoMakie.errorbars!(ax4, noise_levels .+ overlap_shift, goal_p1_mean_errors, goal_p1_stds, color = (:blue, 0.75))
    baseline_p1 = CairoMakie.scatter!(ax4, noise_levels, baseline_goal_p1_mean_errors, color = (:red, 0.75))
    CairoMakie.errorbars!(ax4, noise_levels, baseline_goal_p1_mean_errors, baseline_goal_p1_stds, color = (:red, 0.75))

    our_p2_method = CairoMakie.scatter!(ax4, noise_levels .+ overlap_shift, goal_p2_mean_errors, color = (:purple, 0.75))
    CairoMakie.errorbars!(ax4, noise_levels .+ overlap_shift, goal_p2_mean_errors, goal_p2_stds, color = (:purple, 0.75))
    baseline_p2 = CairoMakie.scatter!(ax4, noise_levels, baseline_goal_p2_mean_errors, color = (:orange, 0.75))
    CairoMakie.errorbars!(ax4, noise_levels, baseline_goal_p2_mean_errors, baseline_goal_p2_stds, color = (:orange, 0.75))

    CairoMakie.axislegend(ax4, [our_p1_method, our_p2_method, baseline_p1, baseline_p2], ["Our Method Player 1", "Our Method Player 2", "Baseline Player 1", "Baseline Player 2"], position = :lc)
    CairoMakie.save(directory*"GoalErrorPlayer.png", fig4)


    fig5 = CairoMakie.Figure()
    ax5 = CairoMakie.Axis(fig5[1, 1],
    xlabel = "Max Absolute Postion Observation Error",
    ylabel = "Mean Absolute Discount Factor Prediction Error",
    xticks = 0.0:0.01:0.1,
    xminorticks = IntervalsBetween(2),
    )

    discount_indices = [3, 6]
    discount_errors = [mean([norm(ground_truth_parameters[discount_indices] - recovered_parameters[i][j][discount_indices]) for j in 1:num_trials]) for i in noise_levels]
    discount_stds = [std([norm(ground_truth_parameters[discount_indices] - recovered_parameters[i][j][discount_indices]) for j in 1:num_trials]) for i in noise_levels]
    our_method = CairoMakie.scatter!(ax5, noise_levels, discount_errors, color = (:blue, 0.75))
    CairoMakie.errorbars!(ax5, noise_levels, discount_errors, discount_stds, color = (:blue, 0.75))
    mean_discounts = [mean(recovered_parameters[i][j][discount_indices]) for i in noise_levels, j in 1:num_trials]
    discount_std = [std(recovered_parameters[i][j][discount_indices]) for i in noise_levels, j in 1:num_trials]
    baseline_discount_errors = [mean([norm(ground_truth_parameters[discount_indices] - baseline_recovered_parameters[i][j][discount_indices]) for j in 1:num_trials]) for i in noise_levels]
    baseline_discount_stds = [std([norm(ground_truth_parameters[discount_indices] - baseline_recovered_parameters[i][j][discount_indices]) for j in 1:num_trials]) for i in noise_levels]
    CairoMakie.save(directory*"DiscountError.png", fig5)
    
end