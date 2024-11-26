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
    # BicycleDynamics
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
    # CairoMakie.Legend(fig4[1, 2], [our_p1_method, our_p2_method, baseline_p1, baseline_p2], ["Our Method Player 1", "Our Method Player 2", "Baseline Player 1", "Baseline Player 2"])
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

    # environment = PolygonEnvironment(6, 8)
    # game = n_player_collision_avoidance(2; environment, min_distance = 0.5, collision_avoidance_coefficient = 5.0)
    # baseline_game = n_player_collision_avoidance(2; environment, min_distance = 0.5, collision_avoidance_coefficient = 5.0, myopic = false)
    # horizon = 25
    # solver = MCPCoupledOptimizationSolver(game.game, horizon, blocksizes(hidden_params, 1))
    # baseline_solver = MCPCoupledOptimizationSolver(baseline_game.game, horizon, [2, 2])
    # mcp_game = solver.mcp_game
    # baseline_mcp_game = baseline_solver.mcp_game


    # forward_solution = solve_mcp_game(mcp_game, initial_state, hidden_params; verbose = false)
    
end