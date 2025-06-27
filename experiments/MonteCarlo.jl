module MonteCarloSim

using Random
using BlockArrays: Block, mortar, BlockVector, blocksizes
using LinearAlgebra: norm_sqr
using TrajectoryGamesBase: num_players, state_dim
using TrajectoryGamesExamples: BicycleDynamics
using Statistics: mean, std
using HypothesisTests: OneSampleTTest, pvalue
using Infiltrator
include("../src/InverseGameDiscountFactor.jl")
include("GameUtils.jl")
include("graphing/ExperimentGraphingUtils.jl")
include("In-D/InD.jl")

function run_monte_carlo(;
    frames = [26158, 26320],
    tracks = [201, 205, 207, 208],
    downsample_rate = 6,
    rng = Random.MersenneTwister(1),
    num_trials = 30,
    σs = [0.02*i for i in 0:25],
    verbose = true,
    store_all = false,
    full_state = true,
    baseline_filename="mc_study_baseline_traj_errors_0.02.txt",
    new_filename="mc_study_new_traj_errors_0.02.txt"
)
    # Set randomness
    Random.seed!(rng)

    # Get real trajectory data
    InD_observations = GameUtils.pull_trajectory("07";
        track = tracks, 
        downsample_rate = downsample_rate, 
        all = false, 
        frames = frames
    )
    init_state = InD_observations[1]
    InD_observations = full_state ? 
        InD_observations :
        [BlockVector(
            mapreduce(x -> x[1:2], vcat, observation.blocks),
            [2 for _ in 1:length(tracks)]) 
        for observation in InD_observations]

    trk_201_lane_center(x) = 0.0  # Placeholder
    trk_205_lane_center(x) = 0.0  
    trk_207_lane_center(x) = -6.535465682649165e-04*x^6 + 
                            -0.069559792458210*x^5 + 
                            -3.033950160533982*x^4 + 
                            -69.369975733866840*x^3 + 
                            -8.760325006936075e+02*x^2 + 
                            -5.782944928944775e+03*x + 
                            -1.547509969706588e+04
    trk_208_lane_center(x) = 8.304049624037807*x + 1.866183521575921e+02
    lane_centers = [trk_201_lane_center, trk_205_lane_center, trk_207_lane_center, trk_208_lane_center]
    
    dynamics = BicycleDynamics(;
        dt = 0.04*downsample_rate,
        l = 1.0,
        state_bounds = (; lb = [-Inf, -Inf, -Inf, -Inf], ub = [Inf, Inf, Inf, Inf]),
        control_bounds = (; lb = [-5, -pi/4], ub = [5, pi/4]),
        integration_scheme = :forward_euler
    )

    # Initialize game with full state observation
    init = GameUtils.init_bicycle_test_game(
        full_state;
        initial_state = init_state,
        game_params = mortar([
            [[InD_observations[end][Block(i)][1:2]..., 1.0, 1.0, 1.0, 1.0, 1.0, 10.0] for i in 1:length(tracks)]...]),
        horizon = length(frames[1]:downsample_rate:frames[2]),
        n = length(tracks),
        dt = 0.04*downsample_rate,
        myopic=true,
        verbose = false,
        dynamics = dynamics,
        lane_centers = lane_centers
    )

    init_baseline = GameUtils.init_bicycle_test_game(
        full_state;
        initial_state = init.initial_state,
        game_params = mortar([
            [[InD_observations[end][Block(i)][1:2]..., 1.0, 1.0, 1.0, 1.0, 10.0] for i in 1:length(tracks)]...]),
        horizon = init.horizon,
        n = length(tracks),
        dt = 0.04*downsample_rate,
        myopic=false,
        verbose = false,
        dynamics = dynamics,
        lane_centers = lane_centers
    )
    # Create MCP game solvers
    mcp_solver = InverseGameDiscountFactor.MCPCoupledOptimizationSolver(
        init.game_structure.game,
        init.horizon,
        blocksizes(init.game_parameters, 1)
    )

    baseline_solver = InverseGameDiscountFactor.MCPCoupledOptimizationSolver(
        init_baseline.game_structure.game,
        init_baseline.horizon,
        [7 for _ in 1:length(tracks)]
    )
    errors = Array{Float64}(undef, num_trials)
    baseline_errors = Array{Float64}(undef, num_trials)
    
    # observed_trajectories = []
    # recovered_trajectories = []
    # recovered_params = []

    for σ_idx in eachindex(σs)
        σ = σs[σ_idx]
        errors .= -1.0
        baseline_errors .= -1.0
        for trial_idx in 1:num_trials
            verbose && println("std: ", σ, " trial: ", trial_idx)

            # Add noise to real observations
            noisy_observations = map(InD_observations) do obs
                obs .+ σ * randn(size(obs))
            end

            # Solve inverse game with both methods
            method_sol = InverseGameDiscountFactor.solve_myopic_inverse_game(
                mcp_solver.mcp_game,
                noisy_observations,
                init.observation_model,
                blocksizes(init.game_parameters, 1);
                initial_state = init.initial_state,
                hidden_state_guess = init.game_parameters,
                max_grad_steps = 200,
                verbose = false,
                dynamics = dynamics,
            )

            baseline_sol = InverseGameDiscountFactor.solve_myopic_inverse_game(
                baseline_solver.mcp_game,
                noisy_observations,
                init_baseline.observation_model,
                blocksizes(init_baseline.game_parameters, 1);
                initial_state = init_baseline.initial_state,
                hidden_state_guess = init_baseline.game_parameters,
                max_grad_steps = 200,
                verbose = false,
                dynamics = dynamics,
            )

            # Record errors for both methods
            errors[trial_idx] = norm_sqr(vcat(InD_observations...) - method_sol.recovered_trajectory) / length(InD_observations)
            baseline_errors[trial_idx] = norm_sqr(vcat(InD_observations...) - baseline_sol.recovered_trajectory) / length(InD_observations)
            
        end
        open("experiments/In-D/"*new_filename, "a") do f
            write(f, "m"*string(σ)*" "*join(string.(round.(errors, digits=4)), " ")*"\n")
        end
        open("experiments/In-D/"*baseline_filename, "a") do f
            write(f, "b"*string(σ)*" "*join(string.(round.(baseline_errors, digits=4)), " ")*"\n")
        end
    end
    println("mc study done")
    
    # Generate summary statistics
    generate_monte_carlo_summary(σs, num_trials, new_filename, baseline_filename, "mc_study_summary_0.02.txt")
end

function run_monte_carlo_crosswalk(;
    rng = Random.MersenneTwister(1),
    num_trials = 40,
    noise_level_cap = 0.1,
    noise_resolution = 20,
    verbose = true,
    store_all = false,
    reset = false
)
    σs = [(noise_level_cap / noise_resolution)*i for i in 0:noise_resolution]
    Random.seed!(rng)

    coeffs = [1.0, 0.1, 5.0]
    horizon = 25
    initial_state = mortar([
        [2.0, 2.0, 0, 0],
        [0, 2.0, 0, 0]
    ])
    game_params = mortar([
        [0, 0],
        [2, 0]
    ])

    fo_init = GameUtils.init_crosswalk_game(
        true;
        myopic = true,
        initial_state = initial_state,
        game_params = BlockVector(
                    vcat([vcat(param..., [0.6]) for (i, param) in enumerate(game_params.blocks)]...),
                    [3 for _ in 1:length(game_params.blocks)]),
        coeffs = coeffs,
        horizon = horizon
    )

    fo_init_baseline = GameUtils.init_crosswalk_game(
        true;
        myopic = false,
        initial_state = fo_init.initial_state,
        game_params = game_params,
        coeffs = coeffs,
        horizon = horizon
    )

    po_init = GameUtils.init_crosswalk_game(
        false;
        myopic = true,
        initial_state = initial_state,
        game_params = BlockVector(
                    vcat([vcat(param..., [0.6]) for (i, param) in enumerate(game_params.blocks)]...),
                    [3 for _ in 1:length(game_params.blocks)]),
        coeffs = coeffs,
        horizon = horizon
    )

    po_init_baseline = GameUtils.init_crosswalk_game(
        false;
        myopic = false,
        initial_state = po_init.initial_state,
        game_params = game_params,
        coeffs = coeffs,
        horizon = horizon
    )

    fo_mcp_game = InverseGameDiscountFactor.MCPCoupledOptimizationSolver(
        fo_init.game_structure.game,
        fo_init.horizon,
        blocksizes(fo_init.game_parameters, 1)
    ).mcp_game

    fo_baseline_solver = InverseGameDiscountFactor.MCPCoupledOptimizationSolver(
        fo_init_baseline.game_structure.game,
        fo_init_baseline.horizon,
        blocksizes(fo_init_baseline.game_parameters, 1)
    ).mcp_game

    po_mcp_game = InverseGameDiscountFactor.MCPCoupledOptimizationSolver(
        po_init.game_structure.game,
        po_init.horizon,
        blocksizes(po_init.game_parameters, 1)
    ).mcp_game

    po_baseline_solver = InverseGameDiscountFactor.MCPCoupledOptimizationSolver(
        po_init_baseline.game_structure.game,
        po_init_baseline.horizon,
        blocksizes(po_init_baseline.game_parameters, 1)
    ).mcp_game

    forward_solution = InverseGameDiscountFactor.reconstruct_solution(
        InverseGameDiscountFactor.solve_mcp_game(
            fo_mcp_game,
            fo_init.initial_state,
            mortar([
                [0.0, 0.0, 0.6],
                [2.0, 0.0, 0.6]
            ]);
            verbose = false
        ),
        fo_init.game_structure.game,
        fo_init.horizon
    )
    ExperimentGraphingUtils.graph_crosswalk_trajectories(
        "forward_solution",
        [forward_solution],
        fo_init.game_structure,
        fo_init.horizon;
        colors = [[(:red, 1.0), (:blue, 1.0), (:green, 1.0)], [(:red, 0.25), (:blue, 0.25), (:green, 0.25)]],
    )

    fo_errors = Array{Float64}(undef, num_trials)
    fo_baseline_errors = Array{Float64}(undef, num_trials)
    po_errors = Array{Float64}(undef, num_trials)
    po_baseline_errors = Array{Float64}(undef, num_trials)

    if reset 
        open("experiments/crosswalk/fo_ours.txt", "w") do f
            write(f, "")
        end
        open("experiments/crosswalk/fo_baseline.txt", "w") do f
            write(f, "")
        end
        open("experiments/crosswalk/po_ours.txt", "w") do f
            write(f, "")
        end
        open("experiments/crosswalk/po_baseline.txt", "w") do f
            write(f, "")
        end
    end
    f = open("experiments/crosswalk/exp_statistics.txt", "a")

    for σ_idx in eachindex(σs)
        σ = σs[σ_idx]
        fo_errors .= -1.0
        fo_baseline_errors .= -1.0
        po_errors .= -1.0
        po_baseline_errors .= -1.0
        
        for trial_idx in 1:num_trials
            verbose && println("std: ", σ, " trial: ", trial_idx)

            noisy_observations = map(forward_solution.blocks) do block
                block .+ σ * randn(size(block))
            end

            po_noisy_observations = map(forward_solution.blocks) do block
                vcat((block .+ σ * randn(size(block)))[1:2], (block .+ σ * randn(size(block)))[5:6])
            end
            println("σ: ", σ, " fo_new:")
            fo_method_sol = InverseGameDiscountFactor.solve_myopic_inverse_game(
                fo_mcp_game,
                noisy_observations,
                fo_init.observation_model,
                blocksizes(fo_init.game_parameters, 1);
                initial_state = fo_init.initial_state,
                hidden_state_guess = fo_init.game_parameters,
                max_grad_steps = 200,
                retries_on_divergence = 3,
                verbose = false,
                use_warm_start = false,
                lr = 1e-3,
                regularization = 1.0
            )
            write(f, "fo_new: "*string(σ)*" "*string(count(x -> x == InverseGameDiscountFactor.PATHSolver.MCP_Solved, fo_method_sol.solving_status)/length(fo_method_sol.solving_status))*"\n")
            println("σ: ", σ, " fo_baseline:")
            fo_baseline_sol = InverseGameDiscountFactor.solve_myopic_inverse_game(
                fo_baseline_solver,
                noisy_observations,
                fo_init_baseline.observation_model,
                blocksizes(fo_init_baseline.game_parameters, 1);
                initial_state = fo_init_baseline.initial_state,
                hidden_state_guess = fo_init_baseline.game_parameters,
                max_grad_steps = 200,
                verbose = false,
                use_warm_start = false,
                lr = 1e-3,
                regularization = 1.0
            )
            write(f, "fo_baseline: "*string(σ)*" "*string(count(x -> x == InverseGameDiscountFactor.PATHSolver.MCP_Solved, fo_baseline_sol.solving_status)/length(fo_baseline_sol.solving_status))*"\n")
            println("σ: ", σ, " po_new:")
            po_method_sol = InverseGameDiscountFactor.solve_myopic_inverse_game(
                po_mcp_game,
                po_noisy_observations,
                po_init.observation_model,
                blocksizes(po_init.game_parameters, 1);
                initial_state = po_init.initial_state,
                hidden_state_guess = po_init.game_parameters,
                max_grad_steps = 200,
                verbose = false,
                use_warm_start = false,
                lr = 1e-3,
                regularization = 1.0
            )
            write(f, "po_new: "*string(σ)*" "*string(count(x -> x == InverseGameDiscountFactor.PATHSolver.MCP_Solved, po_method_sol.solving_status)/length(po_method_sol.solving_status))*"\n")
            println("σ: ", σ, " po_baseline:")
            po_baseline_sol = InverseGameDiscountFactor.solve_myopic_inverse_game(
                po_baseline_solver,
                po_noisy_observations,
                po_init_baseline.observation_model,
                blocksizes(po_init_baseline.game_parameters, 1);
                initial_state = po_init_baseline.initial_state,
                hidden_state_guess = po_init_baseline.game_parameters,
                max_grad_steps = 200,
                verbose = false,
                use_warm_start = false,
                lr = 1e-3,
                regularization = 1.0
            )
            write(f, "po_baseline: "*string(σ)*" "*string(count(x -> x == InverseGameDiscountFactor.PATHSolver.MCP_Solved, po_baseline_sol.solving_status)/length(po_baseline_sol.solving_status))*"\n")

            fo_errors[trial_idx] = norm_sqr(forward_solution - fo_method_sol.recovered_trajectory) / length(forward_solution.blocks)
            fo_baseline_errors[trial_idx] = norm_sqr(forward_solution - fo_baseline_sol.recovered_trajectory) / length(forward_solution.blocks)
            po_errors[trial_idx] = norm_sqr(forward_solution - po_method_sol.recovered_trajectory) / length(forward_solution.blocks)
            po_baseline_errors[trial_idx] = norm_sqr(forward_solution - po_baseline_sol.recovered_trajectory) / length(forward_solution.blocks)
        end
        open("experiments/crosswalk/fo_ours.txt", "a") do f
            write(f, "m"*string(σ)*" "*join(string.(round.(fo_errors, digits=4)), " ")*"\n")
        end
        open("experiments/crosswalk/fo_baseline.txt", "a") do f
            write(f, "b"*string(σ)*" "*join(string.(round.(fo_baseline_errors, digits=4)), " ")*"\n")
        end
        open("experiments/crosswalk/po_ours.txt", "a") do f
            write(f, "m"*string(σ)*" "*join(string.(round.(po_errors, digits=4)), " ")*"\n")
        end
        open("experiments/crosswalk/po_baseline.txt", "a") do f
            write(f, "b"*string(σ)*" "*join(string.(round.(po_baseline_errors, digits=4)), " ")*"\n")
        end
    end
    close(f)
    ExperimentGraphingUtils.process_and_graph_crosswalk_results()
end

"""
    generate_monte_carlo_summary(σs, num_trials)

Generate summary statistics from Monte Carlo error files.
Reads the error files and computes statistical metrics including:
- Average improvement
- Percentage improvement
- Standard deviations
- Statistical significance tests

Results are written to a summary file.
"""
function generate_monte_carlo_summary(σs, num_trials, new_filename, baseline_filename, summary_filename)
    # Create or clear the summary file
    open("experiments/In-D/"*summary_filename, "w") do f
        write(f, "# Monte Carlo Study Summary\n\n")
    end
    
    for σ in σs
        # Read the error data from files
        new_errors = Float64[]
        baseline_errors = Float64[]
        
        # Read new method errors
        open("experiments/In-D/"*new_filename, "r") do f
            for line in eachline(f)
                if startswith(line, "m"*string(σ)*" ")
                    error_strings = split(line[length("m"*string(σ)*" "):end])
                    append!(new_errors, parse.(Float64, error_strings))
                end
            end
        end
        
        # Read baseline errors
        open("experiments/In-D/"*baseline_filename, "r") do f
            for line in eachline(f)
                if startswith(line, "b"*string(σ)*" ")
                    error_strings = split(line[length("b"*string(σ)*" "):end])
                    append!(baseline_errors, parse.(Float64, error_strings))
                end
            end
        end
        
        # Check for invalid error values (-1.0)
        invalid_new = count(x -> x == -1.0, new_errors)
        invalid_baseline = count(x -> x == -1.0, baseline_errors)
        
        if invalid_new > 0
            @warn "Found $invalid_new invalid values (-1.0) in new method errors for σ=$σ"
        end
        
        if invalid_baseline > 0
            @warn "Found $invalid_baseline invalid values (-1.0) in baseline errors for σ=$σ"
        end
        
        # Filter out invalid values for statistics calculation
        filtered_new_errors = filter(x -> x != -1.0, new_errors)
        filtered_baseline_errors = filter(x -> x != -1.0, baseline_errors)
        
        # Ensure we have enough valid data points
        if length(filtered_new_errors) == 0 || length(filtered_baseline_errors) == 0
            @error "No valid error data for σ=$σ after filtering invalid values"
            continue
        end
        
        # Also check if we have the right number of data points
        if length(new_errors) != num_trials || length(baseline_errors) != num_trials
            @warn "Expected $num_trials trials but found $(length(new_errors)) new method errors and $(length(baseline_errors)) baseline errors for σ=$σ"
        end
        
        # Calculate statistics using filtered data
        open("experiments/In-D/"*summary_filename, "a") do f
            write(f, "## Noise level σ = $σ\n\n")
            
            if invalid_new > 0 || invalid_baseline > 0
                write(f, "⚠️ Warning: Found $(invalid_new + invalid_baseline) invalid data points (-1.0) that were excluded from analysis\n\n")
            end
            
            average_improvement = mean(filtered_baseline_errors) - mean(filtered_new_errors)
            write(f, "average_improvement @ $σ $(round(average_improvement, digits=4))\n")

            average_percentage_improvement = average_improvement / mean(filtered_baseline_errors) * 100
            write(f, "average_percentage_improvement @ $σ $(round(average_percentage_improvement, digits=4))\n")
            
            baseline_std = std(filtered_baseline_errors)
            new_std = std(filtered_new_errors)
            write(f, "baseline_std @ $σ $(round(baseline_std, digits=4))\n")
            write(f, "new_std @ $σ $(round(new_std, digits=4))\n")
            
            # Perform paired t-test to check statistical significance only if we have paired data
            valid_pairs = [(b, n) for (b, n) in zip(baseline_errors, new_errors) if b != -1.0 && n != -1.0]
            
            if length(valid_pairs) > 1
                # For paired t-test, we test if the difference (baseline - new) is significantly > 0
                error_differences = [pair[1] - pair[2] for pair in valid_pairs]
                t_test = OneSampleTTest(error_differences)
                p_val = pvalue(t_test, tail=:right)  # right-tailed test for baseline > new
                
                write(f, "p_value @ $σ $(round(p_val, digits=6))\n")
                write(f, "significant @ $σ $(p_val < 0.05)\n")
                
                # Calculate likelihood as a percentage
                significance_likelihood = (1 - p_val) * 100
                write(f, "likelihood_of_improvement @ $σ $(round(significance_likelihood, digits=2))%\n\n")
            else
                write(f, "p_value @ $σ N/A (insufficient valid paired data for statistical testing)\n\n")
            end
        end
    end
    println("Summary statistics generated")
end

end