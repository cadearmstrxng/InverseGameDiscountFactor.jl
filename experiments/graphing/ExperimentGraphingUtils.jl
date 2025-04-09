module ExperimentGraphingUtils

using Statistics:
    mean, std
using CairoMakie
using TrajectoryGamesBase:
    num_players, state_dim
using BlockArrays:
    Block, blocks
using ImageTransformations
using Rotations
using OffsetArrays:Origin
using CSV
using DataFrames
using LinearAlgebra: norm

using Infiltrator

include("../../src/utils/Utils.jl")

# Export functions to make them available to users of this module
export graph_metrics, graph_trajectories, graph_crosswalk_trajectories
export parse_crosswalk_results, process_and_graph_crosswalk_results

function graph_metrics(
    baseline_file::String,
    our_method_file::String;
    observation_mode="both",
    pre_prefix = "./experiments/In-D/",
    partial_state_included=true,
    std_dev_threshold=2.5,
    show_outliers=false,
    show_data_points=true
)
    prefix = pre_prefix
    CairoMakie.activate!()

    data_point_opacity = 0.3
    outlier_point_opacity = data_point_opacity / 3
    band_opacity = 0.35

    # Read the CSV files with explicit header=false to ensure we get all rows
    baseline_data = CSV.read(baseline_file, DataFrame, header=false)
    our_method_data = CSV.read(our_method_file, DataFrame, header=false)

    # Calculate the number of noise levels by assuming equal division between full and partial state
    num_noise_levels = partial_state_included ? div(nrow(baseline_data), 2) : nrow(baseline_data)
    
    # Split the dataframes into full state and partial state parts
    baseline_full_data = baseline_data[1:num_noise_levels, :]
    baseline_partial_data = partial_state_included ? baseline_data[(num_noise_levels+1):end, :] : nothing
    
    our_method_full_data = our_method_data[1:num_noise_levels, :]
    our_method_partial_data = partial_state_included ? our_method_data[(num_noise_levels+1):end, :] : nothing
    
    σs = [parse(Float64, replace(String(level), r"^[bm]" => "")) for level in baseline_full_data[:, 1]]

    # Debug print to check the data
    println("Number of noise levels detected: ", num_noise_levels)
    println("Noise levels: ", σs)
    println("Number of trials: ", length(baseline_full_data[1, 2:end]))
    
    # Extract all individual data points for analysis and filtering
    baseline_full_individual_points = Matrix(baseline_full_data[:, 2:end])
    our_method_full_individual_points = Matrix(our_method_full_data[:, 2:end])
    
    baseline_partial_individual_points = partial_state_included ? Matrix(baseline_partial_data[:, 2:end]) : nothing
    our_method_partial_individual_points = partial_state_included ? Matrix(our_method_partial_data[:, 2:end]) : nothing
    
    # Calculate filtered means and standard deviations
    baseline_full_means, baseline_full_stds, baseline_full_outlier_masks = 
        filter_outliers_and_compute_stats(baseline_full_individual_points; std_dev_threshold=std_dev_threshold)
    
    our_method_full_means, our_method_full_stds, our_method_full_outlier_masks = 
        filter_outliers_and_compute_stats(our_method_full_individual_points; std_dev_threshold=std_dev_threshold)
    
    # Process partial state data if included
    if partial_state_included
        baseline_partial_means, baseline_partial_stds, baseline_partial_outlier_masks = 
            filter_outliers_and_compute_stats(baseline_partial_individual_points; std_dev_threshold=std_dev_threshold)
        
        our_method_partial_means, our_method_partial_stds, our_method_partial_outlier_masks = 
            filter_outliers_and_compute_stats(our_method_partial_individual_points; std_dev_threshold=std_dev_threshold)
    else
        baseline_partial_means, baseline_partial_stds, baseline_partial_outlier_masks = nothing, nothing, nothing
        our_method_partial_means, our_method_partial_stds, our_method_partial_outlier_masks = nothing, nothing, nothing
    end
    
    # Print information about outliers
    for noise_idx in eachindex(σs)
        total_outliers = 0
        if !isnothing(baseline_full_outlier_masks)
            n_outliers = sum(baseline_full_outlier_masks[noise_idx])
            total_outliers += n_outliers
            (n_outliers > 0) && println("Noise level $(σs[noise_idx]): Baseline (Full State) - $n_outliers outliers removed")
        end
        
        if !isnothing(our_method_full_outlier_masks)
            n_outliers = sum(our_method_full_outlier_masks[noise_idx])
            total_outliers += n_outliers
            (n_outliers > 0) && println("Noise level $(σs[noise_idx]): Our Method (Full State) - $n_outliers outliers removed")
        end
        
        if partial_state_included && !isnothing(baseline_partial_outlier_masks)
            n_outliers = sum(baseline_partial_outlier_masks[noise_idx])
            total_outliers += n_outliers
            (n_outliers > 0) && println("Noise level $(σs[noise_idx]): Baseline (Partial State) - $n_outliers outliers removed")
        end
        
        if partial_state_included && !isnothing(our_method_partial_outlier_masks)
            n_outliers = sum(our_method_partial_outlier_masks[noise_idx])
            total_outliers += n_outliers
            (n_outliers > 0) && println("Noise level $(σs[noise_idx]): Our Method (Partial State) - $n_outliers outliers removed")
        end
        
        println("Noise level $(σs[noise_idx]): Total $total_outliers outliers removed")
    end
    
    # Create figure for all methods together (original plot)
    fig_all = Figure()
    ax_all = Axis(fig_all[1, 1],
        xlabel = "Noise Level (σ)",
        ylabel = "Trajectory Error",
        title = "Trajectory Error vs Noise Level (All Methods)"
    )

    # Plot baseline with non-negative lower bound (full state)
    lines!(ax_all, σs, baseline_full_means, color=:red, label="Baseline (Full State)")
    band!(ax_all, σs, 
        max.(baseline_full_means - baseline_full_stds, 0), 
        baseline_full_means + baseline_full_stds, 
        color=(:red, band_opacity))

    # Scatter plot for individual baseline full state data points
    show_data_points && for col in eachindex(baseline_full_individual_points[1, :])
        for row in eachindex(σs)
            # Determine if this point is an outlier
            is_outlier = baseline_full_outlier_masks[row][col]
            is_outlier && !show_outliers && continue
            point_opacity = is_outlier ? outlier_point_opacity : data_point_opacity
            
            scatter!(ax_all, [σs[row]], [baseline_full_individual_points[row, col]], 
                    color=(:red, point_opacity), markersize=4)
        end
    end

    # Plot our method with non-negative lower bound (full state)
    lines!(ax_all, σs, our_method_full_means, color=:blue, label="Our Method (Full State)")
    band!(ax_all, σs, 
        max.(our_method_full_means - our_method_full_stds, 0), 
        our_method_full_means + our_method_full_stds, 
        color=(:blue, band_opacity))
        
    # Scatter plot for individual our method full state data points
    show_data_points && for col in eachindex(our_method_full_individual_points[1, :])
        for row in eachindex(σs)
            # Determine if this point is an outlier
            is_outlier = our_method_full_outlier_masks[row][col]
            is_outlier && !show_outliers && continue
            point_opacity = is_outlier ? outlier_point_opacity : data_point_opacity
            
            scatter!(ax_all, [σs[row]], [our_method_full_individual_points[row, col]], 
                    color=(:blue, point_opacity), markersize=4)
        end
    end

    # Plot baseline with non-negative lower bound (partial state)
    partial_state_included && lines!(ax_all, σs, baseline_partial_means, color=:green, linestyle=:dash, label="Baseline (Partial State)")
    partial_state_included && band!(ax_all, σs, 
        max.(baseline_partial_means - baseline_partial_stds, 0), 
        baseline_partial_means + baseline_partial_stds, 
        color=(:green, band_opacity))
        
    # Scatter plot for individual baseline partial state data points
    if partial_state_included && show_data_points
        for col in eachindex(baseline_partial_individual_points[1, :])
            for row in eachindex(σs)
                # Determine if this point is an outlier
                is_outlier = baseline_partial_outlier_masks[row][col]
                is_outlier && !show_outliers && continue
                point_opacity = is_outlier ? outlier_point_opacity : data_point_opacity
                
                scatter!(ax_all, [σs[row]], [baseline_partial_individual_points[row, col]], 
                        color=(:green, point_opacity), markersize=4)
            end
        end
    end
    
    # Plot our method with non-negative lower bound (partial state)
    partial_state_included && lines!(ax_all, σs, our_method_partial_means, color=:brown, linestyle=:dash, label="Our Method (Partial State)")
    partial_state_included && band!(ax_all, σs, 
        max.(our_method_partial_means - our_method_partial_stds, 0), 
        our_method_partial_means + our_method_partial_stds, 
        color=(:brown, band_opacity))
        
    # Scatter plot for individual our method partial state data points
    if partial_state_included && show_data_points
        for col in eachindex(our_method_partial_individual_points[1, :])
            for row in eachindex(σs)
                # Determine if this point is an outlier
                is_outlier = our_method_partial_outlier_masks[row][col]
                is_outlier && !show_outliers && continue
                point_opacity = is_outlier ? outlier_point_opacity : data_point_opacity
                
                scatter!(ax_all, [σs[row]], [our_method_partial_individual_points[row, col]], 
                        color=(:brown, point_opacity), markersize=4)
            end
        end
    end

    # Add legend
    axislegend(ax_all, position=:lt)

    # Save the all methods figure
    save(prefix*"/"*observation_mode*"_AllMethods_TrajectoryErrorGraph.png", fig_all)
    
    # Create figure for baseline comparison
    fig_baseline = Figure()
    ax_baseline = Axis(fig_baseline[1, 1],
        xlabel = "Noise Level (σ)",
        ylabel = "Trajectory Error",
        title = "Baseline: Full State vs Partial State"
    )
    
    # Plot baseline full state
    lines!(ax_baseline, σs, baseline_full_means, color=:red, label="Full State")
    band!(ax_baseline, σs, 
        max.(baseline_full_means - baseline_full_stds, 0), 
        baseline_full_means + baseline_full_stds, 
        color=(:red, band_opacity))
        
    # Scatter plot for individual baseline full state data points
    show_data_points && for col in eachindex(baseline_full_individual_points[1, :])
        for row in eachindex(σs)
            # Determine if this point is an outlier
            is_outlier = baseline_full_outlier_masks[row][col]
            is_outlier && !show_outliers && continue
            point_opacity = is_outlier ? outlier_point_opacity : data_point_opacity
            scatter!(ax_baseline, [σs[row]], [baseline_full_individual_points[row, col]], 
                    color=(:red, point_opacity), markersize=4)
        end
    end
    
    # Plot baseline partial state
    partial_state_included && lines!(ax_baseline, σs, baseline_partial_means, color=:green, linestyle=:dash, label="Partial State")
    partial_state_included && band!(ax_baseline, σs, 
        max.(baseline_partial_means - baseline_partial_stds, 0), 
        baseline_partial_means + baseline_partial_stds, 
        color=(:green, band_opacity))
        
    # Scatter plot for individual baseline partial state data points
    if partial_state_included
        for col in eachindex(baseline_partial_individual_points[1, :])
            for row in eachindex(σs)
                # Determine if this point is an outlier
                is_outlier = baseline_partial_outlier_masks[row][col]
                is_outlier && !show_outliers && continue
                point_opacity = is_outlier ? outlier_point_opacity : data_point_opacity
                
                scatter!(ax_baseline, [σs[row]], [baseline_partial_individual_points[row, col]], 
                        color=(:green, point_opacity), markersize=4)
            end
        end
    end
    
    axislegend(ax_baseline, position=:lt)
    
    # Save the baseline comparison figure
    save(prefix*"/"*observation_mode*"_Baseline_Comparison.png", fig_baseline)
    
    # Create figure for our method comparison
    fig_our = Figure()
    ax_our = Axis(fig_our[1, 1],
        xlabel = "Noise Level (σ)",
        ylabel = "Trajectory Error",
        title = "Our Method: Full State vs Partial State"
    )
    
    # Plot our method full state
    lines!(ax_our, σs, our_method_full_means, color=:blue, label="Full State")
    band!(ax_our, σs, 
        max.(our_method_full_means - our_method_full_stds, 0), 
        our_method_full_means + our_method_full_stds, 
        color=(:blue, band_opacity))
        
    # Scatter plot for individual our method full state data points
    show_data_points && for col in eachindex(our_method_full_individual_points[1, :])
        for row in eachindex(σs)
            # Determine if this point is an outlier
            is_outlier = our_method_full_outlier_masks[row][col]
            is_outlier && !show_outliers && continue
            point_opacity = is_outlier ? outlier_point_opacity : data_point_opacity
            
            scatter!(ax_our, [σs[row]], [our_method_full_individual_points[row, col]], 
                    color=(:blue, point_opacity), markersize=4)
        end
    end
    
    # Plot our method partial state
    partial_state_included && lines!(ax_our, σs, our_method_partial_means, color=:brown, linestyle=:dash, label="Partial State")
    partial_state_included && band!(ax_our, σs, 
        max.(our_method_partial_means - our_method_partial_stds, 0), 
        our_method_partial_means + our_method_partial_stds, 
        color=(:brown, band_opacity))
        
    # Scatter plot for individual our method partial state data points
    if partial_state_included && show_data_points
        for col in eachindex(our_method_partial_individual_points[1, :])
            for row in eachindex(σs)
                # Determine if this point is an outlier
                is_outlier = our_method_partial_outlier_masks[row][col]
                is_outlier && !show_outliers && continue
                point_opacity = is_outlier ? outlier_point_opacity : data_point_opacity
                
                scatter!(ax_our, [σs[row]], [our_method_partial_individual_points[row, col]], 
                        color=(:brown, point_opacity), markersize=4)
            end
        end
    end
    
    axislegend(ax_our, position=:lt)
    
    # Save the our method comparison figure
    save(prefix*"/"*observation_mode*"_OurMethod_Comparison.png", fig_our)
    
    # return σs, baseline_full_means, our_method_full_means, baseline_partial_means, our_method_partial_means
end

# Function to filter outliers and compute statistics
function filter_outliers_and_compute_stats(data_matrix; std_dev_threshold=2.5)
    num_rows, num_cols = size(data_matrix)
    filtered_data = similar(data_matrix, Float64)
    outlier_masks = []
    
    for noise_level in 1:num_rows
        row_data = data_matrix[noise_level, :]
        row_mean = mean(row_data)
        row_std = std(row_data)
        
        # Identify which points are within threshold
        is_within_threshold = abs.(row_data .- row_mean) .<= std_dev_threshold * row_std
        push!(outlier_masks, .!is_within_threshold)
        
        # Filter the data for this noise level
        filtered_row_data = row_data[is_within_threshold]
        
        # If all points were filtered out, keep the original data
        if isempty(filtered_row_data)
            filtered_row_data = row_data
        end
        
        # Pad the filtered data to fit the matrix
        padded_filtered = fill(NaN, num_cols)
        padded_filtered[1:length(filtered_row_data)] .= filtered_row_data
        filtered_data[noise_level, :] = padded_filtered'
    end
    
    # Calculate statistics using filtered data
    means = [mean(filter(!isnan, filtered_data[i, :])) for i in 1:num_rows]
    stds = [std(filter(!isnan, filtered_data[i, :])) for i in 1:num_rows]
    
    return means, stds, outlier_masks
end

function graph_trajectories(
    plot_name,
    trajectories,
    game_structure,
    horizon;
    colors = [[(:red, 1.0), (:blue, 1.0), (:green, 1.0)], [(:red, 0.75), (:blue, 0.75), (:green, 0.75)]],
    constraints = nothing,
    p_state_dim = nothing
    # TODO automatically generate default colors based on number of players?
)
    CairoMakie.activate!()
    fig = CairoMakie.Figure()
    ax = CairoMakie.Axis(fig[1,1], aspect = DataAspect())

    image_data = CairoMakie.load("experiments/data/07_background.png")
    image_data = image_data[end:-1:1, :]
    image_data = image_data'
    # ax1 = Axis(fig[1,1], aspect = DataAspect())
    trfm = ImageTransformations.recenter(Rotations.RotMatrix(-2.303611),center(image_data))
    x_crop_min = 430
    x_crop_max = 875
    y_crop_min = 225
    y_crop_max = 1025
    
    scale = 1/10.25

    x = (x_crop_max - x_crop_min) * scale
    y = (y_crop_max - y_crop_min) * scale

    image_data = ImageTransformations.warp(image_data, trfm)
    image_data = Origin(0)(image_data)
    image_data = image_data[x_crop_min:x_crop_max, y_crop_min:y_crop_max]
    
    x_offset = -34.75
    y_offset = 22

    CairoMakie.image!(ax,
        x_offset..(x+x_offset),
        y_offset..(y+y_offset),
        image_data)

    #TODO horizon can probably be calculated from trajectory
    #TODO same with num_players/player state_dim (in reconstruct_solution)
    # Assumes first two elements in each state vector is x, y position

    

    n = num_players(game_structure.game.dynamics)
    p_state_dim = (p_state_dim === nothing) ? Int64(state_dim(game_structure.game.dynamics) // n) : p_state_dim

    for i in 1:n
        for j in eachindex(trajectories)
            if j == 1 #handle observations pulled from data file differently, format is different :/
                CairoMakie.lines!(ax, 
                    [trajectories[1][t][Block(i)][1] for t in eachindex(trajectories[1])],
                    [trajectories[1][t][Block(i)][2] for t in eachindex(trajectories[1])], 
                    color = colors[1][i], markersize = 5)
                CairoMakie.scatter!(ax,
                    [trajectories[1][end][Block(i)][1]],
                    [trajectories[1][end][Block(i)][2]],
                    color = colors[1][i], marker=:star5)
            else
                CairoMakie.lines!(ax, 
                    [trajectories[j][Block(t)][(i - 1) * p_state_dim + 1] for t in eachindex(blocks(trajectories[j]))],
                    [trajectories[j][Block(t)][(i - 1) * p_state_dim + 2] for t in eachindex(blocks(trajectories[j]))], 
                    color = colors[j][i], markersize = 5)
                CairoMakie.scatter!(ax,
                    [trajectories[j][Block(length(blocks(trajectories[j])))][(i - 1) * p_state_dim + 1]],
                    [trajectories[j][Block(length(blocks(trajectories[j])))][(i - 1) * p_state_dim + 2]],
                    color = colors[j][i], marker=:star5)
            end         
        end
        # CairoMakie.scatter!(ax,
        #         [trajectories[j][t][Block(length(blocks(trajectories[])))][(i - 1) * p_state_dim + 1]],
        #         [trajectories[j][t][Block(length(blocks(trajectories[2])))][(i - 1) * p_state_dim + 2]],
        #         color = colors[j][i], marker=:star5)
    end

        # CairoMakie.scatter!(ax, 
        #     [trajectories[2][Block(t)][(i - 1) * p_state_dim + 1] for t in eachindex(blocks(trajectories[2]))],
        #     [trajectories[2][Block(t)][(i - 1) * p_state_dim + 2] for t in eachindex(blocks(trajectories[2]))], 
        #     color = colors[2][i], markersize = 5)
        
        
    if constraints !== nothing
        x = LinRange(-40, 15, 100)
        y = LinRange(10, 105, 100)
        for i in x
            for j in y
                if any(constraints([i, j]) .< 0)
                    scatter!(ax, [i], [j], color = :black, markersize = 2)
                end
            end
        end
    end

    CairoMakie.save(plot_name*".png", fig)
end

function graph_crosswalk_trajectories(
    plot_name,
    trajectories,
    game_structure,
    horizon;
    colors = [[(:red, 1.0), (:blue, 1.0), (:green, 1.0)], [(:red, 0.25), (:blue, 0.25), (:green, 0.25)]],
    constraints = nothing,
    observations = nothing
)
    CairoMakie.activate!()
    fig = CairoMakie.Figure()
    ax = CairoMakie.Axis(fig[1,1], aspect = DataAspect())
    n = num_players(game_structure.game.dynamics)
    p_state_dim = Int64(state_dim(game_structure.game.dynamics) // n)

    for i in 1:n
        for j in eachindex(trajectories)
            CairoMakie.lines!(ax, 
                [trajectories[j][Block(t)][(i - 1) * p_state_dim + 1] for t in eachindex(blocks(trajectories[j]))],
                [trajectories[j][Block(t)][(i - 1) * p_state_dim + 2] for t in eachindex(blocks(trajectories[j]))], 
                color = colors[j][i])
            CairoMakie.scatter!(ax,
                [trajectories[j][Block(length(blocks(trajectories[j])))][(i - 1) * p_state_dim + 1]],
                [trajectories[j][Block(length(blocks(trajectories[j])))][(i - 1) * p_state_dim + 2]],
                color = colors[j][i], marker=:star5)
        end
    end
    if !isnothing(observations)
        for i in 1:n
            CairoMakie.scatter!(ax, 
                [observations[t][Block(i)][1] for t in eachindex(blocks(observations))],
                [observations[t][Block(i)][2] for t in eachindex(blocks(observations))], 
                color = (colors[2][i], 0.5))
        end
    end
    CairoMakie.save(plot_name*"_tmp.png", fig)
end

function parse_crosswalk_results(file_path::String)
    lines = filter(line -> !isempty(strip(line)), readlines(file_path))
    noise_levels = Float64[]
    errors = Float64[]
    
    for line in lines
        parts = split(strip(line))
        if length(parts) >= 2
            noise_level_str = parts[1]
            noise_level = parse(Float64, noise_level_str[2:end])
            error_value = parse(Float64, parts[2])
            
            push!(noise_levels, noise_level)
            push!(errors, error_value)
        end
    end
    
    error_matrix = reshape(errors, length(errors), 1)
    return error_matrix, noise_levels
end

function process_and_graph_crosswalk_results(;
    fo_baseline_file::String = "experiments/crosswalk/fo_baseline.txt",
    fo_our_method_file::String = "experiments/crosswalk/fo_ours.txt",
    po_baseline_file::String = "experiments/crosswalk/po_baseline.txt",
    po_our_method_file::String = "experiments/crosswalk/po_ours.txt",
    output_prefix = "./experiments/crosswalk/",
    std_dev_threshold = 2.5,
    show_outliers = false,
    y_axis_limit = [nothing, nothing]
)
    pdf_dir = joinpath(output_prefix, "pdf_plots")
    isdir(pdf_dir) || mkpath(pdf_dir)
    
    fo_baseline_matrix, fo_baseline_noise_levels = parse_crosswalk_results(fo_baseline_file)
    fo_our_method_matrix, fo_our_method_noise_levels = parse_crosswalk_results(fo_our_method_file)
    
    if fo_baseline_noise_levels != fo_our_method_noise_levels
        @warn "Noise levels in full observation baseline and our method files don't match"
    end
    
    fo_noise_levels = fo_baseline_noise_levels
    
    po_noise_levels = nothing
    po_baseline_matrix = nothing
    po_our_method_matrix = nothing
    
    if !isnothing(po_baseline_file) && !isnothing(po_our_method_file)
        po_baseline_matrix, po_baseline_noise_levels = parse_crosswalk_results(po_baseline_file)
        po_our_method_matrix, po_our_method_noise_levels = parse_crosswalk_results(po_our_method_file)
        
        if po_baseline_noise_levels != po_our_method_noise_levels
            @warn "Noise levels in partial observation baseline and our method files don't match"
        end
        
        if fo_noise_levels != po_baseline_noise_levels
            @warn "Noise levels in full and partial observation files don't match"
        end
        
        po_noise_levels = po_baseline_noise_levels
    end
    
    fo_baseline_means, fo_baseline_stds, fo_baseline_outlier_masks = 
        filter_outliers_and_compute_stats(fo_baseline_matrix; std_dev_threshold=std_dev_threshold)
    
    fo_our_method_means, fo_our_method_stds, fo_our_method_outlier_masks = 
        filter_outliers_and_compute_stats(fo_our_method_matrix; std_dev_threshold=std_dev_threshold)
    
    if !isnothing(po_baseline_matrix) && !isnothing(po_our_method_matrix)
        po_baseline_means, po_baseline_stds, po_baseline_outlier_masks = 
            filter_outliers_and_compute_stats(po_baseline_matrix; std_dev_threshold=std_dev_threshold)
        
        po_our_method_means, po_our_method_stds, po_our_method_outlier_masks = 
            filter_outliers_and_compute_stats(po_our_method_matrix; std_dev_threshold=std_dev_threshold)
    end
    
    fig = Figure(size = (800, 600), margins = (10, 10, 10, 10))
    ax = Axis(fig[1, 1],
        xlabel = "Noise Level (σ)",
        ylabel = "Trajectory Error",
        title = !isnothing(po_baseline_file) ? "Crosswalk: Full vs Partial Observation" : "Crosswalk: Trajectory Error",
        limits = (nothing, (y_axis_limit[1], y_axis_limit[2]))
    )
    
    colors = ["coral", "tan2", "olivedrab", "steelblue"]
    fo_b = 1
    fo_m = 2
    po_b = 3
    po_m = 4
    
    data_point_opacity = 0.2
    outlier_point_opacity = data_point_opacity / 3
    band_opacity = 0.3
    
    lines!(ax, fo_noise_levels, fo_baseline_means, color=colors[fo_b], label="Full Obs. Baseline")
    band!(ax, fo_noise_levels, 
        max.(fo_baseline_means - fo_baseline_stds, 0), 
        fo_baseline_means + fo_baseline_stds, 
        color=(colors[fo_b], band_opacity))
    
    if show_outliers
        for col in 1:size(fo_baseline_matrix, 2)
            for row in 1:size(fo_baseline_matrix, 1)
                is_outlier = fo_baseline_outlier_masks[row][col]
                point_opacity = is_outlier ? outlier_point_opacity : data_point_opacity
                
                scatter!(ax, [fo_noise_levels[row]], [fo_baseline_matrix[row, col]], 
                        color=(colors[fo_b], point_opacity), markersize=4)
            end
        end
    end
    
    lines!(ax, fo_noise_levels, fo_our_method_means, color=colors[fo_m], label="Full Obs. Our Method")
    band!(ax, fo_noise_levels, 
        max.(fo_our_method_means - fo_our_method_stds, 0), 
        fo_our_method_means + fo_our_method_stds, 
        color=(colors[fo_m], band_opacity))
    
    if show_outliers
        for col in 1:size(fo_our_method_matrix, 2)
            for row in 1:size(fo_our_method_matrix, 1)
                is_outlier = fo_our_method_outlier_masks[row][col]
                point_opacity = is_outlier ? outlier_point_opacity : data_point_opacity
                
                scatter!(ax, [fo_noise_levels[row]], [fo_our_method_matrix[row, col]], 
                        color=(colors[fo_m], point_opacity), markersize=4)
            end
        end
    end
    
    if !isnothing(po_baseline_matrix) && !isnothing(po_our_method_matrix)
        lines!(ax, po_noise_levels, po_baseline_means, color=colors[po_b], linestyle=:dash, label="Partial Obs. Baseline")
        band!(ax, po_noise_levels, 
            max.(po_baseline_means - po_baseline_stds, 0), 
            po_baseline_means + po_baseline_stds, 
            color=(colors[po_b], band_opacity))
        
        if show_outliers
            for col in 1:size(po_baseline_matrix, 2)
                for row in 1:size(po_baseline_matrix, 1)
                    is_outlier = po_baseline_outlier_masks[row][col]
                    point_opacity = is_outlier ? outlier_point_opacity : data_point_opacity
                    
                    scatter!(ax, [po_noise_levels[row]], [po_baseline_matrix[row, col]], 
                            color=(colors[po_b], point_opacity), markersize=4)
                end
            end
        end
        
        lines!(ax, po_noise_levels, po_our_method_means, color=colors[po_m], linestyle=:dash, label="Partial Obs. Our Method")
        band!(ax, po_noise_levels, 
            max.(po_our_method_means - po_our_method_stds, 0), 
            po_our_method_means + po_our_method_stds, 
            color=(colors[po_m], band_opacity))
        
        if show_outliers
            for col in 1:size(po_our_method_matrix, 2)
                for row in 1:size(po_our_method_matrix, 1)
                    is_outlier = po_our_method_outlier_masks[row][col]
                    point_opacity = is_outlier ? outlier_point_opacity : data_point_opacity
                    
                    scatter!(ax, [po_noise_levels[row]], [po_our_method_matrix[row, col]], 
                            color=(colors[po_m], point_opacity), markersize=4)
                end
            end
        end
    end
    
    axislegend(ax, position=:lt)
    
    save(joinpath(output_prefix, "fo_po_both_crosswalk.png"), fig)
    save(joinpath(pdf_dir, "fo_po_both_crosswalk.pdf"), fig, pt_per_unit=1, pt_per_inch=72)
    
    if !isnothing(po_baseline_matrix) && !isnothing(po_our_method_matrix)
        fig_fo = Figure(size = (800, 600), margins = (10, 10, 10, 10))
        ax_fo = Axis(fig_fo[1, 1],
            xlabel = "Noise Level (σ)",
            ylabel = "Trajectory Error",
            title = "Crosswalk Full Observation: Baseline vs Our Method",
            limits = (nothing, (y_axis_limit[1], y_axis_limit[2]))
        )
        
        lines!(ax_fo, fo_noise_levels, fo_baseline_means, color=colors[fo_b], label="Baseline")
        band!(ax_fo, fo_noise_levels, 
            max.(fo_baseline_means - fo_baseline_stds, 0), 
            fo_baseline_means + fo_baseline_stds, 
            color=(colors[fo_b], band_opacity))
        
        lines!(ax_fo, fo_noise_levels, fo_our_method_means, color=colors[fo_m], label="Our Method")
        band!(ax_fo, fo_noise_levels, 
            max.(fo_our_method_means - fo_our_method_stds, 0), 
            fo_our_method_means + fo_our_method_stds, 
            color=(colors[fo_m], band_opacity))
        
        axislegend(ax_fo, position=:lt)
        
        save(joinpath(output_prefix, "fo_both_crosswalk.png"), fig_fo)
        save(joinpath(pdf_dir, "fo_both_crosswalk.pdf"), fig_fo, pt_per_unit=1, pt_per_inch=72)
        
        fig_po = Figure(size = (800, 600), margins = (10, 10, 10, 10))
        ax_po = Axis(fig_po[1, 1],
            xlabel = "Noise Level (σ)",
            ylabel = "Trajectory Error",
            title = "Crosswalk Partial Observation: Baseline vs Our Method",
            limits = (nothing, (y_axis_limit[1], y_axis_limit[2]))
        )
        
        lines!(ax_po, po_noise_levels, po_baseline_means, color=colors[po_b], label="Baseline")
        band!(ax_po, po_noise_levels, 
            max.(po_baseline_means - po_baseline_stds, 0), 
            po_baseline_means + po_baseline_stds, 
            color=(colors[po_b], band_opacity))
        
        lines!(ax_po, po_noise_levels, po_our_method_means, color=colors[po_m], label="Our Method")
        band!(ax_po, po_noise_levels, 
            max.(po_our_method_means - po_our_method_stds, 0), 
            po_our_method_means + po_our_method_stds, 
            color=(colors[po_m], band_opacity))
        
        axislegend(ax_po, position=:lt)
        
        save(joinpath(output_prefix, "po_both_crosswalk.png"), fig_po)
        save(joinpath(pdf_dir, "po_both_crosswalk.pdf"), fig_po, pt_per_unit=1, pt_per_inch=72)
        
        fig_b = Figure(size = (800, 600), margins = (10, 10, 10, 10))
        ax_b = Axis(fig_b[1, 1],
            xlabel = "Noise Level (σ)",
            ylabel = "Trajectory Error",
            title = "Crosswalk Baseline: Full vs Partial Observation",
            limits = (nothing, (y_axis_limit[1], y_axis_limit[2]))
        )
        
        lines!(ax_b, fo_noise_levels, fo_baseline_means, color=colors[fo_b], label="Full Observation")
        band!(ax_b, fo_noise_levels, 
            max.(fo_baseline_means - fo_baseline_stds, 0), 
            fo_baseline_means + fo_baseline_stds, 
            color=(colors[fo_b], band_opacity))
        
        lines!(ax_b, po_noise_levels, po_baseline_means, color=colors[po_b], linestyle=:dash, label="Partial Observation")
        band!(ax_b, po_noise_levels, 
            max.(po_baseline_means - po_baseline_stds, 0), 
            po_baseline_means + po_baseline_stds, 
            color=(colors[po_b], band_opacity))
        
        axislegend(ax_b, position=:lt)
        
        save(joinpath(output_prefix, "fo_po_baseline_crosswalk.png"), fig_b)
        save(joinpath(pdf_dir, "fo_po_baseline_crosswalk.pdf"), fig_b, pt_per_unit=1, pt_per_inch=72)
        
        fig_m = Figure(size = (800, 600), margins = (10, 10, 10, 10))
        ax_m = Axis(fig_m[1, 1],
            xlabel = "Noise Level (σ)",
            ylabel = "Trajectory Error",
            title = "Crosswalk Our Method: Full vs Partial Observation",
            limits = (nothing, (y_axis_limit[1], y_axis_limit[2]))
        )
        
        lines!(ax_m, fo_noise_levels, fo_our_method_means, color=colors[fo_m], label="Full Observation")
        band!(ax_m, fo_noise_levels, 
            max.(fo_our_method_means - fo_our_method_stds, 0), 
            fo_our_method_means + fo_our_method_stds, 
            color=(colors[fo_m], band_opacity))
        
        lines!(ax_m, po_noise_levels, po_our_method_means, color=colors[po_m], linestyle=:dash, label="Partial Observation")
        band!(ax_m, po_noise_levels, 
            max.(po_our_method_means - po_our_method_stds, 0), 
            po_our_method_means + po_our_method_stds, 
            color=(colors[po_m], band_opacity))
        
        axislegend(ax_m, position=:lt)
        
        save(joinpath(output_prefix, "fo_po_our_method_crosswalk.png"), fig_m)
        save(joinpath(pdf_dir, "fo_po_our_method_crosswalk.pdf"), fig_m, pt_per_unit=1, pt_per_inch=72)
    end
    
    result = Dict(
        "noise_levels" => fo_noise_levels,
        "full_observation" => Dict(
            "baseline" => (means=fo_baseline_means, stds=fo_baseline_stds),
            "our_method" => (means=fo_our_method_means, stds=fo_our_method_stds)
        )
    )
    
    if !isnothing(po_baseline_matrix) && !isnothing(po_our_method_matrix)
        result["partial_observation"] = Dict(
            "baseline" => (means=po_baseline_means, stds=po_baseline_stds),
            "our_method" => (means=po_our_method_means, stds=po_our_method_stds)
        )
    end
    
    return result
end


function parse_rh_file(filename)
    times = Int[]
    inverse_costs = Float64[]
    baseline_costs = Float64[]
    params_history = Vector{Float64}[]
    baseline_params_history = Vector{Float64}[]
    
    current_time = nothing
    current_inverse_cost = nothing
    current_baseline_cost = nothing
    current_params = nothing
    current_baseline_params = nothing
    
    for line in eachline(filename)
        if startswith(line, "t:")
            if !isnothing(current_time)
                push!(times, current_time)
                push!(inverse_costs, current_inverse_cost)
                push!(baseline_costs, current_baseline_cost)
                push!(params_history, current_params)
                push!(baseline_params_history, current_baseline_params)
            end
            parts = split(line, ":")
            current_time = parse(Int, strip(parts[2]))
        elseif startswith(line, "inverse_costs_future:")
            parts = split(line, ":")
            current_inverse_cost = parse(Float64, strip(parts[2]))
        elseif startswith(line, "baseline_inverse_costs_future:")
            parts = split(line, ":")
            current_baseline_cost = parse(Float64, strip(parts[2]))
        elseif startswith(line, "params_future:")
            parts = split(line, ":", limit=2)
            param_str = strip(parts[2])
            
            # Process the parameter string to handle brackets
            # First, remove overall brackets if present
            if startswith(param_str, "[") && endswith(param_str, "]")
                param_str = param_str[2:end-1]
            end
            
            # Split the parameter string by commas
            param_parts = split(param_str, ",")
            
            # Process each part to clean any remaining brackets
            for i in 1:length(param_parts)
                param_parts[i] = strip(param_parts[i])
                # Remove any left bracket at the beginning
                if startswith(param_parts[i], "[")
                    param_parts[i] = param_parts[i][2:end]
                end
                # Remove any right bracket at the end
                if endswith(param_parts[i], "]")
                    param_parts[i] = param_parts[i][1:end-1]
                end
            end
            
            # Parse the cleaned parameter parts
            current_params = parse.(Float64, param_parts)
        elseif startswith(line, "baseline_params_future:")
            parts = split(line, ":", limit=2)
            param_str = strip(parts[2])
            
            # Process the parameter string to handle brackets
            # First, remove overall brackets if present
            if startswith(param_str, "[") && endswith(param_str, "]")
                param_str = param_str[2:end-1]
            end
            
            # Split the parameter string by commas
            param_parts = split(param_str, ",")
            
            # Process each part to clean any remaining brackets
            for i in 1:length(param_parts)
                param_parts[i] = strip(param_parts[i])
                # Remove any left bracket at the beginning
                if startswith(param_parts[i], "[")
                    param_parts[i] = param_parts[i][2:end]
                end
                # Remove any right bracket at the end
                if endswith(param_parts[i], "]")
                    param_parts[i] = param_parts[i][1:end-1]
                end
            end
            
            # Parse the cleaned parameter parts
            current_baseline_params = parse.(Float64, param_parts)
        end
    end
    
    # Push the last entry
    if !isnothing(current_time)
        push!(times, current_time)
        push!(inverse_costs, current_inverse_cost)
        push!(baseline_costs, current_baseline_cost)
        push!(params_history, current_params)
        push!(baseline_params_history, current_baseline_params)
    end
    
    return times, inverse_costs, baseline_costs, params_history, baseline_params_history
end


function plot_rh_costs(filename; title="Inverse Costs Over Time")
    times, inverse_costs, baseline_costs, _, _ = parse_rh_file(filename)
    
    CairoMakie.activate!()
    fig = Figure(resolution=(800, 600))
    ax = Axis(fig[1, 1], title=title, xlabel="Time Step", ylabel="Cost")
    
    lines!(ax, times, inverse_costs, label="Inverse Costs", color=:blue)
    lines!(ax, times, baseline_costs, label="Baseline Costs", color=:red)
    
    axislegend(ax, position=:rt)
    save("costs over time.pdf", fig, pt_per_unit=1, pt_per_inch=72)
    return fig
end

"""
    plot_rh_parameter_differences(filename, true_params; title="Parameter Differences Over Time")

Plot the L2 norm of parameter differences from true parameters over time using data from an rh.txt file.

# Arguments
- `filename`: Path to the rh.txt file
- `true_params`: The true parameter vector to compare against
- `title`: Optional title for the plot

# Returns
- A Makie figure showing the parameter difference trajectory
"""
function plot_rh_parameter_differences(filename, true_params; title="Parameter Differences Over Time")
    times, _, _, params_history, baseline_params_history = parse_rh_file(filename)
    
    CairoMakie.activate!()
    fig = Figure(resolution=(800, 600))
    ax = Axis(fig[1, 1], title=title, xlabel="Time Step", ylabel="L2 Norm of Parameter Difference")
    
    # Extract goal positions for both methods
    # Note: Regular params have 8 parameters per player, baseline has 7 per player
    goal_indices_regular = [1,2, 9,10, 17,18, 25,26]
    goal_indices_baseline = [1,2, 8,9, 15,16, 22,23]
    
    # Calculate L2 norm of differences for goal positions
    differences = [norm(params[goal_indices_regular] - true_params[goal_indices_regular]) for params in params_history]
    baseline_differences = [norm(baseline_params[goal_indices_baseline] - true_params[goal_indices_regular]) for baseline_params in baseline_params_history]
    
    lines!(ax, times, differences, label="Our Method", color=:blue)
    lines!(ax, times, baseline_differences, label="Baseline", color=:blue, linestyle=:dash)
    
    axislegend(ax, position=:rt)
    save("parameter differences over time.pdf", fig, pt_per_unit=1, pt_per_inch=72)
    
    # Create per-player parameter differences plot
    plot_rh_parameter_differences_per_player(filename, true_params)
    
    return fig
end

function plot_rh_parameter_differences_per_player(filename, true_params; title="Parameter Differences Per Player Over Time")
    times, _, _, params_history, baseline_params_history = parse_rh_file(filename)
    
    CairoMakie.activate!()
    fig = Figure(resolution=(800, 600))
    ax = Axis(fig[1, 1], title=title, xlabel="Time Step", ylabel="L2 Norm of Parameter Difference")
    
    # Regular parameters have 8 params per player
    player1_indices_regular = collect(1:8)      # First player parameters
    player2_indices_regular = collect(9:16)     # Second player parameters
    player3_indices_regular = collect(17:24)    # Third player parameters
    player4_indices_regular = collect(25:32)    # Fourth player parameters
    
    # Baseline parameters have 7 params per player
    player1_indices_baseline = collect(1:7)     # First player parameters
    player2_indices_baseline = collect(8:14)    # Second player parameters
    player3_indices_baseline = collect(15:21)   # Third player parameters
    player4_indices_baseline = collect(22:28)   # Fourth player parameters
    
    # True parameters structure matches regular params (8 per player)
    player1_indices_true = collect(1:8)
    player2_indices_true = collect(9:16)
    player3_indices_true = collect(17:24)
    player4_indices_true = collect(25:32)
    
    # Calculate differences for each player - regular parameters
    player1_differences = [norm(params[player1_indices_regular] - true_params[player1_indices_true]) for params in params_history]
    player2_differences = [norm(params[player2_indices_regular] - true_params[player2_indices_true]) for params in params_history]
    player3_differences = [norm(params[player3_indices_regular] - true_params[player3_indices_true]) for params in params_history]
    player4_differences = [norm(params[player4_indices_regular] - true_params[player4_indices_true]) for params in params_history]
    
    # For baseline, we just compare the available parameters
    # Since baseline has 7 params and true has 8, we'll compare just the first 7
    player1_baseline_differences = [norm(params[player1_indices_baseline][1:7] - true_params[player1_indices_true][1:7]) for params in baseline_params_history]
    player2_baseline_differences = [norm(params[player2_indices_baseline][1:7] - true_params[player2_indices_true][1:7]) for params in baseline_params_history]
    player3_baseline_differences = [norm(params[player3_indices_baseline][1:7] - true_params[player3_indices_true][1:7]) for params in baseline_params_history]
    player4_baseline_differences = [norm(params[player4_indices_baseline][1:7] - true_params[player4_indices_true][1:7]) for params in baseline_params_history]
    
    # Define colors for each player
    player_colors = [:blue, :green, :purple, :orange]
    
    # Player 1 - Our Method and Baseline (same color, different line styles)
    lines!(ax, times, player1_differences, label="Player 1 Our Method", color=player_colors[1])
    lines!(ax, times, player1_baseline_differences, label="Player 1 Baseline", color=player_colors[1], linestyle=:dash)
    
    # Player 2 - Our Method and Baseline
    lines!(ax, times, player2_differences, label="Player 2 Our Method", color=player_colors[2])
    lines!(ax, times, player2_baseline_differences, label="Player 2 Baseline", color=player_colors[2], linestyle=:dash)
    
    # Player 3 - Our Method and Baseline
    lines!(ax, times, player3_differences, label="Player 3 Our Method", color=player_colors[3])
    lines!(ax, times, player3_baseline_differences, label="Player 3 Baseline", color=player_colors[3], linestyle=:dash)
    
    # Player 4 - Our Method and Baseline
    lines!(ax, times, player4_differences, label="Player 4 Our Method", color=player_colors[4])
    lines!(ax, times, player4_baseline_differences, label="Player 4 Baseline", color=player_colors[4], linestyle=:dash)
    
    axislegend(ax, position=:rt)
    save("parameter differences per player over time.pdf", fig, pt_per_unit=1, pt_per_inch=72)
    
    # Create a separate plot for goal differences per player
    plot_rh_goal_differences_per_player(filename, true_params)
    
    return fig
end

function plot_rh_goal_differences_per_player(filename, true_params; title="Goal Position Differences Per Player Over Time")
    times, _, _, params_history, baseline_params_history = parse_rh_file(filename)
    
    CairoMakie.activate!()
    fig = Figure(resolution=(800, 600))
    ax = Axis(fig[1, 1], title=title, xlabel="Time Step", ylabel="L2 Norm of Goal Position Difference")
    
    # Goal position indices for each player - regular params
    player1_goal_indices_regular = [1, 2]     # First player goal position
    player2_goal_indices_regular = [9, 10]    # Second player goal position
    player3_goal_indices_regular = [17, 18]   # Third player goal position
    player4_goal_indices_regular = [25, 26]   # Fourth player goal position
    
    # Goal position indices for each player - baseline params
    player1_goal_indices_baseline = [1, 2]    # First player goal position
    player2_goal_indices_baseline = [8, 9]    # Second player goal position
    player3_goal_indices_baseline = [15, 16]  # Third player goal position
    player4_goal_indices_baseline = [22, 23]  # Fourth player goal position
    
    # Calculate goal position differences for each player
    player1_goal_differences = [norm(params[player1_goal_indices_regular] - true_params[player1_goal_indices_regular]) for params in params_history]
    player2_goal_differences = [norm(params[player2_goal_indices_regular] - true_params[player2_goal_indices_regular]) for params in params_history]
    player3_goal_differences = [norm(params[player3_goal_indices_regular] - true_params[player3_goal_indices_regular]) for params in params_history]
    player4_goal_differences = [norm(params[player4_goal_indices_regular] - true_params[player4_goal_indices_regular]) for params in params_history]
    
    player1_baseline_goal_differences = [norm(params[player1_goal_indices_baseline] - true_params[player1_goal_indices_regular]) for params in baseline_params_history]
    player2_baseline_goal_differences = [norm(params[player2_goal_indices_baseline] - true_params[player2_goal_indices_regular]) for params in baseline_params_history]
    player3_baseline_goal_differences = [norm(params[player3_goal_indices_baseline] - true_params[player3_goal_indices_regular]) for params in baseline_params_history]
    player4_baseline_goal_differences = [norm(params[player4_goal_indices_baseline] - true_params[player4_goal_indices_regular]) for params in baseline_params_history]
    
    # Define colors for each player
    player_colors = [:blue, :green, :purple, :orange]
    
    # Player 1 - Our Method and Baseline (same color, different line styles)
    lines!(ax, times, player1_goal_differences, label="Player 1 Our Method", color=player_colors[1])
    lines!(ax, times, player1_baseline_goal_differences, label="Player 1 Baseline", color=player_colors[1], linestyle=:dash)
    
    # Player 2 - Our Method and Baseline
    lines!(ax, times, player2_goal_differences, label="Player 2 Our Method", color=player_colors[2])
    lines!(ax, times, player2_baseline_goal_differences, label="Player 2 Baseline", color=player_colors[2], linestyle=:dash)
    
    # Player 3 - Our Method and Baseline
    lines!(ax, times, player3_goal_differences, label="Player 3 Our Method", color=player_colors[3])
    lines!(ax, times, player3_baseline_goal_differences, label="Player 3 Baseline", color=player_colors[3], linestyle=:dash)
    
    # Player 4 - Our Method and Baseline
    lines!(ax, times, player4_goal_differences, label="Player 4 Our Method", color=player_colors[4])
    lines!(ax, times, player4_baseline_goal_differences, label="Player 4 Baseline", color=player_colors[4], linestyle=:dash)
    
    axislegend(ax, position=:rt)
    save("goal differences per player over time.pdf", fig, pt_per_unit=1, pt_per_inch=72)
    
    return fig
end

function graph_rh_snapshot(
    plot_name,
    observations,
    inverse_trajectory,
    baseline_inverse_trajectory,
    predicted_trajectory,
    baseline_predicted_trajectory,
    game_structure,
    horizon;
    colors = [
        [(:red, 1.0), (:blue, 1.0), (:green, 1.0), (:purple, 1.0)],  # Observations
        [(:red, 0.8), (:blue, 0.8), (:green, 0.8), (:purple, 0.8)],  # Inverse trajectory
        [(:red, 0.6), (:blue, 0.6), (:green, 0.6), (:purple, 0.6)],  # Baseline inverse
        [(:red, 0.4), (:blue, 0.4), (:green, 0.4), (:purple, 0.4)],  # Predicted
        [(:red, 0.2), (:blue, 0.2), (:green, 0.2), (:purple, 0.2)]   # Baseline predicted
    ],
    constraints = nothing,
    p_state_dim = nothing
)
    CairoMakie.activate!()
    fig = CairoMakie.Figure(resolution=(1000, 800))
    ax = CairoMakie.Axis(fig[1,1], aspect = DataAspect())

    # Load and process background image
    image_data = CairoMakie.load("experiments/data/07_background.png")
    image_data = image_data[end:-1:1, :]
    image_data = image_data'
    trfm = ImageTransformations.recenter(Rotations.RotMatrix(-2.303611),center(image_data))
    x_crop_min = 430
    x_crop_max = 875
    y_crop_min = 225
    y_crop_max = 1025
    
    scale = 1/10.25
    x = (x_crop_max - x_crop_min) * scale
    y = (y_crop_max - y_crop_min) * scale

    image_data = ImageTransformations.warp(image_data, trfm)
    image_data = Origin(0)(image_data)
    image_data = image_data[x_crop_min:x_crop_max, y_crop_min:y_crop_max]
    
    x_offset = -34.75
    y_offset = 22

    CairoMakie.image!(ax,
        x_offset..(x+x_offset),
        y_offset..(y+y_offset),
        image_data)

    n = num_players(game_structure.game.dynamics)
    p_state_dim = (p_state_dim === nothing) ? Int64(state_dim(game_structure.game.dynamics) // n) : p_state_dim

    # Create legend elements
    legend_elements = []
    legend_labels = []

    # Plot observations
    for i in 1:n
        scatter_plot = CairoMakie.scatter!(ax,
            [observations[t][Block(i)][1] for t in eachindex(observations)],
            [observations[t][Block(i)][2] for t in eachindex(observations)],
            color = colors[1][i], markersize = 5)
        push!(legend_elements, scatter_plot)
        push!(legend_labels, "Player $i Obs")
    end

    # Plot inverse trajectories
    for i in 1:n
        line_plot = CairoMakie.lines!(ax,
            [inverse_trajectory[Block(t)][(i-1)*p_state_dim + 1] for t in eachindex(blocks(inverse_trajectory))],
            [inverse_trajectory[Block(t)][(i-1)*p_state_dim + 2] for t in eachindex(blocks(inverse_trajectory))],
            color = colors[2][i])
        push!(legend_elements, line_plot)
        push!(legend_labels, "Player $i Inv")
    end

    # Plot baseline inverse trajectories
    for i in 1:n
        line_plot = CairoMakie.lines!(ax,
            [baseline_inverse_trajectory[Block(t)][(i-1)*p_state_dim + 1] for t in eachindex(blocks(baseline_inverse_trajectory))],
            [baseline_inverse_trajectory[Block(t)][(i-1)*p_state_dim + 2] for t in eachindex(blocks(baseline_inverse_trajectory))],
            color = colors[3][i], linestyle=:dash)
        push!(legend_elements, line_plot)
        push!(legend_labels, "Player $i Base Inv")
    end

    # Plot predicted trajectories
    for i in 1:n
        line_plot = CairoMakie.lines!(ax,
            [predicted_trajectory[Block(t)][(i-1)*p_state_dim + 1] for t in eachindex(blocks(predicted_trajectory))],
            [predicted_trajectory[Block(t)][(i-1)*p_state_dim + 2] for t in eachindex(blocks(predicted_trajectory))],
            color = colors[4][i])
        push!(legend_elements, line_plot)
        push!(legend_labels, "Player $i Pred")
    end

    # Plot baseline predicted trajectories
    for i in 1:n
        line_plot = CairoMakie.lines!(ax,
            [baseline_predicted_trajectory[Block(t)][(i-1)*p_state_dim + 1] for t in eachindex(blocks(baseline_predicted_trajectory))],
            [baseline_predicted_trajectory[Block(t)][(i-1)*p_state_dim + 2] for t in eachindex(blocks(baseline_predicted_trajectory))],
            color = colors[5][i], linestyle=:dash)
        push!(legend_elements, line_plot)
        push!(legend_labels, "Player $i Base Pred")
    end

    # Add constraints if provided
    if constraints !== nothing
        x = LinRange(-40, 15, 100)
        y = LinRange(10, 105, 100)
        for i in x
            for j in y
                if any(constraints([i, j]) .< 0)
                    scatter!(ax, [i], [j], color = :black, markersize = 2)
                end
            end
        end
    end

    # Create legend in a separate column
    Legend(fig[1,2], legend_elements, legend_labels, 
        "Trajectory Types", 
        framevisible = true,
        padding = (10, 10, 10, 10),
        rowgap = 5)

    # Save the figure
    CairoMakie.save(plot_name*".png", fig)
end

end