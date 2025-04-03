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

using Infiltrator

include("../../src/utils/Utils.jl")

function graph_metrics(
    baseline_file::String,
    our_method_file::String;
    observation_mode="both",
    pre_prefix = "./experiments/In-D/",
    partial_state_included=true,
    std_dev_threshold=2.5,
    show_outliers=false,
    y_axis_limit=150
)
    prefix = pre_prefix
    CairoMakie.activate!()

    # Create PDF directory if it doesn't exist
    pdf_dir = joinpath(prefix, "pdf_plots")
    isdir(pdf_dir) || mkpath(pdf_dir)

    colors = ["coral", "tan2", "olivedrab", "steelblue"]
    fo_b = 1
    fo_n = 2
    po_b = 3
    po_n = 4

    data_point_opacity = 0.2
    outlier_point_opacity = data_point_opacity / 3
    band_opacity = 0.3

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
    fig_all = Figure(size = (800, 600), margins = (10, 10, 10, 10))
    ax_all = Axis(fig_all[1, 1],
        xlabel = "Noise Level (σ)",
        ylabel = "Trajectory Error",
        title = "Trajectory Error vs Noise Level (All Methods)",
        limits = (nothing, (0, y_axis_limit))
    )

    # Plot baseline with non-negative lower bound (full state)
    lines!(ax_all, σs, baseline_full_means, color=colors[fo_b], label="Baseline (Full State)")
    band!(ax_all, σs, 
        max.(baseline_full_means - baseline_full_stds, 0), 
        baseline_full_means + baseline_full_stds, 
        color=(colors[fo_b], band_opacity))

    # Scatter plot for individual baseline full state data points
    for col in eachindex(baseline_full_individual_points[1, :])
        for row in eachindex(σs)
            # Determine if this point is an outlier
            is_outlier = baseline_full_outlier_masks[row][col]
            is_outlier && !show_outliers && continue
            point_opacity = is_outlier ? outlier_point_opacity : data_point_opacity
            
            scatter!(ax_all, [σs[row]], [baseline_full_individual_points[row, col]], 
                    color=(colors[fo_b], point_opacity), markersize=4)
        end
    end

    # Plot our method with non-negative lower bound (full state)
    lines!(ax_all, σs, our_method_full_means, color=colors[fo_n], label="Our Method (Full State)")
    band!(ax_all, σs, 
        max.(our_method_full_means - our_method_full_stds, 0), 
        our_method_full_means + our_method_full_stds, 
        color=(colors[fo_n], band_opacity+0.1))
        
    # Scatter plot for individual our method full state data points
    for col in eachindex(our_method_full_individual_points[1, :])
        for row in eachindex(σs)
            # Determine if this point is an outlier
            is_outlier = our_method_full_outlier_masks[row][col]
            is_outlier && !show_outliers && continue
            point_opacity = is_outlier ? outlier_point_opacity : data_point_opacity
            
            scatter!(ax_all, [σs[row]], [our_method_full_individual_points[row, col]], 
                    color=(colors[fo_n], point_opacity), markersize=4)
        end
    end

    # Plot baseline with non-negative lower bound (partial state)
    partial_state_included && lines!(ax_all, σs, baseline_partial_means, color=colors[po_b], linestyle=:dash, label="Baseline (Partial State)")
    partial_state_included && band!(ax_all, σs, 
        max.(baseline_partial_means - baseline_partial_stds, 0), 
        baseline_partial_means + baseline_partial_stds, 
        color=(colors[po_b], band_opacity))
        
    # Scatter plot for individual baseline partial state data points
    if partial_state_included
        for col in eachindex(baseline_partial_individual_points[1, :])
            for row in eachindex(σs)
                # Determine if this point is an outlier
                is_outlier = baseline_partial_outlier_masks[row][col]
                is_outlier && !show_outliers && continue
                point_opacity = is_outlier ? outlier_point_opacity : data_point_opacity
                
                scatter!(ax_all, [σs[row]], [baseline_partial_individual_points[row, col]], 
                        color=(colors[po_b], point_opacity), markersize=4)
            end
        end
    end
    
    # Plot our method with non-negative lower bound (partial state)
    partial_state_included && lines!(ax_all, σs, our_method_partial_means, color=colors[po_n], linestyle=:dash, label="Our Method (Partial State)")
    partial_state_included && band!(ax_all, σs, 
        max.(our_method_partial_means - our_method_partial_stds, 0), 
        our_method_partial_means + our_method_partial_stds, 
        color=(colors[po_n], band_opacity))
        
    # Scatter plot for individual our method partial state data points
    if partial_state_included
        for col in eachindex(our_method_partial_individual_points[1, :])
            for row in eachindex(σs)
                # Determine if this point is an outlier
                is_outlier = our_method_partial_outlier_masks[row][col]
                is_outlier && !show_outliers && continue
                point_opacity = is_outlier ? outlier_point_opacity : data_point_opacity
                
                scatter!(ax_all, [σs[row]], [our_method_partial_individual_points[row, col]], 
                        color=(colors[po_n], point_opacity), markersize=4)
            end
        end
    end

    # Add legend
    axislegend(ax_all, position=:lt)

    # Save the all methods figure
    save(prefix*"/"*"po_fo_all_ind.png", fig_all)
    save(pdf_dir*"/"*"po_fo_all_ind.pdf", fig_all, pt_per_unit=1, pt_per_inch=72)
    
    # Create figure for baseline comparison
    fig_baseline = Figure(size = (800, 600), margins = (10, 10, 10, 10))
    ax_baseline = Axis(fig_baseline[1, 1],
        xlabel = "Noise Level (σ)",
        ylabel = "Trajectory Error",
        title = "Baseline: Full State vs Partial State",
        limits = (nothing, (0, y_axis_limit))
    )
    
    # Plot baseline full state
    lines!(ax_baseline, σs, baseline_full_means, color=colors[fo_b], label="Full State")
    band!(ax_baseline, σs, 
        max.(baseline_full_means - baseline_full_stds, 0), 
        baseline_full_means + baseline_full_stds, 
        color=(colors[fo_b], band_opacity))
        
    # Scatter plot for individual baseline full state data points
    for col in eachindex(baseline_full_individual_points[1, :])
        for row in eachindex(σs)
            # Determine if this point is an outlier
            is_outlier = baseline_full_outlier_masks[row][col]
            is_outlier && !show_outliers && continue
            point_opacity = is_outlier ? outlier_point_opacity : data_point_opacity
            scatter!(ax_baseline, [σs[row]], [baseline_full_individual_points[row, col]], 
                    color=(colors[fo_b], point_opacity), markersize=4)
        end
    end
    
    # Plot baseline partial state
    partial_state_included && lines!(ax_baseline, σs, baseline_partial_means, color=colors[po_b], linestyle=:dash, label="Partial State")
    partial_state_included && band!(ax_baseline, σs, 
        max.(baseline_partial_means - baseline_partial_stds, 0), 
        baseline_partial_means + baseline_partial_stds, 
        color=(colors[po_b], band_opacity))
        
    # Scatter plot for individual baseline partial state data points
    if partial_state_included
        for col in eachindex(baseline_partial_individual_points[1, :])
            for row in eachindex(σs)
                # Determine if this point is an outlier
                is_outlier = baseline_partial_outlier_masks[row][col]
                is_outlier && !show_outliers && continue
                point_opacity = is_outlier ? outlier_point_opacity : data_point_opacity
                
                scatter!(ax_baseline, [σs[row]], [baseline_partial_individual_points[row, col]], 
                        color=(colors[po_b], point_opacity), markersize=4)
            end
        end
    end
    
    axislegend(ax_baseline, position=:lt)
    
    # Save the baseline comparison figure
    save(prefix*"/"*"po_fo_baseline_ind.png", fig_baseline)
    save(pdf_dir*"/"*"po_fo_baseline_ind.pdf", fig_baseline, pt_per_unit=1, pt_per_inch=72)
    
    # Create figure for our method comparison
    fig_our = Figure(size = (800, 600), margins = (10, 10, 10, 10))
    ax_our = Axis(fig_our[1, 1],
        xlabel = "Noise Level (σ)",
        ylabel = "Trajectory Error",
        title = "Our Method: Full State vs Partial State",
        limits = (nothing, (0, y_axis_limit))
    )
    
    # Plot our method full state
    lines!(ax_our, σs, our_method_full_means, color=colors[fo_n], label="Full State")
    band!(ax_our, σs, 
        max.(our_method_full_means - our_method_full_stds, 0), 
        our_method_full_means + our_method_full_stds, 
        color=(colors[fo_n], band_opacity))
        
    # Scatter plot for individual our method full state data points
    for col in eachindex(our_method_full_individual_points[1, :])
        for row in eachindex(σs)
            # Determine if this point is an outlier
            is_outlier = our_method_full_outlier_masks[row][col]
            is_outlier && !show_outliers && continue
            point_opacity = is_outlier ? outlier_point_opacity : data_point_opacity
            
            scatter!(ax_our, [σs[row]], [our_method_full_individual_points[row, col]], 
                    color=(colors[fo_n], point_opacity), markersize=4)
        end
    end
    
    # Plot our method partial state
    partial_state_included && lines!(ax_our, σs, our_method_partial_means, color=colors[po_n], linestyle=:dash, label="Partial State")
    partial_state_included && band!(ax_our, σs, 
        max.(our_method_partial_means - our_method_partial_stds, 0), 
        our_method_partial_means + our_method_partial_stds, 
        color=(colors[po_n], band_opacity))
        
    # Scatter plot for individual our method partial state data points
    if partial_state_included
        for col in eachindex(our_method_partial_individual_points[1, :])
            for row in eachindex(σs)
                # Determine if this point is an outlier
                is_outlier = our_method_partial_outlier_masks[row][col]
                is_outlier && !show_outliers && continue
                point_opacity = is_outlier ? outlier_point_opacity : data_point_opacity
                
                scatter!(ax_our, [σs[row]], [our_method_partial_individual_points[row, col]], 
                        color=(colors[po_n], point_opacity), markersize=4)
            end
        end
    end
    
    axislegend(ax_our, position=:lt)
    
    # Save the our method comparison figure
    save(prefix*"/"*"po_fo_our_ind.png", fig_our)
    save(pdf_dir*"/"*"po_fo_our_ind.pdf", fig_our, pt_per_unit=1, pt_per_inch=72)
    
    # Create figure for full state methods comparison
    fig_full = Figure(size = (800, 600), margins = (10, 10, 10, 10))
    ax_full = Axis(fig_full[1, 1],
        xlabel = "Noise Level (σ)",
        ylabel = "Trajectory Error",
        title = "Full State: Baseline vs Our Method",
        limits = (nothing, (0, y_axis_limit))
    )
    
    # Plot baseline with full state
    lines!(ax_full, σs, baseline_full_means, color=colors[fo_b], label="Baseline")
    band!(ax_full, σs, 
        max.(baseline_full_means - baseline_full_stds, 0), 
        baseline_full_means + baseline_full_stds, 
        color=(colors[fo_b], band_opacity))
    
    # Scatter plot for individual baseline full state data points
    for col in eachindex(baseline_full_individual_points[1, :])
        for row in eachindex(σs)
            # Determine if this point is an outlier
            is_outlier = baseline_full_outlier_masks[row][col]
            is_outlier && !show_outliers && continue
            point_opacity = is_outlier ? outlier_point_opacity : data_point_opacity
            
            scatter!(ax_full, [σs[row]], [baseline_full_individual_points[row, col]], 
                    color=(colors[fo_b], point_opacity), markersize=4)
        end
    end
    
    # Plot our method with full state
    lines!(ax_full, σs, our_method_full_means, color=colors[fo_n], label="Our Method")
    band!(ax_full, σs, 
        max.(our_method_full_means - our_method_full_stds, 0), 
        our_method_full_means + our_method_full_stds, 
        color=(colors[fo_n], band_opacity+0.1))
    
    # Scatter plot for individual our method full state data points
    for col in eachindex(our_method_full_individual_points[1, :])
        for row in eachindex(σs)
            # Determine if this point is an outlier
            is_outlier = our_method_full_outlier_masks[row][col]
            is_outlier && !show_outliers && continue
            point_opacity = is_outlier ? outlier_point_opacity : data_point_opacity
            
            scatter!(ax_full, [σs[row]], [our_method_full_individual_points[row, col]], 
                    color=(colors[fo_n], point_opacity), markersize=4)
        end
    end
    
    axislegend(ax_full, position=:lt)
    
    # Save the full state comparison figure
    save(prefix*"/"*"fo_both_ind.png", fig_full)
    save(pdf_dir*"/"*"fo_both_ind.pdf", fig_full, pt_per_unit=1, pt_per_inch=72)
    
    # Create figure for partial state methods comparison (only if partial state data included)
    if partial_state_included
        fig_partial = Figure(size = (800, 600), margins = (10, 10, 10, 10))
        ax_partial = Axis(fig_partial[1, 1],
            xlabel = "Noise Level (σ)",
            ylabel = "Trajectory Error",
            title = "Partial State: Baseline vs Our Method",
            limits = (nothing, (0, y_axis_limit))
        )
        
        # Plot baseline with partial state
        lines!(ax_partial, σs, baseline_partial_means, color=colors[po_b], label="Baseline")
        band!(ax_partial, σs, 
            max.(baseline_partial_means - baseline_partial_stds, 0), 
            baseline_partial_means + baseline_partial_stds, 
            color=(colors[po_b], band_opacity))
        
        # Scatter plot for individual baseline partial state data points
        for col in eachindex(baseline_partial_individual_points[1, :])
            for row in eachindex(σs)
                # Determine if this point is an outlier
                is_outlier = baseline_partial_outlier_masks[row][col]
                is_outlier && !show_outliers && continue
                point_opacity = is_outlier ? outlier_point_opacity : data_point_opacity
                
                scatter!(ax_partial, [σs[row]], [baseline_partial_individual_points[row, col]], 
                        color=(colors[po_b], point_opacity), markersize=4)
            end
        end
        
        # Plot our method with partial state
        lines!(ax_partial, σs, our_method_partial_means, color=colors[po_n], label="Our Method")
        band!(ax_partial, σs, 
            max.(our_method_partial_means - our_method_partial_stds, 0), 
            our_method_partial_means + our_method_partial_stds, 
            color=(colors[po_n], band_opacity))
        
        # Scatter plot for individual our method partial state data points
        for col in eachindex(our_method_partial_individual_points[1, :])
            for row in eachindex(σs)
                # Determine if this point is an outlier
                is_outlier = our_method_partial_outlier_masks[row][col]
                is_outlier && !show_outliers && continue
                point_opacity = is_outlier ? outlier_point_opacity : data_point_opacity
                
                scatter!(ax_partial, [σs[row]], [our_method_partial_individual_points[row, col]], 
                        color=(colors[po_n], point_opacity), markersize=4)
            end
        end
        
        axislegend(ax_partial, position=:lt)
        
        # Save the partial state comparison figure
        save(prefix*"/"*"po_both_ind.png", fig_partial)
        save(pdf_dir*"/"*"po_both_ind.pdf", fig_partial, pt_per_unit=1, pt_per_inch=72)
    end
    
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
            @warn "All data points were considered outliers at noise level $(σs[noise_level]). Using original data."
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
    colors = [[(:red, 1.0), (:blue, 1.0), (:green, 1.0), (:yellow, 1.0)], [(:red, 0.75), (:blue, 0.75), (:green, 0.75), (:yellow, 0.75)]],
    constraints = nothing,
    lane_centers = nothing
    # TODO automatically generate default colors based on number of players?
)
    CairoMakie.activate!()
    fig = CairoMakie.Figure()
    ax = CairoMakie.Axis(fig[1,1], aspect = DataAspect())

    # image_data = CairoMakie.load("experiments/data/07_background.png")
    # image_data = image_data[end:-1:1, :]
    # image_data = image_data'
    # # ax1 = Axis(fig[1,1], aspect = DataAspect())
    # trfm = ImageTransformations.recenter(Rotations.RotMatrix(-2.303611),center(image_data))
    # x_crop_min = 430
    # x_crop_max = 875
    # y_crop_min = 225
    # y_crop_max = 1025
    
    # scale = 1/10.25

    # x = (x_crop_max - x_crop_min) * scale
    # y = (y_crop_max - y_crop_min) * scale

    # image_data = ImageTransformations.warp(image_data, trfm)
    # image_data = Origin(0)(image_data)
    # image_data = image_data[x_crop_min:x_crop_max, y_crop_min:y_crop_max]
    
    # x_offset = -34.75
    # y_offset = 22

    # CairoMakie.image!(ax,
    #     x_offset..(x+x_offset),
    #     y_offset..(y+y_offset),
    #     image_data)

    #TODO horizon can probably be calculated from trajectory
    #TODO same with num_players/player state_dim (in reconstruct_solution)
    # Assumes first two elements in each state vector is x, y position

    

    n = num_players(game_structure.game.dynamics)
    p_state_dim = Int64(state_dim(game_structure.game.dynamics) // n)

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
    if lane_centers !== nothing
        for i in eachindex(lane_centers)
            # min_x = minimum([trajectories[1][t][Block(i)][1] for t in eachindex(trajectories[1])])
            # max_x = maximum([trajectories[1][t][Block(i)][2] for t in eachindex(trajectories[1])])
            # xs = collect(min_x:10:max_x)
            # println(xs)
            # println([lane_centers[i](x) for x in xs])
            ys = [lane_centers[i](x) for x in [trajectories[1][t][Block(i+2)][1] for t in eachindex(trajectories[1])]]
            CairoMakie.lines!(ax,
                [trajectories[1][t][Block(i+2)][1] for t in eachindex(trajectories[1])],
                ys,
                color = :black, linestyle=:dash)
        end
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
    constraints = nothing
)
    CairoMakie.activate!()
    fig = CairoMakie.Figure()
    ax = CairoMakie.Axis(fig[1,1], aspect = DataAspect())
    n = num_players(game_structure.game.dynamics)
    p_state_dim = Int64(state_dim(game_structure.game.dynamics) // n)

    for i in 1:n
        CairoMakie.lines!(ax, 
            [trajectories[1][Block(t)][(i - 1) * p_state_dim + 1] for t in eachindex(blocks(trajectories[1]))],
            [trajectories[1][Block(t)][(i - 1) * p_state_dim + 2] for t in eachindex(blocks(trajectories[1]))], 
            color = colors[1][i])
        CairoMakie.lines!(ax, 
            [trajectories[2][Block(t)][(i - 1) * p_state_dim + 1] for t in eachindex(blocks(trajectories[2]))],
            [trajectories[2][Block(t)][(i - 1) * p_state_dim + 2] for t in eachindex(blocks(trajectories[2]))], 
            color = colors[2][i])
        CairoMakie.scatter!(ax,
            [trajectories[1][Block(length(blocks(trajectories[1])))][(i - 1) * p_state_dim + 1]],
            [trajectories[1][Block(length(blocks(trajectories[1])))][(i - 1) * p_state_dim + 2]],
            color = colors[1][i], marker=:star5)
        CairoMakie.scatter!(ax,
            [trajectories[2][Block(length(blocks(trajectories[2])))][(i - 1) * p_state_dim + 1]],
            [trajectories[2][Block(length(blocks(trajectories[2])))][(i - 1) * p_state_dim + 2]],
            color = colors[2][i], marker=:star5)
    end
    CairoMakie.save(plot_name*"_tmp.png", fig)
end


end