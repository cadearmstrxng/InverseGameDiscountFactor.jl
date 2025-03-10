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
    observation_mode="full state",
    pre_prefix = "./experiments/In-D/",
)
    prefix = pre_prefix
    CairoMakie.activate!()

    # Read the CSV files
    baseline_data = CSV.read(baseline_file, DataFrame)
    our_method_data = CSV.read(our_method_file, DataFrame)

    # Extract noise levels, strip 'b' prefix, and convert to numeric values
    σs = parse.(Float64, replace.(baseline_data[:, 1], "b" => ""))
    
    # Calculate means and standard deviations
    baseline_means = mean(Matrix(baseline_data[:, 2:end]), dims=2)[:, 1]
    baseline_stds = std(Matrix(baseline_data[:, 2:end]), dims=2)[:, 1]
    
    our_method_means = mean(Matrix(our_method_data[:, 2:end]), dims=2)[:, 1]
    our_method_stds = std(Matrix(our_method_data[:, 2:end]), dims=2)[:, 1]

    # Create the figure
    fig = Figure()
    ax = Axis(fig[1, 1],
        xlabel = "Noise Level (σ)",
        ylabel = "Trajectory Error",
        title = "Trajectory Error vs Noise Level"
    )

    # Plot baseline with non-negative lower bound
    lines!(ax, σs, baseline_means, color=:red, label="Baseline")
    band!(ax, σs, 
          max.(baseline_means - baseline_stds, 0), 
          baseline_means + baseline_stds, 
          color=(:red, 0.2), label="Baseline ±1σ")

    # Plot our method with non-negative lower bound
    lines!(ax, σs, our_method_means, color=:blue, label="Our Method")
    band!(ax, σs, 
          max.(our_method_means - our_method_stds, 0), 
          our_method_means + our_method_stds, 
          color=(:blue, 0.2), label="Our Method ±1σ")

    # Add legend
    axislegend(ax, position=:lt)

    # Save the figure
    save(prefix*"/"*observation_mode*"_TrajectoryErrorGraph.png", fig)
end

function graph_trajectories(
    plot_name,
    trajectories,
    game_structure,
    horizon;
    colors = [[(:red, 1.0), (:blue, 1.0), (:green, 1.0)], [(:red, 0.75), (:blue, 0.75), (:green, 0.75)]],
    constraints = nothing
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