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

using Infiltrator

include("../../src/utils/Utils.jl")

function graph_metrics(
    baseline_errors,
    errors, 
    baseline_parameter_error,
    parameter_error,
    σs;
    observation_mode="full state"
) #TODO needs cleanup

    prefix = "experiments/crosswalk_sim/"*observation_mode*"/"
    CairoMakie.activate()

    fig1 = CairoMakie.Figure()
    ax1 = CairoMakie.Axis(fig1[1, 1])

    mean_errors = [Statistics.mean(errors[i, :]) for i in 1:size(errors, 1)]
    stds = [Statistics.std(errors[i, :]) for i in 1:size(errors, 1)]

    baseline_mean_errors = [Statistics.mean(baseline_errors[i, :]) for i in 1:size(baseline_errors, 1)]
    baseline_stds = [Statistics.std(baseline_errors[i, :]) for i in 1:size(baseline_errors, 1)]

    our_method = CairoMakie.scatter!(ax1, σs, mean_errors, color = (:blue, 0.75))
    CairoMakie.errorbars!(ax1, σs, mean_errors, stds, color = (:blue, 0.75))

    baseline = CairoMakie.scatter!(ax1, σs, baseline_mean_errors, color = (:red, 0.75))
    CairoMakie.errorbars!(ax1, σs, baseline_mean_errors, baseline_stds, color = (:red, 0.75))

    CairoMakie.axislegend(ax1, [our_method, baseline], ["Our Method", "Baseline"], position = :lt)
    CairoMakie.save(prefix*"TrajectoryErrorGraph.png", fig1)


    fig2 = CairoMakie.Figure()
    ax2 = CairoMakie.Axis(fig2[1, 1])

    mean_parameter_errors = [Statistics.mean(parameter_error[i, :]) for i in 1:size(parameter_error, 1)]
    parameter_stds = [Statistics.std(parameter_error[i, :]) for i in 1:size(parameter_error, 1)]

    baseline_mean_parameter_errors = [Statistics.mean(baseline_parameter_error[i, :]) for i in 1:size(baseline_parameter_error, 1)]
    baseline_parameter_stds = [Statistics.std(baseline_parameter_error[i, :]) for i in 1:size(baseline_parameter_error, 1)]

    our_method = CairoMakie.scatter!(ax2, σs, mean_parameter_errors, color = (:blue, 0.75))
    CairoMakie.errorbars!(ax2, σs, mean_parameter_errors, parameter_stds, color = (:blue, 0.75))

    baseline = CairoMakie.scatter!(ax2, σs, baseline_mean_parameter_errors, color = (:red, 0.75))
    CairoMakie.errorbars!(ax2, σs, baseline_mean_parameter_errors, baseline_parameter_stds, color = (:red, 0.75))

    CairoMakie.axislegend(ax2, [our_method, baseline], ["Our Method", "Baseline"], position = :lt)
    CairoMakie.save(prefix*"ParameterErrorGraph.png", fig2)

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
            @infiltrate
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
    CairoMakie.save(plot_name*".png", fig)
end


end