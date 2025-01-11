module ExperimentGraphicUtils

using Statistics:
    mean, std
using CairoMakie
using TrajectoryGamesBase:
    num_players, state_dim

include("../../src/utils/utils.jl")

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
    colors = [[(:red, 1.0), (:blue, 1.0)], [(:red, 0.75), (:blue, 0.75)], [(:red, 0.5), (:blue, 0.5)], [(:red, 0.25), (:blue, 0.25)]]
    # TODO automatically generate default colors based on number of players?
)

    #TODO horizon can probably be calculated from trajectory
    #TODO same with num_players/player state_dim (in reconstruct_solution)
    # Assumes first two elements in each state vector is x, y position
    flattened_trajectories = [reconstruct_solution(trajectory, game_structure.game, horizon) for trajectory in trajectories]

    p_state_dim = state_dim(game_structure.game) // num_players(game_structure.game)

    fig = CairoMakie.Figure()
    ax = CairoMakie.Axis(fig[1,1])

    for traj in eachindex(trajectories)
        trajectory = trajectories[traj]
        for t in 1:horizon
            for player in 1:num_players(game_structure.game)
                CairoMakie.scatter!(
                    ax,
                    trajectory[Block(t)][p_state_dim*(player-1)+1],
                    trajectory[Block(t)][p_state_dim*(player-1)+2],
                    color = colors[traj][player])
            end
        end
    end
    CairoMakie.save("./graphs"*plot_name, fig)
end


end