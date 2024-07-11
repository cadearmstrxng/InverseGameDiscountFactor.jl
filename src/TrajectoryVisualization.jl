module TrajectoryVisualization
import Vegalite

export visualize_trajectory, visualize_trajectory_batch


function trajectory_data end

trajectory_data(control_system, x, player)


function visualize_trajectory(control_system, x; kwargs...)
    td = trajectory_data(control_system, x)
    visualize_trajectory(td; kwargs...)
end

function visualize_trajectory(
    trajectory_data;
    canvas = VegaLite.@vlplot(),
    x_position_domain = extrema(s.px for s in trajectory_data) .+ (-0.01, 0.01),
    y_position_domain = extrema(s.py for s in trajectory_data) .+ (-0.01, 0.01),
    draw_line = true,
    legend = nothing,
)   
    trajectory_visualizer =
        VegaLite.@vlplot(
            encoding = {
                x = {"px:q", scale = {domain = x_position_domain}, title = "Position x [m]"},
                y = {"py:q", scale = {domain = y_position_domain}, title = "Position y [m]"},
                order = "t:q",
                color = {"player:n", title = "Player", legend = legend},
            }
        ) + 
        VegaLite.@vlplot(mark = {"point", shape = "circle", size = 25, clip = true, filled = true})

    if draw_line
        trajectory_visualizer += VegaLite.@vlplot(mark = {"line", clip = true})
    end

    canvas + (trajectory_data |> trajectory_visualizer)
end

function visualize_trajectory_batch(
    control_system,
    trajectory_batch;
    canvas = VegaLite.@vlplot(opacity = {value = 0.2}, width = 200, height = 200),
    kwargs...,
)
    mapreduce(+, trajectory_batch; init = canvas) do x
        visualize_trajectory(control_system, x; kwargs...)
    end
end

function visualize_trajectory_batch(
    trajectory_data_batch;
    canvas = VegaLite.@vlplot(opacity = {value = 0.2}, width = 200, height = 200),
    kwargs...,
)
    mapreduce(+, trajectory_data_batch; init = canvas) do trajectory_data
        visualize_trajectory(trajectory_data; kwargs...)
    end
end

end