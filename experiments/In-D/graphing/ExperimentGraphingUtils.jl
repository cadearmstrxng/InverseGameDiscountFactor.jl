function graph_trajectories(
    title,
    trajectories,
    game_structure,
    horizon;
    colors = [[(:red, 1.0), (:blue, 1.0)]],  # Default colors for 2 tracks
    constraints = nothing,
    background = "experiments/data/07_background.png"
)
    CairoMakie.activate!()
    fig = CairoMakie.Figure()

    if !isnothing(background)
        image_data = CairoMakie.load(background)
        image_data = image_data[end:-1:1, :]
        image_data = image_data'
        ax1 = Axis(fig[1,1], aspect = DataAspect())
        trfm = ImageTransformations.recenter(Rotations.RotMatrix(-2.303611), center(image_data))

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

        image!(ax1,
            x_offset..(x+x_offset),
            y_offset..(y+y_offset),
            image_data)
    else
        ax1 = Axis(fig[1,1], aspect = DataAspect())
    end

    n_players = num_players(game_structure)
    
    for (traj_idx, traj) in enumerate(trajectories)
        for player in 1:n_players
            player_color = colors[traj_idx][player]
            xs = [traj[t][Block(player)][1] for t in 1:horizon]
            ys = [traj[t][Block(player)][2] for t in 1:horizon]
            
            # Only plot if we have valid coordinates
            if any(isfinite.(xs)) && any(isfinite.(ys))
                lines!(ax1, xs, ys, color = player_color)
                scatter!(ax1, [xs[end]], [ys[end]], color = player_color, marker = :star5)
            end
        end
    end

    if !isnothing(constraints)
        # Add constraint visualization if needed
        # ... (existing constraint visualization code)
    end

    CairoMakie.display(fig)
end