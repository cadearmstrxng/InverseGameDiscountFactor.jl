using CairoMakie
using ImageTransformations
using Rotations
using CSV
using LinearAlgebra: norm
using BlockArrays
using OffsetArrays:Origin
# include("../../src/InverseGameDiscountFactor.jl")

function rotate_point(theta, point)
    R = [cos(theta) -sin(theta); sin(theta) cos(theta)]
    R * point
end


function create_env()
    CairoMakie.activate!();
    fig = CairoMakie.Figure()
    image_data = CairoMakie.load("experiments/data/07_background.png")
    image_data = image_data[end:-1:1, :]
    image_data = image_data'
    ax1 = Axis(fig[1,1], aspect = DataAspect())
    trfm = ImageTransformations.recenter(Rotations.RotMatrix(-2.303611),center(image_data))


    x_crop_min = 430
    x_crop_max = 875
    y_crop_min = 225
    y_crop_max = 1025
    
    scale = 1/10.25

    x = (x_crop_max - x_crop_min) * scale
    y = (y_crop_max - y_crop_min) * scale

    println(x,' ', y)

    image_data = ImageTransformations.warp(image_data, trfm)
    image_data = Origin(0)(image_data)
    image_data = image_data[x_crop_min:x_crop_max, y_crop_min:y_crop_max]
    
    x_offset = -34.75
    y_offset = 22

    println(x_offset..(x+x_offset-2), ' ', y_offset..(y-2+y_offset))

    image!(ax1,
        x_offset..(x+x_offset),
        y_offset..(y+y_offset),
        image_data)

    traj = pull_trajectory("07"; track = [17, 19, 22])
    colors = [:red, :blue, :green]
    for j in eachindex(traj)
        lines!(ax1, [i[1] for i in blocks(traj[j])], [i[2] for i in blocks(traj[j])], color = colors[(j % length(colors)) + 1])
    end

    fig
end

function pull_trajectory(recording; dir = "experiments/data/", track = [1, 2, 3], all = false)
    file = CSV.File(dir*recording*"_tracks.csv")
    raw_trajectories = (all) ? [[] for _ in 1:max(file[:trackId]...)+1] : [[] for _ in eachindex(track)]
    data_to_pull = [:xCenter, :yCenter, :heading, :xVelocity, :yVelocity, :xAcceleration, :yAcceleration, :width, :length,]
    for row in file
        idx = (all) ? row.trackId+1 : findfirst(x -> x == row.trackId, track)
        if !isnothing(idx)
            raw_state = [row[i] for i in data_to_pull]
            full_state = [
                rotate_point(2.303611, raw_state[1:2]) # + [390.5, 585.5]/10
                norm(raw_state[4:5])
                (deg2rad(raw_state[3]) + 5.445203653589793) % (2 * pi)
                ]
            push!(raw_trajectories[idx], full_state)
        end
    end

    trajectories = [BlockVector(vcat(raw_trajectories[i]...), [4 for _ in eachindex(raw_trajectories[i])]) for i in eachindex(raw_trajectories)]
    return trajectories
end