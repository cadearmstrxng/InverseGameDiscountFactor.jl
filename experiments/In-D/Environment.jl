using CairoMakie
using ImageTransformations
using Rotations
using CSV
using LinearAlgebra: norm
using BlockArrays
# include("../../src/InverseGameDiscountFactor.jl")

function rotate_point(theta, point)
    R = [cos(theta) -sin(theta); sin(theta) cos(theta)]
    R * point
end

function rotate_matrix(theta, matrix)
    R = [cos(theta) -sin(theta); sin(theta) cos(theta)]
    R * matrix
end


function create_env()
    CairoMakie.activate!();
    fig = CairoMakie.Figure()
    image_data = CairoMakie.load("./data/09_background.png")
    ax1 = Axis(fig[1,1], aspect = DataAspect())
    trfm = ImageTransformations.recenter(Rotations.RotMatrix(2.303611),center(image_data))


    x_crop_min = 200
    x_crop_max = 700
    y_crop_min = 200
    y_crop_max = 1100
    scale = 1/8.5

    image_data = ImageTransformations.warp(image_data, trfm)
    image_data = image_data[x_crop_min:x_crop_max, y_crop_min:y_crop_max]
    image_data = ImageTransformations.imresize(image_data, ratio = scale)

    x_size, y_size = size(image_data)
    x_offset = -35
    y_offset = 15

    image!(ax1,
        x_offset..(x_offset + x_size),
        y_offset..(y_offset + y_size),
        image_data)

    traj = pull_trajectory("07"; track = [17, 19, 22])
    colors = [:red, :blue, :green]
    for j in eachindex(traj)
        lines!(ax1, [i[1] for i in blocks(traj[j])], [i[2] for i in blocks(traj[j])], color = colors[(j % length(colors)) + 1])
    end

    fig
end

function pull_trajectory(recording; dir = "./data/", track = [1, 2, 3], all = false)
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
                (deg2rad(raw_state[3]) + 2.303611) % (2 * pi)
                ]
            push!(raw_trajectories[idx], full_state)
        end
    end

    trajectories = [BlockVector(vcat(raw_trajectories[i]...), [4 for _ in eachindex(raw_trajectories[i])]) for i in eachindex(raw_trajectories)]
    return trajectories
end