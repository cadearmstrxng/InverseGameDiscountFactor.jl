using CairoMakie
using ImageTransformations
using Rotations
using CSV
using LinearAlgebra: norm, inv, norm_sqr
using BlockArrays
using OffsetArrays:Origin
using Infiltrator
using Symbolics: Symbolics, @variables, scalarize
using TrajectoryGamesBase:
    TrajectoryGamesBase,
    get_constraints
import TrajectoryGamesBase: get_constraints
# include("../../src/InverseGameDiscountFactor.jl")

struct indEnvironment{}
    circle_centers::Vector{}
    circle_radii::Vector{}
    line_slopes::Vector{}
    line_intercepts::Vector{}
    line_x_ranges::Vector{}
    line_y_ranges::Vector{}
    points
end


function TrajectoryGamesBase.get_constraints(environment::indEnvironment, player_index = nothing)
    get_constraints(environment, player_index)
end
function get_constraints(environment::indEnvironment, player_index = nothing)
    function (state)
        position = state[1:2]
        constraints = []
        for i in eachindex(environment.circle_centers)
            c = environment.circle_centers[i]
            r = environment.circle_radii[i]
            # @infiltrate
            push!(constraints,
                (norm_sqr(position - [c[1], c[2]]) - r^2)
            )
        end
        sigmoid = (m1, b1, m2, b2, m3, b3) -> (x) -> (1/(2+exp(m1*x[1] - x[2] + b1))) * (1/exp(m2*x[1] - x[2] + b2)) * (1/exp(m3*x[1] - x[2] + b3)) - 0.01
        for i in 1:2:length(environment.line_slopes)
            m1 = environment.line_slopes[i]
            b1 = environment.line_intercepts[i]
            m2 = environment.line_slopes[i+1]
            b2 = environment.line_intercepts[i+1]
            m3, b3 = solve_line(environment.points[i][2], environment.points[i+1][2])
            push!(constraints, sigmoid(m1, b1, m2, b2, m3, b3))
        end

        # push!(constraints, (position[1] > environment.line_x_ranges[1][1]) ? ((position[1] < environment.line_x_ranges[1][2]) ? environment.line_intercepts[1] - position[2] : 0) : 0)
        # push!(constraints, (position[1] > environment.line_x_ranges[2][1]) ? ((position[1] < environment.line_x_ranges[2][2]) ? position[2] - environment.line_intercepts[2] : 0) : 0)
        # push!(constraints, (position[1] > environment.line_x_ranges[3][1]) ? ((position[1] < environment.line_x_ranges[3][2]) ? environment.line_slopes[3]*position[1] + environment.line_intercepts[3] - position[2] : 0) : 0)
        # push!(constraints, (position[1] > environment.line_x_ranges[4][1]) ? ((position[1] < environment.line_x_ranges[4][2]) ? -environment.line_slopes[4]*position[1] - environment.line_intercepts[4] + position[2] : 0) : 0)
        # push!(constraints, (position[1] > environment.line_x_ranges[5][1]) ? ((position[1] < environment.line_x_ranges[5][2]) ? environment.line_slopes[5]*position[1] + environment.line_intercepts[5] - position[2] : 0) : 0)
        # push!(constraints, (position[1] > environment.line_x_ranges[6][1]) ? ((position[1] < environment.line_x_ranges[6][2]) ? -environment.line_slopes[6]*position[1] - environment.line_intercepts[6] + position[2] : 0) : 0)
        # push!(constraints, (position[1] > environment.line_x_ranges[7][1]) ? ((position[1] < environment.line_x_ranges[7][2]) ? -environment.line_slopes[7]*position[1] - environment.line_intercepts[7] + position[2] : 0) : 0)
        # push!(constraints, (position[1] > environment.line_x_ranges[8][1]) ? ((position[1] < environment.line_x_ranges[8][2]) ? environment.line_slopes[8]*position[1] + environment.line_intercepts[8] - position[2] : 0) : 0)

        constraints
    end
end

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

    # println(x,' ', y)

    image_data = ImageTransformations.warp(image_data, trfm)
    image_data = Origin(0)(image_data)
    image_data = image_data[x_crop_min:x_crop_max, y_crop_min:y_crop_max]
    
    x_offset = -34.75
    y_offset = 22

    # println(x_offset..(x+x_offset-2), ' ', y_offset..(y-2+y_offset))

    # image!(ax1,
    #     x_offset..(x+x_offset),
    #     y_offset..(y+y_offset),
    #     image_data)

    # traj = pull_trajectory("07"; track = [17, 19, 22])
    # colors = [:red, :blue, :green]
    # for j in eachindex(traj)
    #     lines!(ax1, [i[1] for i in blocks(traj[j])], [i[2] for i in blocks(traj[j])], color = colors[(j % length(colors)) + 1])
    # end

    # x_points = [i for i in x_offset:1:x+x_offset]
    # y_points = [i for i in y_offset:1:y+y_offset]

    # for i in x_points
    #     for j in y_points
    #         scatter!(ax1, [i], [j], color = :yellow, markersize = 2)
    #     end
    # end

    # Upper Left Circle

    p1 = [-18.5 69]
    p2 = [-16.5 72.5]
    p3 = [-22 68]

    c1, r1 = solve_circle(p1, p2, p3)
    plot_circle(ax1, c1, r1)

    # Upper Right Circle

    p1 = [-7 74]
    p2 = [-3 69]
    p3 = [-6 70.5]

    c2, r2 = solve_circle(p1, p2, p3)
    plot_circle(ax1, c2, r2)

    # Lower Left Circle

    p1 = [-27.5 58.5]
    p2 = [-22.5 57]
    p3 = [-19.5 52]

    c3, r3 = solve_circle(p1, p2, p3)
    r3 = 7.55
    # println(c3, ' ', r3)
    plot_circle(ax1, c3, r3)

    # Lower Right Circle

    p1 = [-3 58]
    p2 = [-7.5 56.5]
    p3 = [-9.5 52]

    c4, r4 = solve_circle(p1, p2, p3)
    plot_circle(ax1, c4, r4)

    # LINE EQUATIONS


    points = []
    

    # -34.75 <= x <= -22
    p1 = [-33.5 68]
    p2 = [-22 68]
    push!(points, [p1, p2])

    m1, b1 = solve_line(p1, p2)
    plot_line(ax1, m1, b1, p1[1], p2[1], :blue)

    # y > 72.5, -16.5 <= x <= -13.5

    p1 = [-16.5 72.5]
    p2 = [-13.5 99.5]
    push!(points, [p1, p2])

    m2, b2 = solve_line(p1, p2)
    plot_line(ax1, m2, b2, p1[1], p2[1], :green)

    #  68.5< y < 69, -3 <= x <= 8.5

    p1 = [-3 69]
    p2 = [8.5 68.5]
    push!(points, [p1, p2])

    m3, b3 = solve_line(p1, p2)
    plot_line(ax1, m3, b3, p1[1], p2[1], :white)


    # y > 74, -7 <= x <= -4.5

    p1 = [-7 74]
    p2 = [-4.5 99]
    push!(points, [p1, p2])

    m4, b4 = solve_line(p1, p2)
    plot_line(ax1, m4, b4, p1[1], p2[1], :red)


    # y > 58, -3 <= x <= 7

    p1 = [-3 58]
    p2 = [7 58]
    push!(points, [p1, p2])

    m5, b5 = solve_line(p1, p2)
    plot_line(ax1, m5, b5, p1[1], p2[1], :purple)

    # y < 52, -9.5 <= x <= -12.5

    p1 = [-9.5 52]
    p2 = [-12.5 22.5]
    push!(points, [p1, p2])

    m6, b6 = solve_line(p1, p2)
    plot_line(ax1, m6, b6, p2[1], p1[1], :black)

    
    # -34.75 <= x <= -29
    p1 = [-33.5 58.5]
    p2 = [-27.5 58.5]
    push!(points, [p1, p2])

    m7, b7 = solve_line(p1, p2)
    plot_line(ax1, m7, b7, p1[1], p2[1], :yellow)

    # y < 52, -21.5 <= x <= -19.5

    p1 = [-19.5 51.5]
    p2 = [-21.5 24]
    push!(points, [p1, p2])

    m8, b8 = solve_line(p1, p2)
    plot_line(ax1, m8, b8, p2[1], p1[1], :orange)
    

    # display(fig)

    circle_centers = [c1, c2, c3, c4]
    circle_radii = [r1, r2, r3, r4]
    line_slopes = [m1, m2, m3, m4, m5, m6, m7, m8]
    line_intercepts = [b1, b2, b3, b4, b5, b6, b7, b8]
    line_x_ranges = [(-33.5, -27.5), (-32.5, -22), (-16.5, -13.5), (-7, -4.5), (-3, 8.5), (-3, 7), (-21.5, -19.5), (-9.5, -12.5)]
    line_y_ranges = [(58.5, 58.5), (68, 68), (72.5, 99.5), (74, 99), (69, 68.5), (58, 58), (51.5, 24), (52, 22.5)]

    return indEnvironment(circle_centers, circle_radii, line_slopes, line_intercepts, line_x_ranges, line_y_ranges, points)
end

function pull_trajectory(recording; dir = "experiments/data/", track = [1, 2, 3], all = false, frames = [0, 10000])
    file = CSV.File(dir*recording*"_tracks.csv")
    raw_trajectories = (all) ? [[] for _ in 1:max(file[:trackId]...)+1] : [[] for _ in eachindex(track)]
    data_to_pull = [:xCenter, :yCenter, :heading, :xVelocity, :yVelocity, :xAcceleration, :yAcceleration, :width, :length,]
    for row in file
        idx = (all) ? row.trackId+1 : findfirst(x -> x == row.trackId, track)
        if !isnothing(idx)
            raw_state = [row[i] for i in data_to_pull]
            if row.frame < frames[1] || row.frame > frames[2]
                continue
            end
            full_state = [
                rotate_point(2.303611, raw_state[1:2]) # + [390.5, 585.5]/10
                norm(raw_state[4:5])
                (deg2rad(raw_state[3]) + 5.445203653589793) % (2 * pi)
                ]
            push!(raw_trajectories[idx], full_state)
        end
    end

    traj = []
    @infiltrate
    for t in eachindex(raw_trajectories[1])
        b = BlockVector(vcat([raw_trajectories[i][t] for i in eachindex(raw_trajectories)]...),
        [4 for _ in eachindex(raw_trajectories)])
        push!(traj, b)
    end
    return traj

    # return [BlockVector(vcat([raw_trajectories[i][t] for i in eachindex(raw_trajectories)]...),
    #             [4 for _ in eachindex(raw_trajectories)]) for t in eachindex(raw_trajectories[1])]

    # trajectories = [BlockVector(vcat(raw_trajectories[1]), [4 for _ in eachindex(raw_trajectories[i])]) for i in eachindex(raw_trajectories)]
    # return trajectories
end

function solve_circle(p1,p2,p3)

    A = [p1[1] p1[2] 1; p2[1] p2[2] 1; p3[1] p3[2] 1]

    b = [-(p1[1]^2 + p1[2]^2),
         -(p2[1]^2 + p2[2]^2),
         -(p3[1]^2 + p3[2]^2)]
    x = A\b

    h = -x[1]/2
    k = -x[2]/2
    r = sqrt(h^2 + k^2 - x[3])

    return (h, k), r
    
end

function plot_circle(ax, center, r)
    theta = LinRange(0, 2*pi, 100)
    x = r*cos.(theta) .+ center[1]
    y = r*sin.(theta) .+ center[2]
    lines!(ax, x, y, color = :yellow)
end

function solve_line(p1, p2)
    m = (p2[2] - p1[2])/(p2[1] - p1[1])
    b = p1[2] - m*p1[1]
    return m, b
end

function plot_line(ax, m, b, x_min, x_max, color = :yellow)
    x = LinRange(x_min, x_max, 100)
    y = m*x .+ b
    lines!(ax, x, y, color = color)
end

function test_env()

    env = create_env()
    test_state = let 
        @variables(test_state[1:4]) |> only |> scalarize
    end 

    constraints = get_constraints(env)

    constraints(test_state)
end