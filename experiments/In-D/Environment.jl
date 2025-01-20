using CairoMakie
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
    fig = CairoMakie.Figure(resolution = (620,641))
    image_data = CairoMakie.load("C:\\Users\\Owner\\Documents\\Research Summer 2024\\InverseGameDiscountFactor.jl\\data\\inD-dataset-v1\\data\\09_background.png")
    ax1 = Axis(fig[1,1], aspect = DataAspect())
    image_data = image!(ax1,image_data; interpolate = false)
    image_data = CairoMakie.rotate!(image_data, pi/2)


    CairoMakie.display(fig)
end