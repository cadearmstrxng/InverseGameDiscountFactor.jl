using CairoMakie
using FileIO
using ColorTypes, ImageCore

"""
    plot_background_with_equations(; equations=[], resolution=(1000, 1000), show_plot=true, 
                                  remove_black=true, black_threshold=0.1)

Plot the background image with optional equation overlays.
- `equations`: Array of functions that take (x,y) coordinates and return a value
- `resolution`: Resolution of the output plot
- `show_plot`: Whether to display the plot
- `remove_black`: Whether to make black pixels transparent
- `black_threshold`: Threshold below which pixels are considered black (0-1 range)

Returns the figure and axis objects.
"""
function plot_background_with_equations(; equations=[], resolution=(1000, 1000), show_plot=true,
                                    remove_black=true, black_threshold=0.1)
    # Create figure
    fig = Figure(resolution=resolution)
    ax = Axis(fig[1, 1], aspect=DataAspect())
    
    # Load background image
    img = load("./src/00_background.png")
    
    # Process image to remove black background
    if remove_black
        # Convert image to RGBA if it's not already
        img_rgba = RGBA.(img)
        
        # Make black pixels transparent
        img = map(img_rgba) do pixel
            # Calculate brightness (simple average of RGB)
            brightness = (red(pixel) + green(pixel) + blue(pixel)) / 3
            
            # If pixel is dark (black), make it transparent
            if brightness < black_threshold
                return RGBA(red(pixel), green(pixel), blue(pixel), 0.0)
            else
                return pixel
            end
        end
    end
        
    img_plot = image!(ax, rotr90(img))
    
    # Get image dimensions for proper scaling
    img_height, img_width = size(img)[1:2]
    xlims!(ax, 0, img_width)
    ylims!(ax, 0, img_height)
    
    # Generate coordinates for equation evaluation
    xmin = 400;
    xmax = 1125;
    ymin = 5;
    ymax = 936;
    step = 2;
    xs = xmin:step:xmax
    ys = ymin:step:ymax
    
    # Collect points for plotting with debug info
    inside_points_x = Float64[]
    inside_points_y = Float64[]
    
    # Track which equations are contributing points
    eq_counts = zeros(Int, length(equations))

    for x in xs
        for y in ys
            for (i, eq) in enumerate(equations)
                val = eq(x, y)
                if val <= 0.5  # Using 0.5 for sigmoid threshold
                    push!(inside_points_x, x)
                    push!(inside_points_y, y)
                    eq_counts[i] += 1
                    break  # Only count each point once
                end
            end
        end
    end
    
    # Print which equations are contributing
    println("Points contributed by each equation:")
    for (i, count) in enumerate(eq_counts)
        println("  Equation $i: $count points")
    end
    
    # Plot the points
    println("Found $(length(inside_points_x)) points outside boundaries")
    scatter!(ax, inside_points_x, inside_points_y, color=:red, markersize=1.5, 
            alpha=0.5, overdraw=true)
    
    # Remove axis decorations for cleaner look
    # hidedecorations!(ax)
    # hidespines!(ax)
    
    # Set figure background to transparent (using explicit RGBA instead of symbol)
    fig.scene.backgroundcolor = RGBA(0, 0, 0, 0)  # Fully transparent
    
    show_plot && display(fig)
    save("00_background_with_equations.png", fig)
    return fig, ax
end

"""
    generate_road_equations()

Generate equations for circles, lines, and ellipses based on points extracted from the road image.
Returns an array of functions that take (x,y) coordinates and return values that are:
- Negative inside the shape
- Zero on the boundary
- Positive outside the shape
"""
function generate_road_equations(;circles = true, ellipses = false, lines = true)
    equations = []
    
    if circles
    # Circle 1 (center circle)
    p1 = [835, 470]
    p2 = [760, 410]
    p3 = [730, 470]
    push!(equations, circle_equation(p1, p2, p3))
    
        # Circle 2 (upper left)
        p1 = [730, 605]
        p2 = [700, 515]
        p3 = [595, 485]
        push!(equations, circle_equation(p1, p2, p3))
        
        # Circle 3 (upper right)
        p1 = [835, 575]
        p2 = [850, 530]
        p3 = [925, 500]
        push!(equations, circle_equation(p1, p2, p3))
        
        # Circle 4 (lower right)
        p1 = [850, 305]
        p2 = [865, 395]
        p3 = [925, 425]
        push!(equations, circle_equation(p1, p2, p3))
        
        # Circle 5 (lower left)
        p1 = [640, 410]
        p2 = [700, 395]
        p3 = [730, 350]
        push!(equations, circle_equation(p1, p2, p3))
    end
    
    if ellipses
        # TODO: make ellipses more accurate, for now they don't work as well as I'd like.
        # Add ellipses from MATLAB code
        # Ellipse 1 (Lower Median)
        ellip1_center = [788.5134, 303.5295]  # X0_in, Y0_in from MATLAB
        ellip1_a = 55.3528                   # Semi-major axis
        ellip1_b = 29.5592                   # Semi-minor axis
        ellip1_phi = -1.0472                 # -phi from MATLAB (convert to radians if needed)
        push!(equations, ellipse_equation(ellip1_center, ellip1_a, ellip1_b, ellip1_phi))
        
        # Ellipse 2 (Right Median)
        ellip2_center = [934.3455, 464.2144]
        ellip2_a = 30.5060
        ellip2_b = 17.9504
        ellip2_phi = -0.1042
        push!(equations, ellipse_equation(ellip2_center, ellip2_a, ellip2_b, ellip2_phi))
        
        # Ellipse 3 (Left Median)
        ellip3_center = [543.2053, 455.7465]
        ellip3_a = 83.2831
        ellip3_b = 13.0020
        ellip3_phi = -0.1294
        push!(equations, ellipse_equation(ellip3_center, ellip3_a, ellip3_b, ellip3_phi))
        
        # Ellipse 4 (Top Median)
        ellip4_center = [753.5862, 754.8981]
        ellip4_a = 167.8549
        ellip4_b = 17.7719
        ellip4_phi = -1.4954
        push!(equations, ellipse_equation(ellip4_center, ellip4_a, ellip4_b, ellip4_phi))
    end
    
    if lines
        # Line 1 (Left, Up)
        p1 = [420, 505]
        p2 = [595, 485]
        # Line 2 (Up, Left)
        p3 = [730, 605]
        p4 = [640, 935]
        #intersector:
        push!(equations, three_line_sigmoid(p1, p2, 1, p3, p4, -1, p2, p3, 1))

        # Line 3 (Up, Right)
        p1 = [800, 795]
        p2 = [835, 575]
        # Line 4 (Right, Up)
        p3 = [925, 500]
        p4 = [1050, 525]
        push!(equations, three_line_sigmoid(p1, p2, 1, p3, p4, 1, p2, p3, 1))
        
        # Line 5 (Right, Lower)
        p1 = [925, 425]
        p2 = [1100, 465]
        
        # Line 6 (Lower, Right)
        p3 = [850, 305]
        p4 = [920, 5]
        push!(equations, three_line_sigmoid(p1, p2, -1, p3, p4, 1, p3, p1, -1))
        
        # Line 7 (Lower, Left)
        p1 = [730, 350]
        p2 = [850, 5]
        
        # Line 8 (Left, Lower)
        p3 = [490, 425]
        p4 = [640, 410]
        push!(equations, three_line_sigmoid(p1, p2, -1, p3, p4, -1, p1, p4, -1))
    end

    return equations
end


"""
    sigmoid(eq)

Apply a sigmoid function to the output of the equation.
"""
function sigmoid(eq::Function)
    return (x, y) -> 1 / (1 + exp(-eq(x, y)))
end

function sigmoid(val::Number; offset = 0.5)
    return 1 / (1 + exp(-val - offset)) 
end

"""
    solve_circle(p1, p2, p3)

Calculate the center (h,k) and radius r of a circle passing through three points.
Returns h, k, r.
"""
function solve_circle(p1, p2, p3)
    A = [p1[1] p1[2] 1; p2[1] p2[2] 1; p3[1] p3[2] 1]
    
    b = [-(p1[1]^2 + p1[2]^2);
         -(p2[1]^2 + p2[2]^2);
         -(p3[1]^2 + p3[2]^2)]
    
    x = A \ b
    
    h = -x[1]/2
    k = -x[2]/2
    r = sqrt(h^2 + k^2 - x[3])
    
    return h, k, r
end

"""
    circle_equation(p1, p2, p3)

Create a function that represents the equation of a circle passing through three points.
The function returns negative values inside the circle, zero on the boundary.
"""
function circle_equation(p1, p2, p3)
    h, k, r = solve_circle(p1, p2, p3)
    
    return (x, y) -> begin
        val = (x - h)^2 + (y - k)^2 - r^2
        return sigmoid(val)
    end
end

"""
    solve_line(p1, p2)

Calculate the slope (m) and y-intercept (b) of a line passing through two points.
Returns m, b.
"""
function solve_line(p1, p2)
    m = (p2[2] - p1[2]) / (p2[1] - p1[1])
    b = p1[2] - m * p1[1]
    
    return m, b
end

function three_line_sigmoid(p1, p2, flip1, p3, p4, flip2, p5, p6, flip3)
    m1, b1 = solve_line(p1, p2)
    m2, b2 = solve_line(p3, p4)
    m3, b3 = solve_line(p5, p6)

    return (x, y) -> sigmoid(flip1*0.01*(m1 * x + b1 - y)) + sigmoid(flip2*0.01*(m2 * x + b2 - y)) + sigmoid(flip3*0.01*(m3 * x + b3 - y))
end

"""
    ellipse_equation(center, a, b, phi)

Create a function that represents the equation of an ellipse with center at `center`,
semi-major axis `a`, semi-minor axis `b`, and rotation angle `phi` (in radians).
The function returns negative values inside the ellipse, zero on the boundary.
"""
function ellipse_equation(center, a, b, phi)
    h, k = center
    
    return (x, y) -> begin
        # Translate to origin
        x_t = x - h
        y_t = y - k
        
        # Rotate coordinates
        x_r = x_t * cos(phi) + y_t * sin(phi)
        y_r = -x_t * sin(phi) + y_t * cos(phi)
        
        # Ellipse equation
        val = (x_r/a)^2 + (y_r/b)^2 - 1
        
        # Apply sigmoid like with circles
        # Returning val actually returns points, but doesn't work well...
        # return val
        return sigmoid(val; offset = 0.25)
    end
end

"""
    plot_road_boundaries(; show_plot=true)

Plot the road boundaries using the equations generated from the MATLAB points.
"""
function plot_road_boundaries(; show_plot=true)
    equations = generate_road_equations(;circles = true, ellipses = false, lines = true)
    return plot_background_with_equations(equations=equations, show_plot=show_plot)
end