module Env00

# Import the packages we need
using CairoMakie
using FileIO
using ImageCore
using Symbolics

# Import the packages that will be used both by this module and by importing modules
import BlockArrays: Block, blocksize, blocksizes, mortar, BlockVector, blocks
import LinearAlgebra: norm, norm_sqr
import Statistics: mean
import TrajectoryGamesBase: TrajectoryGamesBase, ProductDynamics, TrajectoryGame, TrajectoryGameCost, 
                           GeneralSumCostStructure, get_constraints
import TrajectoryGamesExamples: BicycleDynamics

# Reexport these for use by importing modules
export Block, blocksize, blocksizes, mortar, BlockVector, blocks
export norm, norm_sqr, mean
export TrajectoryGamesBase, ProductDynamics, TrajectoryGame, TrajectoryGameCost, GeneralSumCostStructure
export BicycleDynamics

# Include env.jl to get its functionality
include("./env.jl")

# Define and export our environment struct and functions
export EnvJLEnvironment, create_env_from_equations, get_env

# Environment compatible with TrajectoryGamesBase
struct EnvJLEnvironment
    equations::Vector{Function}
end

"""
    create_env_from_equations(; use_circles=true, use_ellipses=false, use_lines=true)

Create an environment with the road boundary equations from env.jl.
"""
function create_env_from_equations(; use_circles=true, use_ellipses=false, use_lines=true)
    equations = generate_road_equations(
        circles = use_circles,
        ellipses = use_ellipses,
        lines = use_lines
    )
    EnvJLEnvironment(equations)
end

"""
    get_env()

Create and return the default environment.
"""
function get_env()
    create_env_from_equations(use_circles=true, use_ellipses=false, use_lines=true)
end

"""
    get_constraints(environment::EnvJLEnvironment, player_index = nothing)

Get constraints for the environment.
"""
function TrajectoryGamesBase.get_constraints(environment::EnvJLEnvironment, player_index = nothing)    
    return (x) -> [eq(x[1], x[2]) for eq in environment.equations]
end

# Export functions from env.jl
export plot_background_with_equations, generate_road_equations
export line_equation, circle_equation, ellipse_equation, sigmoid

end # module 