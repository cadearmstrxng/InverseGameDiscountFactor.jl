module InverseGameDiscountFactor

using Infiltrator

using DifferentiableTrajectoryOptimization:
    get_constraints_from_box_bounds
using TrajectoryGamesExamples:
    TrajectoryGamesExamples,
    PolygonEnvironment,
    planar_double_integrator
using TrajectoryGamesBase:
    TrajectoryGamesBase,
    TrajectoryGame,
    get_constraints,
    num_players,
    state_dim,
    control_dim,
    horizon,
    state_bounds,
    control_bounds,
    JointStrategy,
    rollout,
    ProductDynamics,
    GeneralSumCostStructure,
    TrajectoryGameCost
using BlockArrays: Block, BlockVector, mortar, blocksizes, blocksize
using SparseArrays: findnz
using PATHSolver: PATHSolver
using LinearAlgebra: I, norm_sqr, norm, dot
using Random: Random
using CairoMakie
using Symbolics: Symbolics, @variables, scalarize
using ChainRulesCore: ChainRulesCore
using ForwardDiff: ForwardDiff
using Zygote: Zygote
using Statistics
using MixedComplementarityProblems

# include("utils/ExampleProblems.jl")
# using .ExampleProblems: n_player_collision_avoidance, CollisionAvoidanceGame

include("../experiments/In-D/Environment.jl")

include("utils/Utils.jl")

include("solver/ProblemFormulation.jl")
export MCPGame, MCPCoupledOptimizationSolver

include("solver/Solve.jl")
export solve_mcp_game

include("solver/InverseMCPSolver.jl")
export solve_inverse_mcp_game

include("utils/ExampleProblems.jl")

include("solver/WarmStart.jl")

include("solver/MyopicSolver.jl")
export solve_myopic_inverse_game

include("graphing/GraphingUtilities.jl")
export generate_front_page_figure, generate_partial_state_graphs
end