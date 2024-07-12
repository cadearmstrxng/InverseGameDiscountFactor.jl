import JuMP
import PATHSolver
import Zygote

using JuMP: @variable, @constraint, @objective
using InverseGameDiscountFactor.ForwardGame: solve_inverse_game
using InverseGameDiscountFactor.InverseGames: solve_inverse_game
using InverseGameDiscountFactor.TrajectoryVisualization: visualize_trajectory
using SparseArrays: spzeros
using Test: @test, @testset, @test_broken
using UnPack: @unpack

import Plots

unique!(push!(LOAD_PATH, joinpath(@__DIR__, "utils")))
import TestUtils
import TestDynamics


function objective_p1(x, u; weights)
    weights[:state_velocity] * sum((x[3, :] .- 1.0) .^ 2) + weights[:control_Δv] * sum(u[1, :] .^ 2)
end

function objective_gradients_p1(x, u; weights)
    T = size(x,2)
    dJdx = 2 * weights[:state_velocity] * [zeros(2,T); x[3:3, :] .- 1.0; zeros(1,T)]
    dJdu = 2 * weights[:control_Δv] * [u[1:1, :]; zeros(1,T)]
    (; dx = dJdx, du = dJdu)
end

function objecting_p2(x, u2; weights)
    weights[:state_goal] * sum(x[1:2, :] .^ 2) + weights[:control_Δθ] * sum(u2 .^ 2)
end

function objective_gradients_p2(x, u2; weights)
    T = size(x,2)
    dJdx = 2 * weights[:state_goal] * [x[1:2, :]; zeros(2,T)]
    dJdu = 2 * weights[:control_Δθ] * [zeros(1,T); u2[2:2, :]]
    (; dx = dJdx, du = dJdu)
end

control_system = TestDynamics.Unicycle(0.1)
observation_model = (; σ = 0.0, expected_obeservation = identity)
x0 = [-1, 1, 0, 0]
T = 100

player_cost_models = (
    (;
        player_inputs = [1],
        weights = (; state_velocity = 1, control_Δv = 10),
        objective = objective_p1,
        objective_gradients = objective_gradients_p1,
        add_objective! = function (opt_model, args...; kwargs...)
            @objective(opt_model, Min, objective_p1(args...; kwargs...))
        end,
        add_objective_gradients! = function (opt_model, args...; kwargs...)
            objective_gradients_p1(args...; kwargs...)
        end
    ),
    (;
        player_inputs = [2],
        weights = (; state_goal = 0.1, control_Δθ = 10),
        objective = objective_p12,
        objective_gradients = objective_gradients_p2,
        add_objective! = function (opt_model, args...; kwargs...)
            @objective(opt_model, Min, objective_p2(args...; kwargs...))
        end,
        add_objective_gradients! = function (opt_model, args...; kwargs...)
            objective_gradients_p2(args...; kwargs...)
        end
    ),
)

@testset "Gradient check" begin
    x = rand(4, 100)
    u = rand(2, 100)

    for (player_idx, cost_model) in enumerate(player_cost_models)
        dJdx_ad, dJdu_ad = 
            Zygote.gradient((x,u) -> cost_model.objective(x, u; cost_model.weights), x, u)
        dJ = cost_model.objective_gradients(x, u; cost_model.weights)
        @test dJdx_ad == dJ.dx
        @test dJdu_ad[player_idx, :] == dJ.du[player_idx, :]
    end
end

function test_unicycle_multipliers(system, λ, x, u; player_cost_models)
    df = let 
        As = [
            [
                1 0 system.ΔT*cos(x[4, t]) -system.ΔT*x[3, t]*sin(x[4, t]);
                0 1 system.ΔT*sin(x[4, t]) system.ΔT*x[3, t]*cos(x[4, t]);
                0 0 1 0;
                0 0 0 1
            ] for t in 1:T
        ]

        Bs = [
            [
                0 0;
                0 0;
                1 0;
                0 1
            ] for t in 1:T
        ]

        (;
            dx = reduce((A, x) -> cat(A,x; dims = 3), As),
            du = reduce((A, x) -> cat(A,x; dims = 3), Bs),
        )
        
    end

    for (player_idx, cost_model) in enumerate(player_cost_models)
        @testset "λ$player_idx" begin
            dJ = cost_model.objective_gradients(x, u; cost_model.weights)

            dLdx = [
                dJ.dx[:, t] + λ[:, t-1, player_idx]' - df.dx[:, :, t]' * λ[:, t, player_idx]
                for t in 2:(T-1)
            ]
            dLdu = [
                dJ.du[cost_model.player_inputs, t] - 
                (df.du[:, cost_model.player_inputs,t]' * λ[:, t, player_idx])
                for t in 1:(T-1)
            ]

            @test all(all(isapprox.(x, 0; atol = 1e-6)) for x in dLdx)
            @test all(all(isapprox.(x, 0; atol = 1e-6)) for x in dLdu)
        end
    end
end

@testset "Forward Nash" begin
    forward_converged, forward_solution, forward_model = 
        solve_game(control_system, player_cost_models, x0, T)
    global forward_solution

    @test forward_converged

    visualize_trajectory(control_system, forward_solution.x)

    test_unicycle_multipliers(
        control_system, 
        forward_solution.λ, 
        forward_solution.x, 
        forward_solution.u;
        player_cost_models
    )
end

@testset "Inverse Nash" begin
    inverse_converged, inverse_solution, inverse_model =
        solve_inverse_game(
            forward_solution.x, 
            init = (; forward_solution.u), 
            control_system,
            observation_model,
            player_cost_models,
            max_observation_error = 0.1,
    )
    global inverse_solution

    @test inverse_converged

    for (cost_model, weights) in zip(player_cost_models, inverse_solution.player_weights)
        TestUtils.test_inverse_solution(weights, cost_model.weights)
    end
    TestUtils.test_data_fidelity(
        inverse_model, 
        observation_model, 
        forward_solution.x, 
        forward_solution.x,
    )
end 