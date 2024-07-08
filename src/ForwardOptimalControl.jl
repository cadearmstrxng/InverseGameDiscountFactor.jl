module ForwardOptimalControl

import ..DynamicsModelInterface
import ..JuMPUtils
import PATHSolver
import JuMP

using JuMP: @variable, @constraint, @objective
using UnPack: @unpack

export solve_optimal_control

#===============================================================================================#

function solve_optimal_control(
    control_system,
    cost_model,
    xₒ,
    T;
    fixed_inputs = (),
    init = (),
    solver = PATHSolver.Optimizer,
    solver_attributes = (; print_level = 3),
    verbose = false,
)
    @unpack n_states, n_controls = control_system
    opt_model = JuMP.Model(solver)
    JuMPUtils.set_solver_attributes!(opt_model; solver_attributes...)

    # decision variables
    x = @variable(opt_model, [1:n_states, 1:T])
    u = @variable(opt_model, [1:n_controls, 1:T])

    # initial guess
    JuMPUtils.init_if_hasproperty!(x, init, :x)
    JuMPUtils.init_if_hasproperty!(u, init, :u)

    # fix certain inputs 
    for i in fixed_inputs
        JuMP.fix.(u[i, :], init.u[i, :])
    end

    DynacmicsModelInterface.add_dynamics_constraints!(control_system, opt_model, x, u)

    if !isnothing(xₒ)
        @constraint(opt_model, x[:, 1] .== xₒ)
    end

    cost_model.add_objective!(opt_model, x, u; cost_model.weights)
    time = @elapsed JuMP.optimize!(opt_model)
    verbose && @info time

    JuMPUtils.isconverged(opt_model), JuMPUtils.get_values(; x, u), opt_model
end

end