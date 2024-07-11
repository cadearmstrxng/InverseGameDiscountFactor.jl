module TestDynamics

import InverseGameDiscountFactor.DynamicsModelInterface
import InverseGameDiscountFactor.TrajectoryVisualization
import JuMP
import Plots 
import Vegalite

using JuMP: @variable, @constraint, @NLconstraint

include("unicycle.jl")

end