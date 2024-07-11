using Test: @testset

@testset "Tests" begin
	@testset "Single Player: Forward and Inverse Optimal Control" begin
		include("nonlq_control.jl")
	end

	@testset "Multi Player: Forward and Inverse Infinite Dynamnic Games" begin
		include("shared_unicycle_game.jl")
	end
end
