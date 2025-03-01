module InD

using BlockArrays: blocksizes, Block, mortar, BlockVector
using Infiltrator
using CairoMakie
using TrajectoryGamesBase:
    num_players, state_dim
using LinearAlgebra: norm_sqr

using ImageTransformations
using Rotations
using OffsetArrays:Origin
using TrajectoryGamesExamples: BicycleDynamics, PolygonEnvironment

include("../GameUtils.jl")
include("../graphing/ExperimentGraphingUtils.jl")
include("../../src/InverseGameDiscountFactor.jl")
include("Environment.jl")
export warm_start_check
function warm_start_check()
    warm_start_sol = [[-20.61196393248886, 61.22384244853292, 0.01257837840983515, 1.4700434998064908, -20.61070567210564, 61.33105557264351, 0.012586829254541668, 0.6742183107660736, -20.609446976207018, 61.39803586917073, 0.012587088717887728, 0.6653876469835222, -20.608187939003123, 61.46581732866123, 0.012593655359967003, 0.690241897375055, -20.606928259093493, 61.53632518479196, 0.012599942832619261, 0.7199154740066956, -20.605668256672867, 61.609242038516186, 0.012600105579922704, 0.7384216817451992, -20.604407958437932, 61.68396998142295, 0.012605859118795831, 0.756137244280926, -20.60314737466488, 61.759720685910736, 0.012605816342260103, 0.7588769014230039, -20.60188665767406, 61.8356125223502, 0.012608523473999073, 0.7589598728500271, -20.60062571559915, 61.91150286205566, 0.012610318024248987, 0.7588469575568361, -20.599364683796725, 61.987393759577095, 0.012610318024248937, 0.7589710210129896, -20.5981036519943, 62.063309636099845, 0.01261031802424885, 0.7593465301827134, -20.596842584579647, 62.13926989202214, 0.012611030268844293, 0.7598586020658844, -20.595581481552763, 62.215267261960626, 0.012611030268844315, 0.7600888037123517, -0.0006304951583886553, -10.067004304101372, 8.450844706677894e-5, -7.958253152855062, 2.5946334608751497e-6, -0.08830648679626955, 6.566642079645837e-5, 0.24854354827364536, 6.287472652608473e-5, 0.29673612765070784, 1.6274730345935555e-6, 0.18506217658717616, 5.753538873167799e-5, 0.17715569528704636, -4.2776535858818843e-7, 0.02739662097499226, 2.7071317391516466e-5, 0.0008297493653670818, 1.7945502499397373e-5, -0.0011291283060543148, 6.866831392010643e-20, 0.0012406514599671865, -6.221655254106912e-18, 0.003755102771061694, 7.1224459537806865e-6, 0.0051207254029216305, -3.0649481131731405e-19, 0.002302019443705012], [-3.1985736740875814, 66.4605381189341, -0.5763345700909739, 2.728619944566922, -3.214754688195581, 66.68316907867694, 0.2527142879309777, 1.261999919775047, -3.1882964177180586, 66.77672772554013, 0.27645112161946744, 0.6091724194641926, -3.1596407794664927, 66.83610660562867, 0.2966616434118539, 0.5784050310157663, -3.1291237338250353, 66.8959101530453, 0.3136792694172904, 0.6176657276511194, -3.097048856946501, 66.95974236030513, 0.32781826815339704, 0.6589781857071888, -3.0636891839841365, 67.02638588218993, 0.33937519109389397, 0.6738920069270515, -3.0292889036018433, 67.0944062769848, 0.3486304165519692, 0.6865156972855291, -2.9940648927998645, 67.16383686132222, 0.355849799487613, 0.7020958402976603, -2.9582080836956566, 67.23421480823067, 0.36128638259654555, 0.7054629832813994, -2.9218846540160737, 67.30482483228165, 0.3651822109951069, 0.7067374117076766, -2.885237033584243, 67.37555744291349, 0.36777019764151453, 0.7079147391187671, -2.848384720133801, 67.44641244751126, 0.369276071367323, 0.7091853124306362, -2.8114249000784226, 67.51735856773143, 0.3699203297402466, 0.7097370716453985, -10.0, -10.239061995784885, 8.290488580219515, -10.046199999459692, 0.23736833688489803, -6.528273908443043, 0.20210521792386543, -0.3076736960778373, 0.1701762600543656, 0.3926069329073223, 0.14138998736106648, 0.413124282195571, 0.11556922940496932, 0.14913793431236186, 0.09255225458075352, 0.12623671662298364, 0.07219382935644104, 0.1558013032876811, 0.054365831089328055, 0.0336713439458194, 0.03895828398561567, 0.012744226986440099, 0.02587986646407723, 0.011773237377753149, 0.015058737258085592, 0.012705711623295616, 0.006442583729235974, 0.005517582458166871], [-12.990801626108569, 26.791412151151164, 4.866872347025576, 2.1111696055529054, -12.554929377166788, 26.95245170421921, 2.220647926121466, 0.9548107721816209, -12.382969301377132, 27.04019222857146, 1.0092774894178194, 0.7999998345804845, -12.319181614950137, 27.11095308421246, 0.2664752122461602, 0.6152167783638773, -12.291129586414321, 27.175077751539465, 0.2945655975504572, 0.6672765676065452, -12.255920761148051, 27.244822003123286, 0.4096109457109769, 0.7276084292425704, -12.209322696147805, 27.319366625368964, 0.5223504758108233, 0.7632840573691381, -12.154068474204024, 27.396304854074536, 0.5827340549288619, 0.7754805517101101, -12.093063683110906, 27.47434332073239, 0.6373618457817043, 0.7852888104863242, -12.026370857279415, 27.552870919882082, 0.6964947360923719, 0.7852631959139773, -11.95619394994611, 27.631344857455943, 0.7070434624648354, 0.7842155737758417, -11.88539799620408, 27.70973589477969, 0.7088756514417851, 0.7836051861413893, -11.814456633922344, 27.788092653260144, 0.7099516207698364, 0.7835299924348537, -11.743447752282071, 27.866449865767102, 0.7102260258951096, 0.7836142622838811, -10.5696797784457, -10.146597976728868, -10.162992577054252, -10.015481082980227, -10.02092772809542, -1.5481075987058512, -7.4280257788737, -1.8478325117598369, 0.28090488233844485, 0.5205978856826994, 1.1504538739930312, 0.6033185970396411, 1.12739562872236, 0.356756348404961, 0.6038359959689769, 0.12196498929012742, 0.5462780503957544, 0.09808261985622792, 0.5913290008875778, -0.00025612350355171847, 0.10548732997352203, -0.010476206321377718, 0.018321932941430433, -0.006103866562926937, 0.010759718990194149, -0.0007519312800409565, 0.0027440630779876677, 0.0008427011237496702]]
    frames = [780, 916]
    downsample_rate = 10
    InD_observations = GameUtils.pull_trajectory("07";
        track = [20, 19, 22], downsample_rate = downsample_rate, all = false, frames = frames)
    horizon = length(frames[1]:downsample_rate:frames[2])
    
    n = 3
    p_state_dim = 4

    CairoMakie.activate!()
    fig = CairoMakie.Figure()
    ax = CairoMakie.Axis(fig[1,1], aspect = DataAspect())

    image_data = CairoMakie.load("experiments/data/07_background.png")
    image_data = image_data[end:-1:1, :]
    image_data = image_data'
    # ax1 = Axis(fig[1,1], aspect = DataAspect())
    trfm = ImageTransformations.recenter(Rotations.RotMatrix(-2.303611),center(image_data))
    x_crop_min = 430
    x_crop_max = 875
    y_crop_min = 225
    y_crop_max = 1025
    
    scale = 1/10.25

    x = (x_crop_max - x_crop_min) * scale
    y = (y_crop_max - y_crop_min) * scale

    image_data = ImageTransformations.warp(image_data, trfm)
    image_data = Origin(0)(image_data)
    image_data = image_data[x_crop_min:x_crop_max, y_crop_min:y_crop_max]
    
    x_offset = -34.75
    y_offset = 22

    CairoMakie.image!(ax,
        x_offset..(x+x_offset),
        y_offset..(y+y_offset),
        image_data)
    colors = [[(:red, 1.0), (:blue, 1.0), (:green, 1.0)], [(:red, 0.75), (:blue, 0.75), (:green, 0.75)]]
    for i in 1:n
        # CairoMakie.lines!(ax, 
        #     [InD_observations[t][Block(i)][1] for t in 1:horizon],
        #     [InD_observations[t][Block(i)][2] for t in 1:horizon], 
        #     color = colors[1][i])
        CairoMakie.lines!(ax, # warm start
            [warm_start_sol[n][p_state_dim * (t - 1) + 1] for t in 1:horizon],
            [warm_start_sol[n][p_state_dim * (t - 1) + 2] for t in 1:horizon], 
            color = colors[1][i])
    end
    CairoMakie.lines!(ax, # warm start
            [warm_start_sol[2][p_state_dim * (t - 1) + 1] for t in 1:horizon],
            [warm_start_sol[2][p_state_dim * (t - 1) + 2] for t in 1:horizon], 
            color = colors[1][1])
    CairoMakie.lines!(ax, # warm start
            [warm_start_sol[1][p_state_dim * (t - 1) + 1] for t in 1:horizon],
            [warm_start_sol[1][p_state_dim * (t - 1) + 2] for t in 1:horizon], 
            color = colors[1][2])

    # println([warm_start_sol[1][p_state_dim * (t - 1) + 1] for t in 1:horizon])
    # println([warm_start_sol[1][p_state_dim * (t - 1) + 2] for t in 1:horizon])
    # println([warm_start_sol[2][p_state_dim * (t - 1) + 1] for t in 1:horizon])
    # println([warm_start_sol[2][p_state_dim * (t - 1) + 2] for t in 1:horizon])

    CairoMakie.display(fig)
    
end


export run_bicycle_sim
function run_bicycle_sim(;full_state=true, graph=true, verbose = true)

    # InD_observations = GameUtils.observe_trajectory(forward_solution, init)
    frames = [26158, 26320] # 162
    # 201 25549, 26381
    # 205 25847,26381
    # 207,26098,26320
    # 208,26158,26381,
    tracks = [201, 205, 207, 208]
    downsample_rate = 9
    InD_observations = GameUtils.pull_trajectory("07";
        track = tracks, downsample_rate = downsample_rate, all = false, frames = frames)
    open("InD_observations.tmp.txt", "w") do f
        for i in eachindex(InD_observations)
            write(f, string(round.(InD_observations[i]; digits = 4)), "\n")
        end
    end
    trk_19_lane_center(x) = -0.00610541116510255*x^2 - 0.116553046264268*x + 65.4396555389841 
    trk_20_lane_center(x) = 0.000405356859692973*x^4 + 0.0390723153374032*x^3 + 1.40388631159093*x^2 + 22.3233378977068*x + 193.852722156383
    trk_22_lane_center(x) = 0.238799724199197*x^2 + 14.8710682662040*x + 187.979162321130
    lane_centers = [trk_19_lane_center, trk_20_lane_center, trk_22_lane_center]
    # TODO need to time-synch each trajectory
    # 20: 646 - 1103
    # 19: 620 - 1001
    # 22: 780 - 916
    # 780 -> 916 = 136
    # Take 1:
    # 17: 530- 806
    # 19: 620-1001
    # 22: 780- 916
    # 780 -> 806 = 26

    # potentially use receding horizon - https://arxiv.org/pdf/2302.01999
    # downsample the trajectory by 5

    # @infiltrate
    dynamics = BicycleDynamics(;
        dt = 0.04*downsample_rate, # needs to become framerate
        l = 1.0,
        state_bounds = (; lb = [-Inf, -Inf, -Inf, -Inf], ub = [Inf, Inf, Inf, Inf]),
        control_bounds = (; lb = [-5, -pi/4], ub = [5, pi/4]),
        integration_scheme = :forward_euler
    )

    init = GameUtils.init_bicycle_test_game(
        full_state;
        initial_state = InD_observations[1],
        game_params = mortar([
            [[InD_observations[end][Block(i)][1:2]..., 1.0, 1.0, 1.0, 1.0, 1.0, 1.0] for i in 1:length(tracks)]...]),
        # game_environment = PolygonEnvironment(4, 200),
        horizon = length(frames[1]:downsample_rate:frames[2]),
        n = length(tracks),
        dt = 0.04*downsample_rate,
        myopic=true,
        verbose = verbose,
        dynamics = dynamics,
        lane_centers = lane_centers
    )
    !verbose || println("initial state: ", init.initial_state)
    !verbose || println("initial game parameters: ", init.game_parameters)
    !verbose || println("initial horizon: ", init.horizon)
    !verbose || println("observation model: ", init.observation_model)
    !verbose || println("observation dim: ", init.observation_dim)
    
    InD_observations = (full_state) ? InD_observations : [BlockVector(GameUtils.observe_trajectory(InD_observations[t], init;blocked_by_time = false),
        [init.observation_dim for _ in 1:length(tracks)]) for t in 1:init.horizon]
    
    !verbose || println("game initialized\ninitializing mcp coupled optimization solver")
    mcp_solver = InverseGameDiscountFactor.MCPCoupledOptimizationSolver(
        init.game_structure.game,
        init.horizon,
        blocksizes(init.game_parameters, 1)
    )
    !verbose || println("mcp coupled optimization solver initialized")

    # !verbose || println("solving forward game")
    # fs_temp = InverseGameDiscountFactor.solve_mcp_game(
    #     mcp_solver.mcp_game,
    #     init.initial_state,
    #     init.game_parameters;
    #     verbose = false
    #     )
    # forward_solution = InverseGameDiscountFactor.reconstruct_solution(
    #     fs_temp,
    #     init.game_structure.game,
    #     init.horizon
    # )
    # xs = [BlockVector(forward_solution[Block(t)], [4 for _ in 1:length(tracks)]) for t in 1:init.horizon]
    # xs = vcat([init.initial_state], xs)
    # us = [fs_temp.primals[i][4*init.horizon+1:end] for i in 1:length(tracks)]
    # us = [vcat([us[i][2*t-1:2*t] for i in 1:length(tracks)]...) for t in 1:init.horizon]
    # us = [BlockVector(us[t], [2 for _ in 1:length(tracks)]) for t in 1:init.horizon]

    # cost_val = mcp_solver.mcp_game.game.cost(xs, us, init.game_parameters)
    # !verbose || println("forward game solved, cost: ", cost_val)
    # # @infiltrate

    # # return
    # !verbose||println("forward game solved, status: ", fs_temp.status)
    # forward_game_observations = GameUtils.observe_trajectory(forward_solution, init)

    # Add graph comparing forward solution to observations
    # if graph
    #     ExperimentGraphingUtils.graph_trajectories(
    #         "Observed v. Forward Solution",
    #         [InD_observations, BlockVector(vcat(forward_game_observations...), [length(tracks)*4 for _ in 1:init.horizon])],
    #         init.game_structure,
    #         init.horizon;
    #         colors = [
    #             [(:red, 1.0), (:blue, 1.0), (:green, 1.0)],
    #             [(:red, 0.5), (:blue, 0.5), (:green, 0.5)]
    #         ],
    #         constraints = get_constraints(init.environment)
    #     )
    # end
    # return

    !verbose || println("solving inverse game")
    method_sol = InverseGameDiscountFactor.solve_myopic_inverse_game(
        mcp_solver.mcp_game,
        InD_observations,
        # forward_game_observations,
        init.observation_model,
        Tuple(blocksizes(init.game_parameters, 1)),;
        initial_state = init.initial_state,
        hidden_state_guess = init.game_parameters,
        max_grad_steps = 200,
        retries_on_divergence = 3,
        verbose = verbose,
        dynamics = dynamics,
    )
    !verbose || println("finished inverse game")
    !verbose || println("recovered pararms: ", method_sol.recovered_params)

    if graph
        ExperimentGraphingUtils.graph_trajectories(
            "Observed v. Recovered v. Warm Start",
            [InD_observations, method_sol.recovered_trajectory, method_sol.warm_start_trajectory],
            init.game_structure,
            init.horizon;
            colors = [
                [(:red, 1.0), (:blue, 1.0), (:green, 1.0)],
                [(:red, 0.5), (:blue, 0.5), (:green, 0.5)],
                [(:red, 0.2 ), (:blue, 0.2), (:green, 0.2)]
            ],
            constraints = get_constraints(init.environment)
        )
        ExperimentGraphingUtils.graph_trajectories(
            "Observed v. Recovered",
            [InD_observations, method_sol.recovered_trajectory],
            init.game_structure,
            init.horizon;
            colors = [
                [(:red, 1.0), (:blue, 1.0), (:green, 1.0), (:purple, 1.0)],
                [(:red, 0.5), (:blue, 0.5), (:green, 0.5), (:purple, 0.5)],
                [(:red, 0.2 ), (:blue, 0.2), (:green, 0.2), (:purple, 0.2)]
            ],
            constraints = get_constraints(init.environment)
        )
        ExperimentGraphingUtils.graph_trajectories(
            "Observed v. Warm Start",
            [InD_observations, method_sol.warm_start_trajectory],
            init.game_structure,
            init.horizon;
            colors = [
                [(:red, 1.0), (:blue, 1.0), (:green, 1.0), (:purple, 1.0)],
                [(:red, 0.5), (:blue, 0.5), (:green, 0.5), (:purple, 0.5)],
                [(:red, 0.2 ), (:blue, 0.2), (:green, 0.2), (:purple, 0.2)]
            ],
            constraints = get_constraints(init.environment)
        )
        ExperimentGraphingUtils.graph_trajectories(
            "Recovered v. Warm Start",
            [InD_observations, method_sol.recovered_trajectory, method_sol.warm_start_trajectory],
            init.game_structure,
            init.horizon;
            colors = [
                [(:red, 0.0), (:blue, 0.0), (:green, 0.0), (:purple, 0.0)],
                [(:red, 1.0), (:blue, 1.0), (:green, 1.0), (:purple, 1.0)], 
                [(:red, 0.5), (:blue, 0.5), (:green, 0.5), (:purple, 0.5)]
            ],
            constraints = get_constraints(init.environment)
        )
    end

    sol_error = norm_sqr(method_sol.recovered_trajectory - vcat(InD_observations...))
    !verbose || println("inverse sol error: ", sol_error)
    warm_sol_error = norm_sqr(method_sol.warm_start_trajectory - vcat(InD_observations...))
    !verbose || println("warm start sol error: ", warm_sol_error)
    !verbose || println("% improvement on warm start: ", (warm_sol_error - sol_error) / warm_sol_error * 100)
end

end