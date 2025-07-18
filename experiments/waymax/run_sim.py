import jax
from jax import numpy as jnp
import numpy as np
import mediapy
from tqdm import tqdm
import dataclasses
import pickle
import os
import sys
import random

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
from juliacall import Main as jl
jl.seval("using Pkg")
jl.seval(f'Pkg.activate("{project_root}")')

from waymax import config as _config, datatypes, dynamics, agents, visualization, env as _env

def run_sim(scenario_path: str="./experiments/waymax/data/scenario_iter_1.pkl",
            draw_frames: bool=False,
            save_video: bool=True,
            write_states: bool=True,
            myopic: bool=True,
            noise_level: float=0.0,
            seed: int=0,
            trial_num: int=0):
    random.seed(seed)
    print(f"[run_sim] seed: {seed}")
    print(f"[run_sim] myopic: {myopic}")
    print(f"[run_sim] noise_level: {noise_level}")
    print(f"[run_sim] write_states: {write_states}")
    print(f"[run_sim] save_video: {save_video}")
    print(f"[run_sim] trial_num: {trial_num}")

    jl.seval("include(\"experiments/waymax/Waymax.jl\")")
    
    with open(scenario_path, 'rb') as f:
        scenario = pickle.load(f)

    dynamics_model = dynamics.StateDynamics()
    env = _env.MultiAgentEnvironment(
        dynamics_model=dynamics_model,
        config=dataclasses.replace(
            _config.EnvironmentConfig(),
            max_num_objects=scenario.object_metadata.num_objects,
            controlled_object=_config.ObjectType.VALID,
        ),
    )

    roadgraph_points_file = "experiments/waymax/data/roadgraph_points.txt"
    if not os.path.exists(roadgraph_points_file):
        roadgraph_points = scenario.roadgraph_points
        valid_roadgraph_points = list(map(lambda v: v[0], filter(lambda v: v[1], zip(roadgraph_points.xy, roadgraph_points.valid))))
        with open(roadgraph_points_file, "w") as f:
            for point in valid_roadgraph_points:
                f.write(f"{point[0]} {point[1]}\n")

    agent_ids_to_log = [4, 3, 8, 2]
    id_to_idx_map = {i:int(id) for i, id in enumerate(scenario.object_metadata.ids)}
    agent_1_idx = id_to_idx_map.get(1, -1)
    
    template_state = env.reset(scenario)
    state_vectors = { "data": [] }
    for obs_t in range(0,11):
        state_vectors["data"].append([])
        for x, y, yaw, vel_x, vel_y in zip(
                scenario.log_trajectory.x[agent_ids_to_log, obs_t],
                scenario.log_trajectory.y[agent_ids_to_log, obs_t],
                scenario.log_trajectory.yaw[agent_ids_to_log, obs_t],
                scenario.log_trajectory.vel_x[agent_ids_to_log, obs_t],
                scenario.log_trajectory.vel_y[agent_ids_to_log, obs_t]
            ):
            state_vectors["data"][-1].extend([float(x), float(y), float(jnp.sqrt(vel_x**2 + vel_y**2)), float(yaw)])

    jl.initial_agent_states = state_vectors["data"]
    jl.seval(f"get_action = Waymax.run_waymax_sim(initial_agent_states;verbose=true,myopic={str(myopic).lower()})")

    state = dataclasses.replace(
        template_state,
        timestep=jnp.asarray(10),
    )
    states = [state]
    all_actions = []

    obj_idx = jnp.arange(scenario.object_metadata.num_objects)
    expert_actor = agents.IDMRoutePolicy(
        is_controlled_func=lambda state: obj_idx != 4,
        desired_vel=10.0,
        min_spacing=40.0
    )
    if write_states:
        if myopic:
            os.makedirs(f"experiments/waymax/mc/myopic", exist_ok=True)
            open(f"experiments/waymax/mc/myopic/actions_{noise_level}@{trial_num}.txt", "w").close()
            open(f"experiments/waymax/mc/myopic/contexts_{noise_level}@{trial_num}.txt", "w").close()
        else:
            os.makedirs(f"experiments/waymax/mc/baseline", exist_ok=True)
            open(f"experiments/waymax/mc/baseline/actions_{noise_level}@{trial_num}.txt", "w").close()
            open(f"experiments/waymax/mc/baseline/contexts_{noise_level}@{trial_num}.txt", "w").close()

    def get_action(jl, agent_states, myopic=False):
        jl.current_agent_states = agent_states
        ret = jl.seval("get_action(current_agent_states)")
        action, projected_actions, recovered_params = ret
        if write_states:
            if myopic:
                with open(f"experiments/waymax/mc/myopic/actions_{noise_level}@{trial_num}.txt", "a") as f:
                    f.write(str(projected_actions) + "\n")
                with open(f"experiments/waymax/mc/myopic/contexts_{noise_level}@{trial_num}.txt", "a") as f:
                    f.write(str(recovered_params) + "\n")
            else:
                with open(f"experiments/waymax/mc/baseline/actions_{noise_level}@{trial_num}.txt", "a") as f:
                    f.write(str(projected_actions) + "\n")
                with open(f"experiments/waymax/mc/baseline/contexts_{noise_level}@{trial_num}.txt", "a") as f:
                    f.write(str(recovered_params) + "\n")
        dummy_action = jnp.array([0.0 for _ in range(len(action))])
        return [action if i == 4 else dummy_action for i in range(16)]
    
    controlled_actor = agents.actor_core_factory(
        lambda random_state: [0.0],
        lambda env_state, prev_agent_state, arg3, arg4: agents.WaymaxActorOutput(
            actor_state=jnp.array([0.0]),
            action=datatypes.Action(
                data=jnp.array(get_action(jl, state_vectors["data"][-11:], myopic=myopic)),
                valid=jnp.zeros((scenario.object_metadata.num_objects, 1), dtype=jnp.bool_).at[4, 0].set(True),
            ),
            is_controlled=jnp.zeros(scenario.object_metadata.num_objects, dtype=jnp.bool_).at[4].set(True),
        ),
    )
    agents.actor_core.register_actor_core(controlled_actor)
    actors = [expert_actor, controlled_actor]

    for t in range(50):
        print(f"[run_sim] time: {t}")
        outputs = [actor.select_action({}, state, None, None) for actor in actors]
        action = agents.merge_actions(outputs)
        state = env.step(state, action)
        states.append(state)
        all_actions.append(action.data)
        state_vectors["data"].append([])
        for x, y, yaw, vel_x, vel_y in zip(
            state.current_sim_trajectory.x[agent_ids_to_log, -1:],
            state.current_sim_trajectory.y[agent_ids_to_log, -1:],
            state.current_sim_trajectory.yaw[agent_ids_to_log, -1:],
            state.current_sim_trajectory.vel_x[agent_ids_to_log, -1:],
            state.current_sim_trajectory.vel_y[agent_ids_to_log, -1:]
        ):
            state_vectors["data"][-1].extend([
                float(x[0]), 
                float(y[0]), 
                float(jnp.sqrt(vel_x**2 + vel_y**2)[0]), 
                float(yaw[0])
            ])
        noise = np.zeros(16)
        for i in range(4):
            noise[(4*i):(4*i)+2] = np.random.multivariate_normal(np.zeros(2), np.eye(2)*noise_level)
        state_vectors["data"][-1] += noise

    print("[run_sim] Done!")
    if save_video:
        print("[run_sim] videoing!")
        imgs = []
        data_folder = os.path.join(script_dir, "data")
        os.makedirs(data_folder, exist_ok=True)
        for idx, state in enumerate(states):
            state_to_plot = state
            img = visualization.plot_simulator_state(state_to_plot, use_log_traj=False)
            imgs.append(img)
            if draw_frames:
                frame_filename = os.path.join(data_folder, f"frame_{idx:03d}.png")
                mediapy.write_image(frame_filename, img)
        
        s = "myp" if myopic else "bsl"
        mediapy.write_video(f"experiments/waymax/data/sim-{s}_n-{noise_level}@{trial_num}.mp4", imgs, fps=10)
        print(f"[run_sim] video saved to experiments/waymax/data/sim-{s}_n-{noise_level}@{trial_num}.mp4")
    if write_states:
        if myopic:
            with open(f"experiments/waymax/mc/myopic/states_{noise_level}@{trial_num}.txt", "w") as f:
                for vector in state_vectors["data"]:
                    f.write(f"[{', '.join(map(str, vector))}]\n")
        else:
            with open(f"experiments/waymax/mc/baseline/states_{noise_level}@{trial_num}.txt", "w") as f:
                for vector in state_vectors["data"]:
                    f.write(f"[{', '.join(map(str, vector))}]\n")
    print("Done")

def run_mc(seed: int=0):
    for noise_level in [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]:
    # for noise_level in [0.01]:
        for trial_num in range(15):
        # for trial_num in range(1):
            run_sim(noise_level=noise_level, trial_num=trial_num, myopic=False, seed=seed)
            run_sim(noise_level=noise_level, trial_num=trial_num, myopic=True, seed=seed)

if __name__ == "__main__":
    args = {}
    args["mc"] = False
    args["seed"] = 0
    args["noise_level"] = 0.0
    args["trial_num"] = 0
    args["myopic"] = False
    args["draw_frames"] = False
    args["write_states"] = False

    for arg in sys.argv[1:]:
        if arg == "-mc":
            args["mc"] = True
        elif arg == "-s":
            args["seed"] = int(sys.argv[sys.argv.index(arg) + 1])
        elif arg == "-n":
            args["noise_level"] = float(sys.argv[sys.argv.index(arg) + 1])
        elif arg == "-m":
            args["myopic"] = True
        elif arg == "-d":
            args["draw_frames"] = True
        elif arg == "-w":
            args["write_states"] = True
    if args["mc"]:
        run_mc(seed=args["seed"])
    else:
        run_sim(noise_level=args["noise_level"], trial_num=args["trial_num"], myopic=args["myopic"], seed=args["seed"], write_states=args["write_states"])