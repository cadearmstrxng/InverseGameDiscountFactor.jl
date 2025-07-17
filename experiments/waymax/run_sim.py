import jax
from jax import numpy as jnp
import numpy as np
import mediapy
from tqdm import tqdm
import dataclasses
import pickle
import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
from juliacall import Main as jl
jl.seval("using Pkg")
jl.seval(f'Pkg.activate("{project_root}")')

from waymax import config as _config, dataloader, datatypes, dynamics, agents, visualization, env as _env

import pdbpp as pdb

def run_sim(scenario_path: str="./experiments/waymax/data/scenario_iter_1.pkl"):
    jl.seval("include(\"experiments/waymax/Waymax.jl\")")
    
    with open(scenario_path, 'rb') as f:
        scenario = pickle.load(f)
    init_steps = 11
    dynamics_model = dynamics.InvertibleBicycleModel()
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
    # Note: Correcting this map to be id->idx. The previous version was idx->id,
    # which caused incorrect lookups.
    id_to_idx_map = {i:int(id) for i, id in enumerate(scenario.object_metadata.ids)}
    agent_1_idx = id_to_idx_map.get(1, -1)

    # agent_indices_to_log = []
    # for id in agent_ids_to_log:
    #     if id in id_to_idx_map:
    #         agent_indices_to_log.append(id_to_idx_map[id])
    #         print(f"Agent with ID {id} found at index {id_to_idx_map[id]}")
    #     else:
    #         print(f"Warning: Agent with ID {id} not found in scenario.")

    # output_file_path = "experiments/waymax/agent_states.txt"
    
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
    jl.seval("get_action = Waymax.run_waymax_sim(initial_agent_states;verbose=true)")

    state = dataclasses.replace(
        template_state,
        timestep=jnp.asarray(10),
    )
    states = [state]

    obj_idx = jnp.arange(scenario.object_metadata.num_objects)
    controlled_mask = jnp.zeros((scenario.object_metadata.num_objects, 1), dtype=jnp.bool_).at[4, 0].set(True)

    expert_actor = agents.IDMRoutePolicy(
        # dynamics_model=dynamics_model,
        is_controlled_func=lambda state: obj_idx != 4,
    )
    controlled_actor = agents.actor_core_factory(
        lambda random_state: [0.0],
        lambda env_state, prev_agent_state, arg3, arg4: agents.WaymaxActorOutput(
            actor_state=jnp.array([0.0]),
            action=datatypes.Action(
                data=jnp.array(get_action(jl, state_vectors["data"][-11:])),
                valid=controlled_mask,
            ),
            is_controlled=jnp.zeros(scenario.object_metadata.num_objects, dtype=jnp.bool_).at[4].set(True),
        ),
    )
    agents.actor_core.register_actor_core(controlled_actor)

    actors = [expert_actor, controlled_actor]
    # jit_step = jax.jit(env.step)
    # jit_select_action_list = [jax.jit(actor.select_action) for actor in actors]

    for t in range(states[0].remaining_timesteps):
        print(f"[run_sim] time: {t}")
        # outputs = [jit_select_action({}, state, None, None) for jit_select_action in jit_select_action_list]
        outputs = [actor.select_action({}, state, None, None) for actor in actors]
        action = agents.merge_actions(outputs)
        # state = jit_step(state, action)
        state = env.step(state, action)
        states.append(state)
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

    print("[run_sim] videoing!")
    imgs = []
    for state in states:
        state_to_plot = state
        if agent_1_idx != -1:
            metadata = state.object_metadata
            new_valid = metadata.is_valid.at[1].set(False)
            new_metadata = dataclasses.replace(metadata, is_valid=new_valid)
            state_to_plot = dataclasses.replace(state, object_metadata=new_metadata)
            imgs.append(visualization.plot_simulator_state(state_to_plot, use_log_traj=False))
    mediapy.write_video("experiments/waymax/data/simulation.mp4", imgs, fps=10)
    print("Done")

def get_action(jl, agent_states):
    jl.current_agent_states = agent_states
    action = jnp.array(jl.seval("get_action(current_agent_states)"))
    print(f"Action: {action}")
    dummy_action = jnp.array([0.0 for _ in range(len(action))])
    return [dummy_action if i == 4 else action for i in range(16)]

if __name__ == "__main__":
    run_sim()