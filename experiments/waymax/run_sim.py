import jax
from jax import numpy as jnp
import numpy as np
import mediapy
from tqdm import tqdm
import dataclasses
import pickle

from waymax import config as _config
from waymax import dataloader
from waymax import datatypes
from waymax import dynamics
from waymax import env as _env
from waymax import agents
from waymax import visualization

import pdb

def run_sim(scenario_path: str="./experiments/waymax/data/scenario_iter_1.pkl", horizon: int = 10):
    with open(scenario_path, 'rb') as f:
        scenario = pickle.load(f)
    init_steps = 11
    dynamics_model = dynamics.InvertibleBicycleModel()
    env = _env.MultiAgentEnvironment(
        dynamics_model=dynamics_model,
        config=dataclasses.replace(
            _config.EnvironmentConfig(),
            max_num_objects=scenario.object_metadata.num_objects,
            controlled_object=_config.ObjectType.SDC,
        ),
    )


    # Setup a few actors, see visualization below for how each actor behaves.

    # An actor that doesn't move, controlling all objects with index > 4
    obj_idx = jnp.arange(scenario.object_metadata.num_objects)
    static_actor = agents.create_constant_speed_actor(
        speed=0.0,
        dynamics_model=dynamics_model,
        is_controlled_func=lambda state: obj_idx >= 4,# TODO: change this to find all non-important agents, TODO: is obj_idx 1-indexed?
    ) # use this static actor for all agents not in the game theoretic setup

    # Exper/log actor controlling objects 3 and 4.
    expert_actor = agents.create_expert_actor(
        dynamics_model=dynamics_model,
        is_controlled_func=lambda state: obj_idx < 3, # TODO: change to all non-sdc agents
    )
    controlled_mask = jnp.zeros((scenario.object_metadata.num_objects, 1), dtype=jnp.bool_).at[3, 0].set(True)
    # The controlled actor calls the Julia-based game-theoretic solver.
    controlled_actor = agents.actor_core_factory(
        lambda random_state: [0.0],
        lambda env_state, prev_agent_state, arg3, arg4: agents.WaymaxActorOutput(
            actor_state=jnp.array([0.0]),
            action=datatypes.Action(
                data=jnp.zeros((scenario.object_metadata.num_objects, 2)) 
                .at[3, :]
                .set(jnp.array([1.0, 0.0])), # TODO: change this to the action from the game-theoretic solver
                valid=controlled_mask,
            ),
            is_controlled=jnp.zeros(scenario.object_metadata.num_objects, dtype=jnp.bool_).at[3].set(True),
        ),
    )

    agents.actor_core.register_actor_core(controlled_actor)
    actors = [static_actor, expert_actor, controlled_actor]

    jit_step = jax.jit(env.step)
    jit_select_action_list = [jax.jit(actor.select_action) for actor in actors]

    states = [env.reset(scenario)]
    for _ in range(states[0].remaining_timesteps):
        current_state = states[-1]

        outputs = [jit_select_action({}, current_state, None, None) for jit_select_action in jit_select_action_list]
        # outputs = [actor.select_action({}, current_state, None, None) for actor in actors]
        action = agents.merge_actions(outputs)
        next_state = jit_step(current_state, action)
        # next_state = env.step(current_state, action)

        states.append(next_state)

    imgs = []
    for state in states:
        imgs.append(visualization.plot_simulator_state(state, use_log_traj=False))
    mediapy.write_video("experiments/waymax/data/simulation.mp4", imgs, fps=10)


if __name__ == "__main__":
    run_sim()