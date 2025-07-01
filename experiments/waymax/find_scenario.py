import pickle
import numpy as np
import dataclasses
import jax
from tqdm import tqdm
import matplotlib.pyplot as plt
import mediapy
import os
import pdb

from waymax import config as _config
from waymax import dataloader
from waymax import visualization


def find_and_save_scenario(min_agents=2, max_agents=8):
    """
    Finds a suitable scenario from the Waymo Open Motion Dataset and saves it.
    """
    base_config = _config.WOD_1_1_0_TRAINING
    fixed_path = base_config.path.replace('gs:///', 'gs://')
    config = dataclasses.replace(
        base_config,
        path=fixed_path,
        max_num_objects=16,
    )
    data_iter = dataloader.simulator_state_generator(config=config)
    iteration = -1

    while True:
        iteration += 1
        print(f"--- Loading and Verifying Scenario #{iteration} ---")
        scenario = next(data_iter)
        print("✓ Scenario loaded.")

        metadata = scenario.object_metadata
        vehicle_indices = np.where(metadata.object_types == 1)[0]
        sdc_index_for_plot = -1
        start_ts = -1

        for idx in vehicle_indices:
            if np.any(scenario.log_trajectory.valid[idx]):
                sdc_index_for_plot = idx
                start_ts = np.argmax(scenario.log_trajectory.valid[idx])
                break

        if sdc_index_for_plot != -1:
            print(f"✓ Found candidate vehicle at index: {sdc_index_for_plot}. Becomes valid at t={start_ts}.")
            # Time-shift the data to the first valid timestep
            time_shifted_log_traj = jax.tree_util.tree_map(
                lambda x: x[:, start_ts:], scenario.log_trajectory
            )
            time_shifted_scenario = dataclasses.replace(scenario, log_trajectory=time_shifted_log_traj)

            # Promote the vehicle to SDC
            object_types = time_shifted_scenario.object_metadata.object_types.at[sdc_index_for_plot].set(0)
            new_metadata = dataclasses.replace(time_shifted_scenario.object_metadata, object_types=object_types)
            plot_scenario = dataclasses.replace(time_shifted_scenario, object_metadata=new_metadata)

            batched_plot_scenario = jax.tree_util.tree_map(lambda x: np.expand_dims(x, axis=0), plot_scenario)

            try:
                img = visualization.plot_simulator_state(batched_plot_scenario, use_log_traj=True, batch_idx=0)
                print("✓ Scenario is valid for plotting!")

                plt.figure(figsize=(10, 10))
                plt.imshow(img)
                plt.axis('off')
                plt.title(f'Waymax Visualization (Centered on Object {sdc_index_for_plot}, Original t={start_ts})')
                plt.show()

                user_choice = input("Save this scenario? (y/n): ").lower().strip()

                if user_choice == 'y':
                    output_dir = './experiments/waymax/data/'
                    os.makedirs(output_dir, exist_ok=True)

                    filename = f"scenario_iter_{iteration}.pkl"
                    output_path = os.path.join(output_dir, filename)

                    with open(output_path, 'wb') as f:
                        pickle.dump(plot_scenario, f)
                    print(f"✓ Scenario saved to {output_path}")
                    print(f"--- Scenario #{iteration} saved ---")
                    print(f"Continue? (y/n): ")
                    user_choice = input().lower().strip()
                    if user_choice == 'n':
                        break
                    else:
                        continue
                else:
                    print("✗ Scenario discarded. Searching for a new one...")
            except IndexError:
                print("✗ Scenario has inconsistent data (valid=True, but no xy). Retrying with a new scenario...")
        else:
            print("✗ No vehicles with any valid trajectories found. Retrying with a new scenario...")


if __name__ == "__main__":
    find_and_save_scenario() 