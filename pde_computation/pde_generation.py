import numpy as np
import matplotlib.pyplot as plt
import os
import h5py
from datetime import datetime


def pde_run(temp1, temp2, length, spatial_points, alpha, cooling_coefficient, t_env, safety_factor=0.9):
    # Spatial discretization
    dx = length / spatial_points
    x = np.linspace(0, length, spatial_points)
    dt = safety_factor * (0.5 * dx**2 / alpha)
    
    # Initial array allocation
    chunk_size = 1000  # Number of time steps to add when extending
    u = np.zeros((chunk_size, spatial_points))
    t = np.zeros(chunk_size)
    
    # Set initial condition
    u[0, :] = np.where(x < (length * 0.5), temp1, temp2)
    
    # Check stability
    stability = alpha * dt / dx**2
    if stability >= 0.5:
        raise ValueError(f"Stability condition not met: {stability} >= 0.5")
    
    # Evolution parameters
    tolerance = 1e-4
    n = 0  # Current time step
    
    while True:
        # Extend arrays if needed
        if n >= u.shape[0] - 2:
            u = np.vstack((u, np.zeros((chunk_size, spatial_points))))
            t = np.append(t, np.zeros(chunk_size))
        
        # Update time array
        t[n+1] = t[n] + dt
        
        # Compute next time step
        for i in range(1, spatial_points-1):
            diffusion = stability * (u[n,i+1] - 2*u[n,i] + u[n,i-1])
            cooling = -cooling_coefficient * dt * (u[n,i] - t_env)
            u[n+1,i] = u[n,i] + diffusion + cooling
        
        # Apply boundary conditions
        u[n+1,0] = temp1
        u[n+1,-1] = temp2
        
        # Check convergence
        if n > 0 and np.max(np.abs(u[n+1] - u[n])) < tolerance:
            print(f"Converged after {n+1} timesteps ({t[n+1]:.3f} seconds)")
            # Trim unused space and return
            return x, t[:n+2], u[:n+2,:]
        
        n += 1
        
        # Optional: Add maximum iteration check
        if n >= 1000000:  # Some very large number
            print("Warning: Maximum iterations reached without convergence")
            return x, t[:n+1], u[:n+1,:]

def visualize_heat_diffusion(x, t, u, num_snapshots=5):
    """
    Visualize heat diffusion at different time steps.

    Parameters:
    x: spatial points array
    t: time points array
    u: solution array (time, space)
    num_snapshots: number of time snapshots to show
    """
    # Select time indices to plot
    time_indices = np.linspace(0, len(t)-1, num_snapshots, dtype=int)

    plt.figure(figsize=(12, 6))

    # Plot temperature distribution at different times
    for idx in time_indices:
        time = t[idx]
        plt.plot(x, u[idx, :], label=f't = {time:.3f}')

    plt.xlabel('Position (x)')
    plt.ylabel('Temperature')
    plt.title('Heat Diffusion Over Time')
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.savefig('sample.png')


def save_chunk(chunk_data, chunk_number, base_path='data/heat_equation'):
    """Save a chunk of data to HDF5 format"""
    os.makedirs(base_path, exist_ok=True)
    
    filename = f"{base_path}/chunk_{chunk_number}.h5"
    with h5py.File(filename, 'w') as f:
        for key, value in chunk_data.items():
            if key != 'metadata':  # Handle non-metadata separately
                f.create_dataset(key, data=value, compression='gzip')
        
        # Save metadata as attributes of a group
        if 'metadata' in chunk_data:
            metadata_grp = f.create_group('metadata')
            for key, value in chunk_data['metadata'].items():
                if isinstance(value, dict):
                    # Handle nested dictionaries
                    sub_grp = metadata_grp.create_group(key)
                    for sub_key, sub_value in value.items():
                        sub_grp.attrs[sub_key] = sub_value
                else:
                    metadata_grp.attrs[key] = value

def generate_pin_dataset_in_chunks(n_samples=1000, chunk_size=100, samples_per_sim=50, 
                                 spatial_points=200, input_noise_level=0.01, 
                                 output_noise_level=0.005, base_path='data/heat_equation'):
    """
    Generate dataset in chunks and save each chunk to disk
    """
    parameter_ranges = {
        'temp1': (-20, 50),    
        'temp2': (51, 120),    
        'alpha': (1e-5, 1e-4), 
        'k': (0.01, 0.1),      
        't_env': (15, 35)      
    }
    
    # Generate unique run identifier
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = f"{base_path}/run_{run_id}"
    
    chunk_number = 0
    for chunk_start in range(0, n_samples, chunk_size):
        chunk_end = min(chunk_start + chunk_size, n_samples)
        print(f"\nGenerating chunk {chunk_number + 1} ({chunk_start} to {chunk_end})")
        
        chunk_data = {
            'initial_states': [],
            'elapsed_times': [],
            'parameters': [],
            'final_states': [],
            'clean_initial_states': [],
            'clean_final_states': []
        }
        
        for sample in range(chunk_start, chunk_end):
            # [Previous generation code remains the same until data storage]
            temp1 = np.random.uniform(*parameter_ranges['temp1'])
            temp2 = np.random.uniform(*parameter_ranges['temp2'])
            alpha = np.random.uniform(*parameter_ranges['alpha'])
            k = np.random.uniform(*parameter_ranges['k'])
            t_env = np.random.uniform(*parameter_ranges['t_env'])
            
            x, t, u = pde_run(temp1, temp2, 1.0, spatial_points, alpha, k, t_env)
            
            time_indices = np.unique(
                np.logspace(0, np.log10(len(t)-1), samples_per_sim).astype(int)
            )
            
            start_indices = np.random.choice(
                len(t)-100, size=min(5, len(t)//100), replace=False
            )
            
            for start_idx in start_indices:
                for idx in time_indices:
                    if start_idx + idx < len(t):
                        clean_initial = u[start_idx, :]
                        clean_final = u[start_idx + idx, :]
                        
                        initial_range = np.ptp(clean_initial)
                        final_range = np.ptp(clean_final)
                        
                        noisy_initial = clean_initial + np.random.normal(
                            0, input_noise_level * initial_range, size=clean_initial.shape
                        )
                        noisy_final = clean_final + np.random.normal(
                            0, output_noise_level * final_range, size=clean_final.shape
                        )
                        
                        noisy_params = np.array([alpha, k, t_env]) * (
                            1 + np.random.normal(0, 0.01, size=3)
                        )
                        
                        chunk_data['clean_initial_states'].append(clean_initial)
                        chunk_data['clean_final_states'].append(clean_final)
                        chunk_data['initial_states'].append(noisy_initial)
                        chunk_data['elapsed_times'].append(t[idx])
                        chunk_data['parameters'].append(noisy_params)
                        chunk_data['final_states'].append(noisy_final)
            
            if (sample + 1) % 10 == 0:
                print(f"Generated {sample + 1}/{n_samples} simulations")
        
        # Convert chunk data to arrays
        for key in chunk_data:
            chunk_data[key] = np.array(chunk_data[key])
        
        # Add metadata
        chunk_data['metadata'] = {
            'parameter_ranges': parameter_ranges,
            'spatial_points': spatial_points,
            'chunk_number': chunk_number,
            'chunk_size': chunk_size,
            'x_coordinates': x,
            'noise_levels': {
                'input': input_noise_level,
                'output': output_noise_level,
                'parameters': 0.01
            }
        }
        
        # Save chunk
        save_chunk(chunk_data, chunk_number, save_path)
        chunk_number += 1
        
    # Save run metadata
    run_metadata = {
        'n_samples': n_samples,
        'n_chunks': chunk_number,
        'run_id': run_id,
        'parameter_ranges': parameter_ranges,
        'spatial_points': spatial_points
    }
    
    with open(f"{save_path}/run_metadata.txt", 'w') as f:
        for key, value in run_metadata.items():
            f.write(f"{key}: {value}\n")
    
    return save_path

def merge_chunks(run_path, output_file='merged_dataset.h5'):
    """Merge all chunks in a run directory into a single file"""
    chunk_files = sorted([f for f in os.listdir(run_path) if f.startswith('chunk_')])
    
    with h5py.File(f"{run_path}/{output_file}", 'w') as merged_file:
        # Initialize datasets with first chunk to get shapes
        first_chunk = h5py.File(f"{run_path}/{chunk_files[0]}", 'r')
        
        # Create datasets in merged file
        datasets = {}
        current_idx = 0
        
        for key in first_chunk.keys():
            if key != 'metadata':
                shape = list(first_chunk[key].shape)
                maxshape = [None] + shape[1:]  # Allow first dimension to be expandable
                datasets[key] = merged_file.create_dataset(
                    key, shape=shape, maxshape=maxshape,
                    dtype=first_chunk[key].dtype, compression='gzip'
                )
        
        first_chunk.close()
        
        # Merge all chunks
        for chunk_file in chunk_files:
            with h5py.File(f"{run_path}/{chunk_file}", 'r') as chunk:
                # Get the size of this chunk
                chunk_size = chunk['initial_states'].shape[0]
                
                # Resize datasets to accommodate new data
                for key, dataset in datasets.items():
                    dataset.resize(current_idx + chunk_size, axis=0)
                    dataset[current_idx:current_idx + chunk_size] = chunk[key][:]
                
                current_idx += chunk_size
        
        print(f"Merged {len(chunk_files)} chunks successfully")
        print(f"Final dataset shapes:")
        for key, dataset in datasets.items():
            print(f"{key}: {dataset.shape}")


if __name__ == "__main__":
    # Generate data in chunks
    save_path = generate_pin_dataset_in_chunks(
        n_samples=10000,
        chunk_size=500,
        samples_per_sim=200,
        spatial_points=400
    )
    
    # Optionally merge chunks when done
    # merge_chunks(save_path)
