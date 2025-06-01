from datasets import load_dataset

# Download files locally
dataset = load_dataset("nick-leland/heat1d-pde-dataset", download_mode="force_redownload")

# Read the initial structure (h5py files)
df = dataset['train'].data.to_pandas()
file_path = df['image'][0]['path']
data = h5py.File(file_path, 'r')

# Access data
initial_states = data['initial_states'][:]
final_states = data['final_states'][:]
parameters = data['parameters'][:]
elapsed_times = data['elapsed_times'][:]
