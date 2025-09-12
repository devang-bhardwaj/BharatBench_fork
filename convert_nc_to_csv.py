import xarray as xr
import pandas as pd
import numpy as np
import os

print("Loading the NetCDF file...")
ds = xr.open_dataset('e:/4_Projects/1_Personal Projects/BharatBench_fork/IMDAA_merged_1.08_1990_2020.nc')

# Create a directory to store CSVs if it doesn't exist
csv_dir = 'csv_data'
if not os.path.exists(csv_dir):
    os.makedirs(csv_dir)

# Extract data variables
variables = list(ds.data_vars)
print(f"Variables to process: {variables}")

# Get lat and lon values
lat_values = ds.lat.values
lon_values = ds.lon.values

# Function to convert a single variable to CSV
def convert_var_to_csv(ds, var_name):
    print(f"Processing {var_name}...")
    # Extract the data for this variable
    var_data = ds[var_name]
    
    # Convert to a pandas DataFrame
    # Create a list to hold all rows
    rows = []
    
    # Iterate through all time steps
    for t_idx, t in enumerate(ds.time.values):
        if t_idx % 1000 == 0:  # Progress indicator
            print(f"  Time step {t_idx}/{len(ds.time.values)}")
            
        # For each time step, extract data for all grid points
        time_str = pd.Timestamp(t).strftime('%Y-%m-%d %H:%M:%S')
        
        for lat_idx, lat in enumerate(lat_values):
            for lon_idx, lon in enumerate(lon_values):
                rows.append({
                    'time': time_str,
                    'latitude': lat,
                    'longitude': lon,
                    var_name: var_data.values[t_idx, lat_idx, lon_idx]
                })
                
    # Convert to DataFrame
    df = pd.DataFrame(rows)
    
    # Save to CSV
    csv_filename = os.path.join(csv_dir, f"{var_name}.csv")
    df.to_csv(csv_filename, index=False)
    print(f"Saved {var_name} data to {csv_filename}")
    
    # For memory efficiency, return size of processed data
    return len(df)

# Main processing
for var_name in variables:
    total_rows = convert_var_to_csv(ds, var_name)
    print(f"Processed {var_name}: {total_rows} rows")

print("Conversion complete!")
