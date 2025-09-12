import xarray as xr
import pandas as pd
import numpy as np
import os
from datetime import datetime

print(f"Starting conversion at {datetime.now()}")
print("Loading the NetCDF file...")
ds = xr.open_dataset('e:/4_Projects/1_Personal Projects/BharatBench_fork/IMDAA_merged_1.08_1990_2020.nc')

# Create a directory to store CSVs if it doesn't exist
csv_dir = 'csv_data'
if not os.path.exists(csv_dir):
    os.makedirs(csv_dir)

# Print basic information
print(f"Dataset dimensions: {ds.dims}")
print(f"Variables: {list(ds.data_vars)}")

# Option 1: Yearly CSV files for each variable (more manageable size)
def convert_by_year():
    # Get unique years from the time dimension
    years = np.unique(ds.time.dt.year.values)
    print(f"Processing data for years: {years}")
    
    for year in years:
        print(f"\nProcessing year {year}...")
        # Filter dataset by year
        ds_year = ds.sel(time=ds.time.dt.year == year)
        
        # Process each variable
        for var_name in ds.data_vars:
            print(f"  Converting {var_name} for {year}...")
            # Get the variable data
            var_data = ds_year[var_name]
            
            # Convert 3D data (time, lat, lon) to 2D table format
            # This creates a multi-index DataFrame
            df = var_data.to_dataframe()
            
            # Reset index to convert multi-index to columns
            df = df.reset_index()
            
            # Format the filename
            csv_filename = os.path.join(csv_dir, f"{var_name}_{year}.csv")
            
            # Save to CSV
            print(f"  Saving to {csv_filename}...")
            df.to_csv(csv_filename, index=False)
            
            print(f"  Completed {var_name} for {year} - Shape: {df.shape}")

# Option 2: One CSV with all variables for specific time periods (comprehensive but larger files)
def convert_by_period(period_years=5):
    # Get min and max years
    min_year = ds.time.dt.year.values.min()
    max_year = ds.time.dt.year.values.max()
    
    # Process data in periods of X years
    for start_year in range(min_year, max_year + 1, period_years):
        end_year = min(start_year + period_years - 1, max_year)
        print(f"\nProcessing period {start_year}-{end_year}...")
        
        # Filter dataset by year range
        ds_period = ds.sel(time=(ds.time.dt.year >= start_year) & (ds.time.dt.year <= end_year))
        
        # Create a single DataFrame with all variables
        # First convert a variable to get the structure
        first_var = list(ds.data_vars)[0]
        df = ds_period[first_var].to_dataframe()
        
        # Add other variables
        for var_name in list(ds.data_vars)[1:]:
            df[var_name] = ds_period[var_name].values.flatten()
        
        # Reset index to convert multi-index to columns
        df = df.reset_index()
        
        # Format the filename
        csv_filename = os.path.join(csv_dir, f"all_vars_{start_year}_{end_year}.csv")
        
        # Save to CSV
        print(f"  Saving to {csv_filename}...")
        df.to_csv(csv_filename, index=False)
        
        print(f"  Completed period {start_year}-{end_year} - Shape: {df.shape}")

# Option 3: Extract time series for specific grid points
def extract_gridpoints(lat_indices=[10, 20], lon_indices=[10, 20]):
    print("\nExtracting time series for specific grid points...")
    
    for lat_idx in lat_indices:
        for lon_idx in lon_indices:
            lat_val = ds.lat.values[lat_idx]
            lon_val = ds.lon.values[lon_idx]
            
            print(f"  Extracting data for point lat={lat_val}, lon={lon_val}...")
            
            # Create DataFrame for this grid point
            df = pd.DataFrame({'time': ds.time.values})
            
            # Add all variables
            for var_name in ds.data_vars:
                df[var_name] = ds[var_name].isel(latitude=lat_idx, longitude=lon_idx).values
            
            # Format the filename
            csv_filename = os.path.join(csv_dir, f"point_lat{lat_val}_lon{lon_val}.csv")
            
            # Save to CSV
            df.to_csv(csv_filename, index=False)
            print(f"  Saved {csv_filename}")

# Choose which conversion method to run
print("\nChoose a conversion method:")
print("1. Convert by year (one CSV per variable per year)")
print("2. Convert by time period (one CSV for all variables in a multi-year period)")
print("3. Extract time series for specific grid points")

# Default to Option 1 which is most manageable
choice = 1

if choice == 1:
    convert_by_year()
elif choice == 2:
    convert_by_period(period_years=5)
elif choice == 3:
    # Sample grid points at 25%, 50%, 75% of grid dimensions
    lat_indices = [int(ds.dims['latitude'] * p) for p in [0.25, 0.5, 0.75]]
    lon_indices = [int(ds.dims['longitude'] * p) for p in [0.25, 0.5, 0.75]]
    extract_gridpoints(lat_indices, lon_indices)
else:
    print("Invalid choice. Defaulting to convert by year.")
    convert_by_year()

print(f"\nConversion completed at {datetime.now()}")
