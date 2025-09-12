import xarray as xr
import pandas as pd
import numpy as np

ds = xr.open_dataset('e:/4_Projects/1_Personal Projects/BharatBench_fork/IMDAA_merged_1.08_1990_2020.nc')
print(ds)
print("\nVariables:")
print(ds.variables)
print("\nDimensions:")
print(ds.dims)
