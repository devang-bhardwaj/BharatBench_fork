# BharatBench Project Documentation

## Overview

BharatBench is a dataset for data-driven weather forecasting over India. It includes meteorological data derived from the IMDAA (Indian Monsoon Data Assimilation and Analysis) reanalysis datasets, specifically prepared for AI/ML applications in weather forecasting.

This repository contains an implementation of various weather forecasting models using the BharatBench dataset, ranging from baseline statistical models to deep learning approaches.

## Dataset

The dataset (`IMDAA_merged_1.08_1990_2020.nc`) is in NetCDF format and contains the following meteorological variables:

| Variable | Description | Units | Dimension |
|----------|-------------|-------|-----------|
| HGT_prl  | Geopotential Height at 500hPa | m | (time, latitude, longitude) |
| TMP_prl  | Temperature at 850hPa | K | (time, latitude, longitude) |
| TMP_2m   | 2m Temperature | K | (time, latitude, longitude) |
| APCP_sfc | Total Precipitation (6 hourly accumulated) | kg/m² | (time, latitude, longitude) |

The dataset covers a period from 1990 to 2020 with data points every 6 hours (00, 06, 12, and 18 UTC daily), resulting in 45,292 time steps. The spatial coverage is a 32×32 grid over the Indian subcontinent with a resolution of 1.08 degrees.

## Implemented Models

The repository implements and evaluates several weather forecasting models:

### 1. Baseline Models (`1_climatology_persistence.ipynb`)

This notebook implements two simple baseline models:

#### Persistence Model

The persistence model assumes that weather conditions at a future time step will be the same as the current conditions. This naive approach provides a lower benchmark for more sophisticated models.

```python
persistence_fc = ds.sel(time=test_years).isel(time=slice(0, -j))
persistence_fc['time'] = persistence_fc.time + np.timedelta64(i+1, 'D')
```

#### Climatology Models

- **Daily Climatology**: Uses the historical average for each day of the year
- **Weekly Climatology**: Uses the historical average for each week of the year

```python
clim = ds.sel(time=train_years).groupby('time.dayofyear').mean()
```

### 2. Linear Regression Model (`2_Linear_Regression.ipynb`)

This notebook implements linear regression models for each meteorological variable:

- Uses normalized data (z-score normalization)
- Trains separate models for each variable
- Evaluates performance with RMSE, MAE, and ACC (Anomaly Correlation Coefficient)

The models are built using scikit-learn's LinearRegression implementation:

```python
lr = LinearRegression(n_jobs=16)
lr.fit(X_train, y_train)
```

### 3. Deep Learning Models (`3_CNN.ipynb`)

This notebook implements two deep learning approaches:

#### Convolutional Neural Network (CNN)

A CNN model with multiple convolution, pooling, and upsampling layers:

```python
model = keras.Sequential([
    keras.layers.Conv2D(32, 5, padding='same', activation='swish'),
    keras.layers.MaxPooling2D(),
    # ...more layers...
    keras.layers.Conv2D(1, 5, padding='same')
])
```

#### ConvLSTM (Convolutional LSTM)

A ConvLSTM model that combines convolution operations with LSTM recurrence:

```python
model = keras.Sequential([
    keras.layers.ConvLSTM2D(filters=32, kernel_size=(5, 5), padding="same", 
                          return_sequences=True, activation='swish'),
    keras.layers.MaxPooling3D(pool_size=(1, 2, 2)),
    # ...more layers...
    keras.layers.Conv3D(filters=1, kernel_size=(5, 5, 5), padding="same")
])
```

## Evaluation Metrics

All models are evaluated using three key metrics:

### Root Mean Square Error (RMSE)

Measures the average magnitude of error between predicted and actual values:

```python
def compute_rmse(prediction, actual, mean_dims=('time', 'latitude', 'longitude')):
    error = prediction - actual
    rmse = np.sqrt(((error)**2).mean(mean_dims))
    return rmse
```

### Mean Absolute Error (MAE)

Measures the average absolute difference between predicted and actual values:

```python
def compute_mae(prediction, actual, mean_dims=('time', 'latitude', 'longitude')):
    error = prediction - actual
    mae = np.abs(error).mean(mean_dims)
    return mae
```

### Anomaly Correlation Coefficient (ACC)

Measures the correlation between predicted and actual anomalies relative to climatology:

```python
def compute_acc(prediction, actual):
    clim = actual.mean('time')
    # ...calculation...
    acc = (np.sum(pred_norm * act_norm) / 
          np.sqrt(np.sum(pred_norm ** 2) * np.sum(act_norm ** 2)))
    return acc
```

## Data Splitting

The dataset is split into three periods:

- **Training set**: 1990-2017
- **Validation set**: 2018
- **Test set**: 2019-2020

## Model Training and Evaluation Process

1. Data preprocessing:
   - Normalization (z-score using training set statistics)
   - Splitting into train/validation/test sets

2. Model training:
   - For baseline models: Computing climatology or implementing persistence logic
   - For linear regression: Training with scikit-learn
   - For deep learning models: Training with TensorFlow/Keras

3. Model evaluation:
   - Computing RMSE, MAE, and ACC for each variable
   - Visualizing error metrics across lead times (especially for persistence models)

4. Visualization:
   - Plotting error metrics against lead time
   - Using Cartopy for geographic visualizations
   - Model architecture visualization with visualkeras

## Data Conversion Scripts

Two scripts are provided to convert the NetCDF dataset to CSV format for easier use with other tools:

1. `convert_nc_to_csv.py`: Basic conversion of each variable to separate CSV files
2. `convert_nc_to_csv_optimized.py`: Optimized conversion with multiple options:
   - Conversion by year (separate CSVs for each year and variable)
   - Conversion by time period (all variables for a multi-year period)
   - Extraction of time series for specific grid points

## Key Findings

1. **Persistence Model Performance**: Error increases with lead time, as expected.
2. **Climatology Performance**: Weekly climatology generally outperforms daily climatology.
3. **Linear Regression**: Shows improvement over baselines for most variables.
4. **Deep Learning**: CNN and ConvLSTM models demonstrate superior performance, particularly for:
   - Temperature variables (TMP_prl, TMP_2m)
   - Geopotential height (HGT_prl)
   - Precipitation forecasting (though this remains challenging)

## References

Original BharatBench dataset citation:
Choudhury, A., Panda, J., & Mukherjee, A. (2024). BharatBench: Dataset for data-driven weather forecasting over India. arXiv [Physics.Ao-Ph]. (<http://arxiv.org/abs/2405.07534>)

## Usage

To use this repository:

1. Download the BharatBench dataset from Kaggle (<https://www.kaggle.com/datasets/maslab/bharatbench>)
2. Place the IMDAA_merged_1.08_1990_2020.nc file in the repository root
3. Run the notebooks in sequence to replicate the results
4. Use the conversion scripts if CSV format is required
