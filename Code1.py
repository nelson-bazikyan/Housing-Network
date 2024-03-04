import os
import math

import keras
import pandas as pd
import numpy as np
import scipy as sc
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import tensorflow as tf

os.chdir('C:/Users/Nelson/Documents/School/Projects/Housing Neural Network')

# Geographic Census Data
data = pd.read_csv('2020 census blocks latlong.csv')
data = data[np.isfinite(data['Latitude']) & np.isfinite(data['Longitude'])]
data['Latitude'] = (data['Latitude']*1000).astype(int)
data['Longitude'] = (data['Longitude']*1000).astype(int)
data = data.sort_values(['Latitude', 'Longitude'], ascending=[False, False])
data['Density'] = data['POP20'] / data['ShapeSTAre']
scalar = preprocessing.MinMaxScaler()
data['Density'] = scalar.fit_transform(data['Density'].values.reshape(-1, 1))
filter = data.duplicated(subset=['Latitude', 'Longitude'], keep=False)
data = data[~filter]

# Affordability Data
aff = pd.read_csv('Neighborhood zhvi.csv')
aff = aff[aff['CountyName'] == 'Los Angeles County']
aff['2020-06-30'] = scalar.fit_transform(aff['2020-06-30'].values.reshape(-1, 1))

# FUNCTION FOR CREATING INPUT DATA FROM A COORDINATE
def gen_grid(coord_pair: list, long_diff: int):
    y, x = coord_pair
    angles, grid_dim, longs, lats, grid = long_diff, 20, [], [], []
    df = data[(y - angles < data['Latitude']) & (data['Latitude'] < y) & (x < data['Longitude']) & (data['Longitude'] < x + angles)]
    dist = angles / grid_dim
    lats_full = list(set(df['Latitude']))

    for n in range(0, grid_dim):
        longs.append(int(x + n * dist))
        lats.append(lats_full[int(dist * n)])

    for lat in lats:
        temp = [np.nan for _ in range(grid_dim)]
        for long in longs:
            for temp_long in data[data['Latitude'] == lat]['Longitude'].tolist():
                if long - dist <= temp_long <= long + dist:
                    temp[int((long - x) / dist)] = float(data.loc[(data['Latitude'] == lat) & (data['Longitude'] == temp_long), 'Density'].values)
                    break
        grid.append(temp)

    grid = np.array(grid)
    grid[np.isnan(grid)] = 0

    target_coordinates = []
    for w in range(grid.shape[0]):
        for v in range(grid.shape[1]):
            if grid[w, v] == 0:
                target_coordinates.append((w, v))

    xx, yy = np.meshgrid(np.arange(grid.shape[1]), np.arange(grid.shape[0]))
    kdtree = sc.spatial.cKDTree(list(zip(xx.ravel(), yy.ravel())))

    for w, v in target_coordinates:
        idw_interpolation(w, v, grid, kdtree)

    grid = grid.tolist()
    community = str(df['COMM'].mode().iloc[0])
    city = str(df['CITY'].mode().iloc[0])
    try:
        HAI = float(aff[aff['RegionName'] == community]['2020-06-30'].values)
    except:
        if df['CITY'].mode().isin(['Los Angeles', 'Unincorporated']).any():
            return print(f'{community} does not have Affordability Index.')
        else:
            HAI = float(np.average(aff[aff['City'] == city]['2020-06-30'].values))
    grid = [item for sublist in grid for item in sublist]
    grid.append(HAI)
    print(f'{community} input data obtained!')
    return grid


# GRID INTERPOLATOR
def idw_interpolation(target_x, target_y, data_array, kdtree, p=3):
    distances, indices = kdtree.query([target_x, target_y], k=20)
    weights = 1 / (distances + 1e-10) ** p
    weights /= np.sum(weights)
    interpolated_value = np.sum(data_array.ravel()[indices] * weights)
    data_array[target_x, target_y] = interpolated_value

    return interpolated_value

# Creating Inputs
communities_data = set(data['COMM'].values)
communities_aff = set(aff['RegionName'].values)
communities = [comm for comm in communities_data if comm in communities_aff]

community_widths = []
northwestern_points = []
for comm in communities:
    west_point, east_point = min(data[data['COMM'] == comm]['Longitude']), max(data[data['COMM'] == comm]['Longitude'])
    northern_lat = max(data[data['COMM'] == comm]['Latitude'])
    northwest_point = [northern_lat, min(data[(data['COMM'] == comm) & (data['Latitude'] == northern_lat)]['Longitude'])]
    width = east_point - west_point
    if 30 < width < 60:
        northwestern_points.append(northwest_point)
        community_widths.append(east_point - west_point)
ideal_diff = int(np.average(community_widths))

input_list = []
for pair in northwestern_points:
    try:
        z = gen_grid(pair, ideal_diff)
        if z is not None:
            input_list.append(z)
    except:
        pass

# NEURAL NETWORK
X = np.array([sample[:-1] for sample in input_list])
Y = np.array([sample[-1] for sample in input_list])

output_mae = []

X_train, X_temp, Y_train, Y_temp = train_test_split(X, Y, test_size=0.3, random_state=0)
X_val, X_test, Y_val, Y_test = train_test_split(X_temp, Y_temp, test_size=0.5, random_state=0)

model = keras.Sequential()
model.add(keras.layers.Input(shape=(400,)))
model.add(keras.layers.Dense(16, activation='sigmoid'))
model.add(keras.layers.Dense(8, activation='sigmoid'))
model.add(keras.layers.Dense(1, activation='linear'))
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
model.fit(X_train, Y_train, epochs=15, validation_data=(X_val, Y_val))
train_loss, train_mae = model.evaluate(X_train, Y_train)
test_loss, test_mae = model.evaluate(X_test, Y_test)
print(f'{train_mae}' + ' ' + f'{test_mae}')
