import os
import numpy as np
import netCDF4 as nc
import argparse
import pathlib
import time as Time
import psutil
import xarray as xr
from multiprocessing.pool import Pool
import pandas as pd

file_path = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.dirname(file_path)
input_npy_path = os.path.join(project_path, 'examples', 'output_data')
example_nc_path = os.path.join(project_path, 'examples', 'whole_nc_data')

parser = argparse.ArgumentParser(description='argparse')
parser.add_argument("--input_npy_path", type=str, default=input_npy_path,
                    help="Path containing surface and deep NPY data, e.g. {0}".format(input_npy_path))
parser.add_argument("--nc_path", type=str, default=example_nc_path, help="Output path for generated NC files, e.g. {0}".format(example_nc_path))
parser.add_argument("--targetdate", type=str, required=True, help="Target date in YYYYMMDD format, e.g. 20220101")
args = parser.parse_args()

input_surface_path = os.path.join(args.input_npy_path, 'output_surface_data')
input_deep_path = os.path.join(args.input_npy_path, 'output_deep_data')

# Compatible with the current inference directory structure: <input_npy_path>/surface and <input_npy_path>/deep
if not os.path.isdir(input_surface_path) and os.path.isdir(os.path.join(args.input_npy_path, 'surface')):
    input_surface_path = os.path.join(args.input_npy_path, 'surface')
if not os.path.isdir(input_deep_path) and os.path.isdir(os.path.join(args.input_npy_path, 'deep')):
    input_deep_path = os.path.join(args.input_npy_path, 'deep')

pathlib.Path(args.nc_path).mkdir(exist_ok=True, parents=True)  # Create the NC output directory if it does not exist

# latitude & longitude
lat_path = os.path.join(file_path, 'mercator_lat.npy')
lon_path = os.path.join(file_path, 'mercator_lon.npy')
lat, lon = np.load(lat_path), np.load(lon_path)

depth = np.array(
    [0.4940, 2.6457, 5.0782, 7.9296, 11.4050, 15.8101, 21.5988, 29.4447, 40.3441, 55.7643, 77.8539, 92.3261, 109.7293,
     130.6660, 155.8507, 186.1256, 222.4752, 266.0403, 318.1274, 380.2130, 453.9377, 541.0889, 643.5668])

# time
time = np.array([1])
start = Time.time()

# variables
variables = ['thetao', 'so', 'uo', 'vo', 'zos', 'sst']
layer_index = [[2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50, 54, 58, 62, 66, 70, 74, 78, 82, 86, 90],
               [3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63, 67, 71, 75, 79, 83, 87, 91],
               [4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92],
               [5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 49, 53, 57, 61, 65, 69, 73, 77, 81, 85, 89, 93],
               0,
               1]

target_date_str = str(args.targetdate)
npy_files = [
    f for f in os.listdir(input_deep_path)
    if f.endswith(".npy") and f[-12:-4] == target_date_str
]
npy_files.sort()

if not npy_files:
    raise FileNotFoundError(
        f"No NPY file for date {target_date_str} was found in {input_deep_path}"
    )


def task_file(file):
    print("Processing file [", file, "]")
    npy_data = np.load(os.path.join(input_surface_path, file))
    zos_data = npy_data[:,0:1,:,:]
    sst_temp_so_uovo = npy_data[:,3:,:,:]
    npy_data = np.concatenate([zos_data, sst_temp_so_uovo],axis=1)
    npy_data_2 = np.load(os.path.join(input_deep_path, file))
    npy_data = np.concatenate([npy_data, npy_data_2], axis=1)

    init_day = input_surface_path.split("/")[-3]
    file_date_str = file[-12:-4]

    # Create a date-specific subdirectory under the output root, e.g. <nc_path>/20220101/
    time_path = os.path.join(args.nc_path, file_date_str)
    pathlib.Path(time_path).mkdir(exist_ok=True, parents=True)

    for i in range(len(variables)):
        # Keep the original NC filename format unchanged and save the file under the corresponding date directory
        nc_file_path = os.path.join(time_path, "XiHe_DataFu_" + init_day + "_" + file_date_str + '_' + variables[i] + '.nc')               
        if i < 4:
            dims = ['time', 'depth','latitude','longitude']
            coords_dict = {}
            coords_dict['time']=pd.date_range(file[-12:-8] + "-" + file[-8:-6] + "-" + file[-6:-4] + " 00:00:00", periods=1)
            coords_dict['depth']=depth.astype(np.float32)
            coords_dict['latitude']=lat.astype(np.float32)
            coords_dict['longitude']=lon.astype(np.float32)
            new_dataset = xr.DataArray(npy_data[:, layer_index[i], :, :].astype(np.float32), dims=dims, coords=coords_dict).to_dataset(name=variables[i])
            new_dataset.to_netcdf(nc_file_path)
        else:
            dims = ['time','latitude','longitude']
            coords_dict = {}
            #coords_dict['t']=[np.datetime64('2023-11-01 00:00:00')]
            coords_dict['time']=pd.date_range(file[-12:-8] + "-" + file[-8:-6] + "-" + file[-6:-4] + " 00:00:00", periods=1)
            coords_dict['latitude']=lat.astype(np.float32)
            coords_dict['longitude']=lon.astype(np.float32)
            new_dataset = xr.DataArray(npy_data[:, layer_index[i], :, :].astype(np.float32), dims=dims, coords=coords_dict).to_dataset(name=variables[i])
            new_dataset.to_netcdf(nc_file_path)
    Time.sleep(1)


if __name__ == '__main__':
  available_value = int(psutil.virtual_memory().available / (1024 ** 3) * 0.8)
  course_value = 16
  pool_max = int(available_value / course_value)
  # Get the number of available CPU logical cores
  pool_cpu = psutil.cpu_count()
  # Set the process count
  pool_num = 0
  if pool_max > pool_cpu:
    pool_num = int(pool_cpu / 2)
  else:
    pool_num = pool_max
  with Pool(pool_num) as pool:
    pool.map(task_file, npy_files)
  end = Time.time()
  run_time = end - start
  print("Data merge time:", run_time, "seconds")
  print("END")
