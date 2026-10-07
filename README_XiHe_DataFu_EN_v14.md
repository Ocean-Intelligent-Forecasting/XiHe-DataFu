# XiHe-DataFu

## 1. Project Overview

XiHe-DataFu is a data fusion deep neural network designed for integrating multi-source ocean observations to generate global 1/12° ocean analysis products for eddy-resolving forecasting.

The complete workflow is executed through `XiHe_DataFu_run.sh`.

---

## 2. Project Structure

After downloading and preparing the code, models, observation data, background fields, and auxiliary files, organize the project according to the following directory structure:

```text
XiHe_DataFu/
├── src/
│   ├── XiHe_DataFu_run.sh
│   ├── reconstruction_obs_preprocess.py
│   ├── assimilation_obs_preprocess.py
│   ├── assimilation_fusion_inference.py
│   ├── reconstruction_inference.py
│   ├── reconstruction_obs_preprocess_step.py
│   ├── assimilation_obs_preprocess_step.py
│   ├── reconstruction_config.yaml
│   ├── assimilation_config.yaml
│   └── npy_to_nc/
│
├── glorys12_nc_to_npy/
│   ├── glorys12_nc_to_npy.py
│   ├── input/
│   └── temp/
│       ├── get_variables/
│       ├── get_depth/
│       └── npy_output/
│
├── observations/
│   ├── sla_data/
│   ├── sst_data/
│   ├── surface_current_data/
│   └── sss_data/
│
├── reconstruction_models/
│   ├── reconstruction_best_onnx_1to22.onnx
│   └── reconstruction_best_onnx_23to33.onnx
│
├── assimilation_models/
│   ├── assi_best_1to22.onnx
│   └── assi_best_23to33.onnx
│
├── reconstruction_workspace/
│   ├── input/
│   └── output/
│
└── assimilation_workspace/
    ├── background/
    └── output/
```

Main directory descriptions:

- `src/`: Main processing code for XiHe_DataFu.
- `glorys12_nc_to_npy/`: Preprocessing tools for converting raw NetCDF data into background-field NPY files.
- `observations/`: Stores observation data for sea level anomaly, sea surface temperature, sea surface currents, and sea surface salinity.
- `reconstruction_models/`: Stores ocean reconstruction models.
- `assimilation_models/`: Stores data assimilation models.
- `reconstruction_workspace/`: Stores input and output data for the reconstruction stage.
- `assimilation_workspace/background/`: Stores historical background fields required for data assimilation.
- `assimilation_workspace/output/`: Stores the final generated NetCDF products.

The ONNX models, normalization parameters, Mask files, and sample data required by the project will be available for download at XXX (coming soon). Please keep the default directory structure unchanged.

---

## 3. Environment Setup

The project runs in a Linux environment.

To avoid compatibility issues caused by different Python and dependency versions, it is recommended to directly use the `pycdo` environment package provided with this project:

```text
pycdo.tar.gz
```

Download address:

```text
{{PYCDO_DOWNLOAD_URL}}
```

---

## 4. Data Preparation

### 4.1 Observation Data

XiHe_DataFu uses multi-source ocean observation data as model input.

| Observation Data Type | Main Variables | Description |
|---|---|---|
| Sea level anomaly observations | SLA | Sea level anomaly |
| Sea surface temperature observations | SST | Sea surface temperature |
| Sea surface current observations | `uo`, `vo` | Zonal and meridional ocean currents |
| Sea surface salinity observations | SSS | Sea surface salinity |

Place the downloaded observation data in the following directories:

```text
observations/
├── sla_data/
├── sst_data/
├── surface_current_data/
└── sss_data/
```

---

### 4.2 Background Field Data

When XiHe_DataFu starts the data assimilation workflow for the first time, consecutive historical background fields are required for initialization. After the first assimilation is completed, the background fields used for subsequent dates are automatically updated using the assimilation outputs from the previous stage, so historical background fields do not need to be prepared manually again.

The initial background field data used in this project are obtained from Copernicus Marine Service:

[**Global Ocean Physics Analysis and Forecast**](https://data.marine.copernicus.eu/product/GLOBAL_ANALYSISFORECAST_PHY_001_024/files?path=GLOBAL_ANALYSISFORECAST_PHY_001_024%2Fcmems_mod_glo_phy_anfc_0.083deg_P1D-m_202406%2F&subdataset=cmems_mod_glo_phy-so_anfc_0.083deg_P1D-m_202406)

The project mainly uses the following variables:

| Variable | Description |
|---|---|
| `thetao` | Seawater temperature |
| `so` | Seawater salinity |
| `uo` | Zonal ocean current velocity |
| `vo` | Meridional ocean current velocity |
| `zos` | Sea surface height |

For temperature, salinity, and ocean current variables, the following 23 depth levels are used (unit: m):

```text
0.49, 2.65, 5.08, 7.93, 11.41, 15.81,
21.60, 29.44, 40.34, 55.76, 77.85, 92.32,
109.73, 130.67, 155.85, 186.13, 222.48,
266.04, 318.13, 380.21, 453.94, 541.09, 643.57
```

The project provides processed NPY background field examples for `20211229`, `20211230`, and `20211231` in `assimilation_workspace/background/`. These files can be used directly to initialize the first assimilation for `20220101`.

> If you directly use the background field example data provided with the project, there is no need to perform the background field preprocessing steps below. You can skip directly to **Section 5: Running the Project**.

If background field data for other dates are required, download the corresponding raw NetCDF files from the official website and process them into the NPY format required by the model using the following steps.

#### Background Field Preprocessing

After downloading the raw NetCDF files for the target dates from the official website, place them in:

```text
glorys12_nc_to_npy/input/
```

Enter the `glorys12_nc_to_npy/` directory and run:

```bash
python glorys12_nc_to_npy.py \
    --input-dir ./input \
    --example-dir ./example_data \
    --output-dir ./temp \
    --start-date 20211229 \
    --end-date 20211231
```

Parameter descriptions:

- `--input-dir`: Directory containing the raw NetCDF data.
- `--example-dir`: Directory containing the template NetCDF file.
- `--output-dir`: Output directory for the processed background field data.
- `--start-date`: Start date for processing, in `YYYYMMDD` format.
- `--end-date`: End date for processing, in `YYYYMMDD` format.

The resulting background field NPY files are located in:

```text
temp/npy_output/input_surface_output/
temp/npy_output/input_deep_output/
```

After background field preprocessing is completed, copy the generated NPY files to the corresponding directories under `assimilation_workspace/background/` for the first XiHe_DataFu assimilation:

```text
glorys12_nc_to_npy/temp/npy_output/input_surface_output/
→ assimilation_workspace/background/surface/

glorys12_nc_to_npy/temp/npy_output/input_deep_output/
→ assimilation_workspace/background/deep/
```

Before running XiHe_DataFu for the first time, prepare **3 consecutive days** of historical background fields before the target assimilation date.

For example, to assimilate observation data for `20220101`, download and process the following dates in advance:

```text
20211229
20211230
20211231
```

These three days of background field data are used to initialize the first assimilation for `20220101`.

After processing is complete, place the background field data in:

```text
assimilation_workspace/background/
├── surface/
└── deep/
```

Background field files use the following naming format:

```text
mra5_YYYYMMDD.npy
```

After one day of assimilation is completed, the newly generated assimilation result is automatically used as a historical background field for subsequent dates. The program keeps the most recent three background fields for the next assimilation.

---

## 5. Running the Project

After completing the environment and data preparation described above, XiHe_DataFu can be run.

Enter the main source code directory:

```bash
cd src
```

Run:

```bash
bash XiHe_DataFu_run.sh YYYYMMDD
```

For example, to process `2022-01-01`:

```bash
bash XiHe_DataFu_run.sh 20220101
```

The program automatically performs observation data preprocessing, ocean variable reconstruction, assimilation observation field construction, data assimilation, and NetCDF product generation.

---

## 6. Output Results

After data assimilation for the current date is completed, the model first generates the corresponding NPY result.

The assimilation NPY result for the current date is then converted to NetCDF and saved as the final product.

The final NetCDF products are saved in:

```text
assimilation_workspace/output/YYYYMMDD/
```

For example:

```text
assimilation_workspace/output/20220101/
├── XiHe_DataFu_TEM_20220101.nc
├── XiHe_DataFu_SAL_20220101.nc
├── XiHe_DataFu_CUR_20220101.nc
├── XiHe_DataFu_CVR_20220101.nc
├── XiHe_DataFu_SLA_20220101.nc
└── XiHe_DataFu_SST_20220101.nc
```

The products are described below:

| Product | Description | Unit |
|---|---|---|
| `TEM` | Ocean temperature | °C |
| `SAL` | Ocean salinity | PSU |
| `CUR` | Zonal ocean current velocity | m/s |
| `CVR` | Meridional ocean current velocity | m/s |
| `SLA` | Sea level anomaly | m |
| `SST` | Sea surface temperature | °C |

Where:

- `TEM`, `SAL`, `CUR`, and `CVR` are three-dimensional ocean variables containing 23 depth levels.
- `SLA` and `SST` are two-dimensional sea surface variables.

Newly generated NPY files (for example, `pred_mra5_20220101.npy`) are automatically saved to the background field directory and used as background fields for subsequent assimilation dates. At the same time, the program deletes the oldest NPY file in the background field directory.

For example, before assimilating `20220101`, the background field directory contains `mra5_20211229.npy`, `mra5_20211230.npy`, and `mra5_20211231.npy`. After the assimilation for `20220101` is completed, the newly generated `pred_mra5_20220101.npy` is automatically added to the background field directory, while the oldest file, `mra5_20211229.npy`, is deleted. The directory then contains `mra5_20211230.npy`, `mra5_20211231.npy`, and `pred_mra5_20220101.npy`, which are used for the next observation assimilation.

Therefore, for continuous execution, historical background fields only need to be prepared manually before the first assimilation. Subsequent background fields are automatically updated in a rolling manner using model outputs.

---

## 7. Notes

Before running the project, make sure that:

- Observation data for the target date have been prepared.
- The reconstruction models and assimilation models are located in the correct directories.
- The normalization parameters and Mask files are complete.
- The three consecutive historical background fields required for the first run have been prepared.
- The `pycdo` environment has been configured correctly.
- CDO can be called normally.
- The project paths and data paths are consistent with the local environment.

Intermediate data and logs are generated during project execution. After the complete workflow finishes, intermediate processing data are automatically cleaned up, while the final NetCDF products and the latest background field data are retained.
