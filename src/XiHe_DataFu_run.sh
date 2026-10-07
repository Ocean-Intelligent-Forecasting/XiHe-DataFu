#!/bin/bash
###############Record script start time################
START_TIME=$(date +%s)
START_TIME_STR=$(date -d "@$START_TIME" +"%Y-%m-%d %H:%M:%S")
##########Validate input arguments########################
if [ $# -ne 1 ]; then
    echo "Error: one argument is required: the start date in YYYYMMDD format"
    echo "Usage: bash $0 param1"
    exit 1
fi

set -eo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$SCRIPT_DIR" 
# ==============================
DATA_NASE_DIR="$( dirname "$SCRIPT_DIR")/src/output_path"   #Intermediate-data output root directory
PUBLIC_OBSDATA_PATH="$( dirname "$SCRIPT_DIR")/observations"
mkdir -p $DATA_NASE_DIR
target_date=$1

DATE_FORMATTED="${target_date:0:4}-${target_date:4:2}-${target_date:6:2}"
START_DATE=$DATE_FORMATTED
END_DATE=$DATE_FORMATTED
# ------------------------------
# Environment configuration: specify Conda environments by stage
# ------------------------------
RECON_ENV_NAME="pycdo"          # Environment dedicated to the CDO stage
#Assimilation NPY output path
ASSI_BACKGROUND_INPUT="$( dirname "$SCRIPT_DIR")/assimilation_workspace/background"
FINALL_ASSI_NPY_OUTPUT_DIR_surface="${ASSI_BACKGROUND_INPUT}/surface"  #Fixed path used for both the initial background fields and assimilation background fields; later autoregressive outputs update the background fields, keeping a length of 3.
FINALL_ASSI_NPY_OUTPUT_DIR_deep="${ASSI_BACKGROUND_INPUT}/deep"
# ------------------------------
# Observation inputs (shared)
# ------------------------------
AVISO_INPUT="${PUBLIC_OBSDATA_PATH}/aviso_data"  #Modify according to the local data path
GHRSST_INPUT="${PUBLIC_OBSDATA_PATH}/ghrsst_data"
CMEMS_INPUT="${PUBLIC_OBSDATA_PATH}/cmems_data"
SMAP_INPUT="${PUBLIC_OBSDATA_PATH}/smap_data"
# ------------------------------
# 1_recon Reconstruction outputs (data fusion & model inference)
# ------------------------------
RECON_AVISO_OUT="${DATA_NASE_DIR}/aviso_data"
mkdir -p $RECON_AVISO_OUT   

RECON_SST_OUT="${DATA_NASE_DIR}/ghrsst_data"
mkdir -p $RECON_SST_OUT

RECON_SMAP_OUT="${DATA_NASE_DIR}/smap_data"
mkdir -p $RECON_SMAP_OUT

RECON_CMEMS_OUT="${DATA_NASE_DIR}/cmems_data"
mkdir -p $RECON_CMEMS_OUT


RECON_FUSION_OUT="$( dirname "$SCRIPT_DIR")/reconstruction_workspace/input"

RECON_INF_OUT="$( dirname "$SCRIPT_DIR")/reconstruction_workspace/output"

# ------------------------------
# 2_assi Assimilation data-processing outputs (the assimilation inference input/output paths are fixed in the code; see the README for details) 
# ------------------------------
ASSI_AVISO_OUT="${DATA_NASE_DIR}/th_intermediate/assi_obsdata_cdo_result/aviso_data"
mkdir -p $ASSI_AVISO_OUT

ASSI_GHRSST_OUT="${DATA_NASE_DIR}/th_intermediate/assi_obsdata_cdo_result/ghrsst_data"
mkdir -p $ASSI_GHRSST_OUT

ASSI_CMEMS_OUT="${DATA_NASE_DIR}/th_intermediate/assi_obsdata_cdo_result/cmemes_data"
mkdir -p $ASSI_CMEMS_OUT

ASSI_FUSION_OUT="${DATA_NASE_DIR}/th_intermediate/assi_fusion_result/assi_fusion_data"
mkdir -p $ASSI_FUSION_OUT

# ------------------------------
# 3_Path configuration for fusing reconstruction data with assimilation observations
# ------------------------------
# 3.1_Path configuration for fusing shallow reconstruction data into the assimilation observation field
FUSION_SURFACE_NPY_ROOT="${ASSI_FUSION_OUT}/datalight"  # Output path after fusing 1/12° observations
FUSION_SURFACE_OUTPUT="$( dirname "$SCRIPT_DIR")/assimilation_workspace/observation/surface"  
mkdir -p $FUSION_SURFACE_OUTPUT

# 3.2_Path configuration for fusing deep reconstruction data into the assimilation observation field
FUSION_DEEP_NPY_ROOT="${ASSI_FUSION_OUT}/datadeep"  # Assimilation observation-data path; datadeep is created automatically
FUSION_DEEP_OUTPUT="$( dirname "$SCRIPT_DIR")/assimilation_workspace/observation/deep"  # Output path after reconstruction data are fused into assimilation observations
mkdir -p $FUSION_DEEP_OUTPUT

# Automatic CPU configuration
# ------------------------------
TOTAL_CPUS=$(nproc)
CDO_PROC=$(( TOTAL_CPUS / 4 ))
WORKERS=$(( TOTAL_CPUS / 4 ))
CDO_PROC=$(( CDO_PROC < 1 ? 1 : CDO_PROC ))
WORKERS=$(( WORKERS < 1 ? 1 : WORKERS ))

# ------------------------------
# Logs
# ------------------------------
LOG_DIR="${BASE_DIR}/logs_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${LOG_DIR}"

# # ------------------------------
# # Initialize Conda (allows environment activation in non-interactive shells)
# # ------------------------------
CONDA_INIT_PATH="$HOME/miniconda3/etc/profile.d/conda.sh"  # Adjust according to the actual Conda path
 if [ -f "${CONDA_INIT_PATH}" ]; then
     source "${CONDA_INIT_PATH}"
 else
     echo "WARNING: Conda initialization file not found; trying the system default path..."
     source /etc/profile.d/conda.sh || true
 fi

# Date conversion: convert YYYY-MM-DD to YYYYMMDD for the fusion-script argument format
START_DATE_YYYYMMDD=$(echo "${START_DATE}" | sed 's/-//g')
END_DATE_YYYYMMDD=$(echo "${END_DATE}" | sed 's/-//g')

echo "=================================================="
echo " Full workflow: 1_observation preprocessing -> 2_reconstruction inference -> 3_assimilation inference -> 4_npy2nc"
echo "Date: ${target_date}"
echo "Log directory: ${LOG_DIR}"
echo "CDO environment: ${RECON_ENV_NAME} | inference environment: ${INFERENCE_ENV_NAME}"
echo "=================================================="

# ==================================================
# Stage 0: activate the CDO environment (pycdo)
# ==================================================
echo -e "\n==== Initialization: activate CDO environment ${RECON_ENV_NAME} ===="
if conda info --envs | grep -q "${RECON_ENV_NAME}"; then
    conda activate "${RECON_ENV_NAME}"
    echo "Successfully activated CDO environment: ${RECON_ENV_NAME}"
else
    echo "ERROR: Conda environment not found: ${RECON_ENV_NAME}; exiting workflow!"
    exit 1
fi
# =============================================================================
# Stage 1: reconstruction observation preprocessing (01-05) in parallel with pycdo active, then fusion        
# ============================================================================  

python reconstruction_obs_preprocess.py \
  --start-date "$START_DATE" --end-date "$END_DATE" \
  --aviso-dir "$AVISO_INPUT"   --aviso-out "$RECON_AVISO_OUT" \
  --ghrsst-dir "$GHRSST_INPUT" --sst-out   "$RECON_SST_OUT" \
  --smap-dir "$SMAP_INPUT"     --smap-out  "$RECON_SMAP_OUT" \
  --cmems-dir "$CMEMS_INPUT"   --cmems-out "$RECON_CMEMS_OUT" \
  --output-dir "$RECON_FUSION_OUT" \
  --processes "$CDO_PROC"

echo "Stage 1 reconstruction observation preprocessing completed"

# ==================================================
# Stage 2: reconstruction model inference
# ==================================================
echo -e "\n==== Reconstruction model inference stage ===="
python "$( dirname "$SCRIPT_DIR")/src/reconstruction_inference.py" \
        --input_root "$RECON_FUSION_OUT" --start_date "$target_date" --end_date "$target_date" \
        --mode cpu \
        --gpu_devices 0\
        --save_root "$RECON_INF_OUT"
echo "Reconstruction model inference completed"

# =============================================================================
# Stage 3: assimilation observation preprocessing (07-09) in parallel using pycdo, then fusion
# ==============================================================================

python assimilation_obs_preprocess.py \
  --start-date "$START_DATE" --end-date "$END_DATE" \
  --aviso-dir  "$AVISO_INPUT"   --aviso-out  "$ASSI_AVISO_OUT" \
  --ghrsst-dir "$GHRSST_INPUT"  --ghrsst-out "$ASSI_GHRSST_OUT" \
  --cmems-dir  "$CMEMS_INPUT"   --cmems-out  "$ASSI_CMEMS_OUT" \
  --output-dir "$ASSI_FUSION_OUT" \
  --log-name   "${LOG_DIR}/assi_obs_preprocess.log"


# =====================================================
# Stage 4: fuse assimilation observations with reconstruction outputs and run assimilation inference
# ====================================================

python assimilation_fusion_inference.py \
  --start_date "${START_DATE_YYYYMMDD}" --end_date "${END_DATE_YYYYMMDD}" \
  --surface_obs_root "${FUSION_SURFACE_NPY_ROOT}" \
  --surface_cg_root  "${RECON_INF_OUT}/surface" \
  --surface_out_root "${FUSION_SURFACE_OUTPUT}" \
  --deep_obs_root    "${FUSION_DEEP_NPY_ROOT}" \
  --deep_cg_root     "${RECON_INF_OUT}/deep" \
  --deep_out_root    "${FUSION_DEEP_OUTPUT}" \
  --bg_root   "$(dirname "$SCRIPT_DIR")/assimilation_workspace/background" \
  --obs_root  "$(dirname "$SCRIPT_DIR")/assimilation_workspace/observation" \
  --mode cpu --gpu_devices 0 \
  --save_root "$(dirname "$SCRIPT_DIR")/assimilation_workspace/background"


#######Remove old NPY files; keep only the three most recent dates in the assimilation background directory######
echo "Remove old shallow NPY files: $FINALL_ASSI_NPY_OUTPUT_DIR_surface"
ls -1 "$FINALL_ASSI_NPY_OUTPUT_DIR_surface"/*.npy 2>/dev/null | sort | head -n -3 | xargs -r rm -f
echo "Remove old deep NPY files: $FINALL_ASSI_NPY_OUTPUT_DIR_deep"
ls -1 "$FINALL_ASSI_NPY_OUTPUT_DIR_deep"/*.npy 2>/dev/null | sort | head -n -3 | xargs -r rm -f

# End of full workflow
# ==================================================
echo -e "\n=================================================="
echo "Full workflow completed!"
echo "All logs have been saved to: ${LOG_DIR}"
echo "=================================================="

####### Delete intermediate data########

echo -e "\n===Clean intermediate data: delete all contents under ${DATA_NASE_DIR} ==="
if [ -d "$DATA_NASE_DIR" ]; then
    rm -rf "${DATA_NASE_DIR}"/*
    echo "Intermediate data cleanup completed!"
else
    echo "Intermediate directory ${DATA_NASE_DIR} does not exist"
fi
###################################################  















