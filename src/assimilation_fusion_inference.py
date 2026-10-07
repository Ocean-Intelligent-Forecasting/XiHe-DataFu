import argparse
import datetime
import logging
import os
import pathlib
import sys
import time
import traceback
from datetime import datetime as dt
from datetime import timedelta
from multiprocessing.pool import Pool

import numpy as np
import psutil

# Ensure modules in the same directory can be imported and the assimilation configuration can be found
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# =====================================================================
# Shallow indices (from 13_process_surface_recon_data_with_obsdata_with_pool.py)
# =====================================================================
SURFACE_CG_THETAO_INDEX = [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44]
SURFACE_CG_SO_INDEX     = [1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45]
SURFACE_CG_U0_INDEX     = [6, 10, 14, 18, 26, 30, 34, 38, 42, 46]
SURFACE_CG_V0_INDEX     = [7, 11, 15, 19, 27, 31, 35, 39, 43, 47]

SURFACE_OBS_THETAO_INDEX = [4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48]
SURFACE_OBS_SO_INDEX     = [5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 49]
SURFACE_OBS_UO_INDEX     = [10, 14, 18, 22, 30, 34, 38, 42, 46, 50]
SURFACE_OBS_VO_INDEX     = [11, 15, 19, 23, 31, 35, 39, 43, 47, 51]

SURFACE_IDX = dict(
    cg_thetao=SURFACE_CG_THETAO_INDEX, cg_so=SURFACE_CG_SO_INDEX,
    cg_u0=SURFACE_CG_U0_INDEX,         cg_v0=SURFACE_CG_V0_INDEX,
    obs_thetao=SURFACE_OBS_THETAO_INDEX, obs_so=SURFACE_OBS_SO_INDEX,
    obs_uo=SURFACE_OBS_UO_INDEX,         obs_vo=SURFACE_OBS_VO_INDEX,
)


# =====================================================================
# Deep indices (from 14_process_deep_recon_data_with_obsdata_with_pool.py)
# =====================================================================
DEEP_CG_THETAO_INDEX = [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40]
DEEP_CG_SO_INDEX     = [1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41]
DEEP_CG_U0_INDEX     = [2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42]
DEEP_CG_V0_INDEX     = [3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43]

DEEP_OBS_THETAO_INDEX = [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40]
DEEP_OBS_SO_INDEX     = [1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41]
DEEP_OBS_UO_INDEX     = [2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42]
DEEP_OBS_VO_INDEX     = [3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43]

DEEP_IDX = dict(
    cg_thetao=DEEP_CG_THETAO_INDEX, cg_so=DEEP_CG_SO_INDEX,
    cg_u0=DEEP_CG_U0_INDEX,         cg_v0=DEEP_CG_V0_INDEX,
    obs_thetao=DEEP_OBS_THETAO_INDEX, obs_so=DEEP_OBS_SO_INDEX,
    obs_uo=DEEP_OBS_UO_INDEX,         obs_vo=DEEP_OBS_VO_INDEX,
)


# =====================================================================
# General: date-list utility
# =====================================================================
def _date_list(start_date, end_date):
    """Expand a YYYY-MM-DD / YYYYMMDD range into [YYYYMMDD, ...]."""
    s = dt.strptime(start_date.replace('-', ''), '%Y%m%d')
    e = dt.strptime(end_date.replace('-', ''),   '%Y%m%d')
    out = []
    cur = s
    while cur <= e:
        out.append(cur.strftime('%Y%m%d'))
        cur += timedelta(days=1)
    return out


# =====================================================================
# General: single-day fusion task (shared by shallow/deep)
# =====================================================================
def _fusion_task(date_str, obs_root, cg_root, out_root, idx, do_nan_to_num):
    try:
        task_start = time.time()
        logging.info("Current directory: 【{0}】,processing time: 【{1}】".format(
            date_str, datetime.datetime.now()))

        npy_path    = os.path.join(obs_root, "mra5_" + date_str + '.npy')
        cg_npy_path = os.path.join(cg_root, "pred_mra5_" + date_str + '.npy')

        if not os.path.exists(npy_path):
            logging.warning(f"Corresponding NPY file {os.path.basename(npy_path)} does not exist; skipping. File path: {npy_path}")
            return
        if not os.path.exists(cg_npy_path):
            logging.warning(f"Corresponding NPY file {os.path.basename(cg_npy_path)} does not exist; skipping. File path: {cg_npy_path}")
            return

        data_x  = np.load(npy_path)
        data_cg = np.load(cg_npy_path)

        try:
            # Identify invalid observation locations (< -30000)
            thetao_obs = data_x[:, idx['obs_thetao'], :, :]
            obs_mask_thetao = thetao_obs < -30000
            so_obs = data_x[:, idx['obs_so'], :, :]
            obs_mask_so = so_obs < -30000
            uo_obs = data_x[:, idx['obs_uo'], :, :]
            obs_mask_uo = uo_obs < -30000
            vo_obs = data_x[:, idx['obs_vo'], :, :]
            obs_mask_vo = vo_obs < -30000

            # Extract reconstruction data (float16)
            thetao_cg = data_cg[:, idx['cg_thetao'], :, :].astype(np.float16)
            so_cg     = data_cg[:, idx['cg_so'],     :, :].astype(np.float16)
            uo_cg     = data_cg[:, idx['cg_u0'],     :, :].astype(np.float16)
            vo_cg     = data_cg[:, idx['cg_v0'],     :, :].astype(np.float16)

            # Fill invalid observation locations with reconstruction data
            thetao_with_obs = data_x[:, idx['obs_thetao'], :, :]
            so_with_obs     = data_x[:, idx['obs_so'],     :, :]
            uo_with_obs     = data_x[:, idx['obs_uo'],     :, :]
            vo_with_obs     = data_x[:, idx['obs_vo'],     :, :]

            thetao_with_obs[obs_mask_thetao] = thetao_cg[obs_mask_thetao]
            so_with_obs[obs_mask_so]         = so_cg[obs_mask_so]
            uo_with_obs[obs_mask_uo]         = uo_cg[obs_mask_uo]
            vo_with_obs[obs_mask_vo]         = vo_cg[obs_mask_vo]

            data_x[:, idx['obs_thetao'], :, :] = thetao_with_obs
            data_x[:, idx['obs_so'],     :, :] = so_with_obs
            data_x[:, idx['obs_uo'],     :, :] = uo_with_obs
            data_x[:, idx['obs_vo'],     :, :] = vo_with_obs
        except Exception as e:
            logging.error(f"process {npy_path} while reading the NC file: {str(e)}")
        finally:
            try:
                if 'Climatology_data' in locals():
                    Climatology_data.close()
            except Exception:
                pass

        file_name = "mra5_" + date_str + '.npy'

        # The original shallow workflow applies nan_to_num before saving; the deep workflow does not
        if do_nan_to_num:
            data_x = np.nan_to_num(data_x, nan=-32768)

        np.save(os.path.join(out_root, file_name), data_x)

        logging.info("process【{0}】directory preprocessing time【{1}] seconds".format(
            date_str, time.time() - task_start))
    except Exception as e:
        logging.info("Directory with exception: 【{0}】,processing time: 【{1}】".format(
            date_str, datetime.datetime.now()))
        logging.info(e)
        logging.info(traceback.format_exc())


def _compute_pool_num():
    """Dynamically set the number of processes based on available memory and CPU count (same logic for 13/14)."""
    available_value = int(psutil.virtual_memory().available / (1024 ** 3))
    logging.info("Available memory [{0}】G".format(available_value))
    course_value = 12
    logging.info("Memory per preprocessing process [{0}】G".format(course_value))
    pool_max = int(available_value * 0.8 / course_value)
    pool_cpu = psutil.cpu_count()
    logging.info("Available CPU logical cores [{0}] cores".format(pool_cpu))
    if pool_max > pool_cpu:
        pool_num = int(pool_cpu / 2)
    else:
        pool_num = int(pool_max / 2)
    return max(1, pool_num)


def _run_fusion_phase(start_date, end_date, obs_root, cg_root, out_root,
                      idx, do_nan_to_num, log_name):
    """Run one fusion stage (shallow or deep)."""
    # Use a separate log file for each stage
    logging.root.handlers.clear()
    logging.basicConfig(filename=log_name, filemode='w',
                        format='%(name)s - %(levelname)s - %(message)s')
    logging.root.setLevel(logging.DEBUG)

    pathlib.Path(out_root).mkdir(exist_ok=True, parents=True)

    start = time.time()
    pool_num = _compute_pool_num()
    logging.info("Dynamically set process count to {0}".format(pool_num))

    dates = _date_list(start_date, end_date)
    logging.info("Number of dates: ")
    logging.info(len(dates))

    logging.info("Process-pool tasks started...")
    with Pool(pool_num) as pool:
        pool.starmap(_fusion_task,
                     [(d, obs_root, cg_root, out_root, idx, do_nan_to_num)
                      for d in dates])
    logging.info("Waiting for process-pool tasks to complete...")
    logging.info("Process-pool tasks finished...")

    logging.info("【[{0}]-[{1}]】 total preprocessing time [{2}] seconds".format(
        start_date, end_date, time.time() - start))
    logging.info("END")


# =====================================================================
# =====================================================================
def _infer_on_session(model_cfg, ort_session, start_date, end_date,
                      process_data_cls, torch_mod):
    """Run inference for one data source (shallow or deep) using a single model session."""
    import numpy as np

    class _InferenceGPU:
        def __init__(self, session, layer):
            self.ort_session = session
            self.data_process = process_data_cls(
                os.path.join(os.path.dirname(os.path.abspath(__file__)), 'assimilation_config.yaml'),
                f'variables_{layer}', f'input_{layer}', f'output_{layer}')
            self.denorm = self.data_process.get_denormalize()

        def inference(self, bg1_path, bg2_path, bg3_path, obs_path):
            bg1 = self.data_process.read_data(bg1_path)
            bg2 = self.data_process.read_data(bg2_path)
            bg3 = self.data_process.read_data(bg3_path)
            obs = self.data_process.read_data_obs(obs_path)
            ort_inputs = {"input1": bg1, "input2": bg2, "input3": bg3, "input4": obs}
            pred = self.ort_session.run(None, ort_inputs)[0]
            pred = self.denorm(torch_mod.from_numpy(pred))
            return pred

    def _load_files(data_path, is_background):
        files = []
        for r, d, f in os.walk(data_path):
            for name in f:
                date = name[-12:-4]
                if is_background:
                    files.append((os.path.join(r, name), date))
                else:
                    if start_date <= date <= end_date:
                        files.append((os.path.join(r, name), date))
        return sorted(files, key=lambda x: x[1])

    bg_files  = _load_files(model_cfg["bg_path"],  is_background=True)
    obs_files = _load_files(model_cfg["obs_path"], is_background=False)

    mask = np.load(model_cfg["mask_path"])
    mask = torch_mod.tensor(mask)

    # Use only the first three background-field files (consistent with the original script)
    bg1 = np.load(bg_files[0][0], mmap_mode='r')
    bg2 = np.load(bg_files[1][0], mmap_mode='r')
    bg3 = np.load(bg_files[2][0], mmap_mode='r')

    inferencer = _InferenceGPU(ort_session, model_cfg["layer"])
    for obs_path, target_date in obs_files:
        try:
            obs = np.load(obs_path, mmap_mode='r')
            pred = inferencer.inference(bg1, bg2, bg3, obs)
            pred[mask] = np.nan
            save_path = os.path.join(model_cfg["out_path"],
                                     f"pred_mra5_{target_date}.npy")
            np.save(save_path, pred.numpy().astype(np.float16))
        except Exception:
            traceback.print_exc()


def run_inference(bg_root, obs_root, start_date, end_date,
                  mode, gpu_devices, save_root):
    import onnxruntime as ort
    import torch
    from assimilation_obs_preprocess_step import process_data

    file_path    = os.path.dirname(os.path.abspath(__file__))
    project_path = os.path.dirname(file_path)

    model_configs = [
        {
            "bg_path":   os.path.join(bg_root, "surface"),
            "obs_path":  os.path.join(obs_root, "surface"),
            "out_path":  os.path.join(save_root, "surface"),
            "mask_path": os.path.join(file_path, "mask_surface_52.npy"),
            "layer":     "1to22",
            "onnx":      "assi_best_1to22.onnx",
        },
        {
            "bg_path":   os.path.join(bg_root, "deep"),
            "obs_path":  os.path.join(obs_root, "deep"),
            "out_path":  os.path.join(save_root, "deep"),
            "mask_path": os.path.join(file_path, "mask_deep.npy"),
            "layer":     "23to33",
            "onnx":      "assi_best_23to33.onnx",
        },
    ]

    for cfg in model_configs:
        pathlib.Path(cfg["out_path"]).mkdir(parents=True, exist_ok=True)

    opts = ort.SessionOptions()
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    opts.intra_op_num_threads = 2
    opts.add_session_config_entry("session.use_device_allocator_for_initializers", "1")
    opts.add_session_config_entry("memory.arena.enable", "0")

    sessions = []
    if mode == "cpu":
        for cfg in model_configs:
            onnx_path = os.path.join(project_path, "assimilation_models", cfg["onnx"])
            sessions.append(ort.InferenceSession(
                onnx_path, sess_options=opts, providers=["CPUExecutionProvider"]))
    else:
        if len(gpu_devices) >= 2:
            for i, g in enumerate(gpu_devices[:2]):
                provider = [("CUDAExecutionProvider", {"device_id": g})]
                onnx_path = os.path.join(project_path, "assimilation_models", model_configs[i]["onnx"])
                sessions.append(ort.InferenceSession(
                    onnx_path, sess_options=opts, providers=provider))
        else:
            provider = [("CUDAExecutionProvider", {"device_id": gpu_devices[0]})]
            for cfg in model_configs:
                onnx_path = os.path.join(project_path, "assimilation_models", cfg["onnx"])
                sessions.append(ort.InferenceSession(
                    onnx_path, sess_options=opts, providers=provider))

    for i, s in enumerate(sessions):
        _infer_on_session(model_configs[i], s, start_date, end_date, process_data, torch)


# =====================================================================
# Stage 6: NPY -> NC (call npy_to_nc/npy2nc.sh)
# =====================================================================
def run_npy2nc(save_root, npy2nc_out, npy2nc_script, start_date, end_date):
    """Convert inference NPY outputs to NC and call npy2nc.sh once per date."""
    if not os.path.exists(npy2nc_script):
        print(f"WARNING: npy2nc.sh does not exist; skipping Stage 6: {npy2nc_script}")
        return
    os.makedirs(npy2nc_out, exist_ok=True)

    for d in _date_list(start_date, end_date):
        print("========== NPY to NC, file date [" + d + "] ========")
        print("====== Merge NPY data into NC files ======")
        cmd = f'bash "{npy2nc_script}" "{save_root}" "{npy2nc_out}" "{d}"'
        print(f"cmd: {cmd}")
        rc = os.system(cmd)
        if rc != 0:
            print(f"WARNING: npy2nc.sh returned code {rc}: {cmd}")


# =====================================================================
# Main entry: fusion (shallow + deep) + inference + NPY-to-NC
# =====================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Stages 4+6: fuse reconstruction results into assimilation observations + ONNX inference + NPY-to-NC (combined 13 / 14 / inference / npy2nc)")

    parser.add_argument("--start_date", required=True, help="YYYYMMDD")
    parser.add_argument("--end_date",   required=True, help="YYYYMMDD")

    # Shallow paths (corresponding to shell 4.1)
    parser.add_argument("--surface_obs_root", required=True,
                        help="Shallow observation (datalight) directory")
    parser.add_argument("--surface_cg_root", required=True,
                        help="Shallow reconstruction inference output directory (RECON_INF_OUT/surface)")
    parser.add_argument("--surface_out_root", required=True,
                        help="Shallow fusion output directory (observation/surface)")

    # Deep paths (corresponding to shell 4.2)
    parser.add_argument("--deep_obs_root", required=True,
                        help="Deep observation (datadeep) directory")
    parser.add_argument("--deep_cg_root", required=True,
                        help="Deep reconstruction inference output directory (RECON_INF_OUT/deep)")
    parser.add_argument("--deep_out_root", required=True,
                        help="Deep fusion output directory (observation/deep)")

    parser.add_argument("--log_dir", default=None, help="Fusion log directory (optional)")

    # Inference arguments
    parser.add_argument("--bg_root", required=True,
                        help="Background-field root directory (assimilation_workspace/background)")
    parser.add_argument("--obs_root", required=True,
                        help="Observation-field root directory used for inference (assimilation_workspace/observation)")
    parser.add_argument("--mode", choices=["cpu", "gpu"], default="cpu")
    parser.add_argument("--gpu_devices", nargs="+", type=int, default=[0])
    parser.add_argument("--save_root", required=True,
                        help="Inference-result root directory (also used as the Stage 6 NPY input directory)")

    # Stage 6 arguments
    parser.add_argument("--npy2nc_script", default=None,
                        help="Path to npy2nc.sh (default: npy_to_nc/npy2nc.sh in the script directory)")
    parser.add_argument("--npy2nc_out", default=None,
                        help="NPY-to-NC output directory (default: <project>/assimilation_workspace/output)")

    args = parser.parse_args()

    log_dir = args.log_dir or os.path.dirname(args.surface_out_root)
    os.makedirs(log_dir, exist_ok=True)

    # ---------- Stage 4.1: shallow fusion ----------
    print("==== 4.1 Fuse reconstruction inference results with assimilation observations (shallow) ====")
    _run_fusion_phase(
        start_date=args.start_date, end_date=args.end_date,
        obs_root=args.surface_obs_root, cg_root=args.surface_cg_root,
        out_root=args.surface_out_root, idx=SURFACE_IDX,
        do_nan_to_num=True,
        log_name=os.path.join(log_dir,
                              'fusion_assi_obstata_with_recon_surface_inf.log'),
    )
    print("4.1 Shallow data fusion completed")

    # ---------- Stage 4.2: deep fusion ----------
    print("==== 4.2 Fuse reconstruction inference results with assimilation observations (deep) ====")
    _run_fusion_phase(
        start_date=args.start_date, end_date=args.end_date,
        obs_root=args.deep_obs_root, cg_root=args.deep_cg_root,
        out_root=args.deep_out_root, idx=DEEP_IDX,
        do_nan_to_num=False,
        log_name=os.path.join(log_dir,
                              'deep_obsdata_recon_infdata_process.log'),
    )
    print("4.2 Deep data fusion completed")

    # ---------- Assimilation inference ----------
    print("==== Assimilation inference ====")
    run_inference(
        bg_root=args.bg_root, obs_root=args.obs_root,
        start_date=args.start_date, end_date=args.end_date,
        mode=args.mode, gpu_devices=args.gpu_devices,
        save_root=args.save_root,
    )
    print("Assimilation inference completed")

    # ---------- Stage 6: NPY -> NC ----------
    print("==== Stage 6: NPY to NC ====")
    script_dir   = os.path.dirname(os.path.abspath(__file__))
    project_path = os.path.dirname(script_dir)

    npy2nc_script = args.npy2nc_script or os.path.join(
        script_dir, 'npy_to_nc', 'npy2nc.sh')
    npy2nc_out = args.npy2nc_out or os.path.join(
        project_path, 'assimilation_workspace', 'output')

    run_npy2nc(
        save_root=args.save_root,
        npy2nc_out=npy2nc_out,
        npy2nc_script=npy2nc_script,
        start_date=args.start_date, end_date=args.end_date,
    )
    print("Stage 6 NPY-to-NC completed")


if __name__ == '__main__':
    main()
