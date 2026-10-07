import argparse
import datetime
import logging
import os
import pathlib
import re
import time as Time
import traceback
from datetime import timedelta
from multiprocessing import Pool

import numpy as np
import xarray as xr
from cdo import Cdo

cdo = Cdo()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# ============================================================
# General: filter files by date (corresponding to 07/08/09 date matching)
# ============================================================
def _collect_files(input_path, start_dt, end_dt, recursive):
    """When recursive=True, traverse recursively (08/09); when False, only scan the top level (07)."""
    entries = []
    if recursive:
        for root, _, files in os.walk(input_path):
            for f in files:
                entries.append(os.path.join(root, f))
    else:
        for entry in os.listdir(input_path):
            entries.append(os.path.join(input_path, entry))

    filenames = []
    for fp in entries:
        try:
            fname = os.path.basename(fp)
            matches = re.findall(r"\d{14}|\d{8}", fname) if recursive \
                else re.findall(r"\d{8}", fname)
            keep = False
            for m in matches:
                try:
                    file_date = datetime.datetime.strptime(m[:8], "%Y%m%d").date()
                    if start_dt <= file_date <= end_dt:
                        keep = True
                        break
                except Exception:
                    continue
            if keep and os.path.isfile(fp):
                filenames.append(fp)
        except Exception:
            logging.info(f"Exception while processing entry: {fp}")
            logging.info(traceback.format_exc())
    return filenames


# ============================================================
# Resampling worker (shared by 07/08/09)
# ============================================================
def _remap_one(tag, file_path, grid_path, input_path, output_root_path, keep_rel):
    """keep_rel=True preserves relative paths (08/09); False writes directly to the output directory (07)."""
    try:
        logging.info(f"[{tag}] Current file: 【{file_path}】,processing time: 【{datetime.datetime.now()}】")
        if keep_rel:
            rel_path = os.path.relpath(file_path, input_path)
            out_dir = os.path.join(output_root_path, os.path.dirname(rel_path))
        else:
            out_dir = output_root_path
        pathlib.Path(out_dir).mkdir(exist_ok=True, parents=True)

        out_path = os.path.join(out_dir, os.path.basename(file_path))
        cmd = "cdo -P 4 -remapbil,{0} {1} {2}".format(grid_path, file_path, out_path)
        logging.info(f"[{tag}] Running: {cmd}")
        rc = os.system(cmd)
        if rc != 0:
            logging.warning(f"[{tag}] cdo return code {rc}: {cmd}")
    except Exception as e:
        logging.info(f"[{tag}] File with exception: 【{file_path}】,exception time: 【{datetime.datetime.now()}】")
        logging.info(e)
        logging.info(traceback.format_exc())


# ============================================================
# Collect all resampling tasks into a flat list (avoid nested Pools)
# ============================================================
def _collect_remap_tasks(datasets, start_dt, end_dt):
    """datasets: [(tag, input_dir, output_dir, grid_path, keep_rel), ...]"""
    tasks = []
    for tag, input_dir, output_dir, grid_path, keep_rel in datasets:
        logging.info(f"==== [{tag}] Start collecting files ====")
        logging.info(f"[{tag}] input_dir={input_dir}, grid={grid_path}, keep_rel={keep_rel}")
        filenames = _collect_files(input_dir, start_dt, end_dt, recursive=keep_rel)
        logging.info(f"[{tag}] Collected {len(filenames)} files")
        for f in filenames:
            tasks.append((tag, f, grid_path, input_dir, output_dir, keep_rel))
    return tasks


# ============================================================
# Fusion (corresponding to 10_Process.py)
# ============================================================
def _to_fp16(arr):
    if isinstance(arr, np.ma.MaskedArray):
        arr = arr.filled(-32768)
    arr = arr.astype(np.float16)
    return np.nan_to_num(arr, nan=-32768)


def glob_files(root, pattern):
    import glob as _glob
    return _glob.glob(os.path.join(root, "**", pattern), recursive=True)


def fuse_one(current_date, sst_dir, cmems_dir, aviso_dir, fusion_output_dir):
    path_name = current_date.strftime("%Y%m%d")
    print(f"Processing {path_name}...", flush=True)

    sst_files = glob_files(sst_dir, f"{path_name}*.nc")
    if not sst_files:
        print(f"  Skipped {path_name}: SST file not found")
        return
    aviso_files = glob_files(aviso_dir, f"*{path_name}*.nc")
    if not aviso_files:
        print(f"  Skipped {path_name}: AVISO file not found")
        return
    cmems_files = glob_files(cmems_dir, f"*{path_name}*.nc")
    if not cmems_files:
        print(f"  Skipped {path_name}: CMEMS file not found")
        return

    with xr.open_dataset(sst_files[0]) as ds:
        data_sst = _to_fp16(ds.variables['analysed_sst'][:].values)
    with xr.open_dataset(aviso_files[0]) as ds:
        data_aviso = _to_fp16(ds.variables['sla'][:].values)
    with xr.open_dataset(cmems_files[0]) as ds:
        uo = _to_fp16(ds.variables['uo'][:].values)
        vo = _to_fp16(ds.variables['vo'][:].values)

    u1, v1 = uo[0, 0, :, :], vo[0, 0, :, :]
    u2, v2 = uo[0, 1, :, :], vo[0, 1, :, :]
    uv1 = np.stack((u1, v1), axis=0)
    uv2 = np.stack((u2, v2), axis=0)

    fusion_light = np.full((1, 52, 2041, 4320), -32768, dtype=np.float16)
    fusion_light[0, 0, :, :] = data_aviso
    fusion_light[0, 3, :, :] = data_sst
    fusion_light[0, 6:8, :, :] = uv1
    fusion_light[0, 26:28, :, :] = uv2

    fusion_deep = np.full((1, 44, 2041, 4320), -32768, dtype=np.float16)

    light_dir = os.path.join(fusion_output_dir, 'datalight')
    deep_dir = os.path.join(fusion_output_dir, 'datadeep')
    os.makedirs(light_dir, exist_ok=True)
    os.makedirs(deep_dir, exist_ok=True)

    light_path = os.path.join(light_dir, f'mra5_{path_name}.npy')
    deep_path = os.path.join(deep_dir, f'mra5_{path_name}.npy')
    np.save(light_path, fusion_light)
    np.save(deep_path, fusion_deep)
    print(f"  Saved {light_path} and {deep_path}")


# ============================================================
# Main workflow
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="Assimilation observation 1/12° resampling + fusion (combined 07-10)")
    parser.add_argument("--start-date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--end-date",   required=True, help="YYYY-MM-DD")
    parser.add_argument("--aviso-dir",  required=True)
    parser.add_argument("--ghrsst-dir", required=True)
    parser.add_argument("--cmems-dir",  required=True)
    parser.add_argument("--aviso-out",  required=True)
    parser.add_argument("--ghrsst-out", required=True)
    parser.add_argument("--cmems-out",  required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--processes", type=int, default=4)
    parser.add_argument("--log-name", type=str, default=None)
    parser.add_argument("--skip-preprocess", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    log_path = args.log_name or os.path.join(args.output_dir, "assi_preprocess_fusion.log")
    logging.basicConfig(filename=log_path, filemode='w',
                        format='%(name)s - %(levelname)s - %(message)s')
    logging.root.setLevel(logging.DEBUG)

    start_dt = datetime.datetime.strptime(args.start_date, "%Y-%m-%d").date()
    end_dt   = datetime.datetime.strptime(args.end_date,   "%Y-%m-%d").date()

    # ---------- 1) 1/12° resampling (07/08/09 flat parallelism, no nested Pool) ----------
    if not args.skip_preprocess:
        print("==== 2_1 Assimilation observation preprocessing (1/12°): AVISO/GHRSST/CMEMS in parallel ====")

        datasets = [
            ("AVISO",  args.aviso_dir,  args.aviso_out,
             os.path.join(BASE_DIR, 'aviso.txt'),  False),
            ("GHRSST", args.ghrsst_dir, args.ghrsst_out,
             os.path.join(BASE_DIR, 'GHRSST.txt'), True),
            ("CMEMS",  args.cmems_dir,  args.cmems_out,
             os.path.join(BASE_DIR, 'cmems.txt'),  True),
        ]

        tasks = _collect_remap_tasks(datasets, start_dt, end_dt)
        logging.info(f"Total resampling tasks: {len(tasks)}")
        print(f"Total resampling tasks: {len(tasks)}")

        if tasks:
            t0 = Time.time()
            with Pool(processes=args.processes) as pool:
                pool.starmap(_remap_one, tasks)
            elapsed = Time.time() - t0
            logging.info(f"Total resampling time [{elapsed}] seconds")
            print(f"Total resampling time [{elapsed}] seconds")
        else:
            logging.info("No files require resampling")
            print("WARNING: no files were collected; check the input directories and date range")

        print("2_1 Assimilation observation preprocessing completed")

    # ---------- 2) fusion（10_Process） ----------
    print("==== 2_2 Fuse assimilation observations (surface observations, no EN4) ====")
    current = start_dt
    while current <= end_dt:
        fuse_one(current, args.ghrsst_out, args.cmems_out, args.aviso_out,
                 args.output_dir)
        current += timedelta(days=1)
    print("2_2 Fusion completed")


if __name__ == '__main__':
    main()