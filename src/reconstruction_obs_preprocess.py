import argparse
import glob
import logging
import os
import warnings
from datetime import datetime, timedelta
from multiprocessing import Pool

import netCDF4 as nc
import numpy as np
from cdo import Cdo

warnings.simplefilter("ignore", category=RuntimeWarning)

cdo = Cdo()

GRID_1_1_4 = 'grid_1_1_4.txt'
GRID_1_12  = 'grid_1_12.txt'


# ============================================================
# Utility functions
# ============================================================
def ensure_grid_file(path):
    """Generate the CDO 1/4° lonlat grid-description file if it does not exist."""
    if os.path.exists(path):
        return
    with open(path, "w") as f:
        f.write("gridtype = lonlat\n")
        f.write("xsize    = 1440\n")
        f.write("ysize    = 680\n")
        f.write("xfirst   = -180\n")
        f.write("xinc     = 0.25\n")
        f.write("yfirst   = -80\n")
        f.write("yinc     = 0.25\n")


def date_range(start, end):
    now = start
    while now <= end:
        yield now
        now += timedelta(days=1)


# ============================================================
# CDO resampling
# ============================================================
def _remap(date_str, input_dir, output_dir, grid_file, pattern):
    """Generic remapbil logic: recursive matching -> use the first file -> output YYYYMMDD.nc"""
    files = glob.glob(pattern, recursive=True)
    if not files:
        return f"SKIP {date_str}"
    infile = files[0]
    outfile = os.path.join(output_dir, f"{date_str}.nc")
    os.makedirs(output_dir, exist_ok=True)
    cdo.remapbil(grid_file, input=infile, output=outfile)
    return f"OK {date_str}"


def remap_aviso(date_str, input_dir, output_dir):
    """01_cdo_aviso：AVISO，1/4° remapbil。"""
    return _remap(date_str, input_dir, output_dir, GRID_1_1_4,
                  f"{input_dir}/**/*{date_str}*.nc")


def remap_sst(date_str, input_dir, output_dir):
    """02_cdo_sst：GHRSST，1/4° remapbil。"""
    return _remap(date_str, input_dir, output_dir, GRID_1_1_4,
                  f"{input_dir}/**/*{date_str}*.nc")


def remap_smap(date_str, input_dir, output_dir):
    """03_cdo_smap: SMAP, matched by YYYY_DDD and remapped to 1/4° with remapbil."""
    dt = datetime.strptime(date_str, "%Y%m%d")
    pattern = f"{input_dir}/**/*{dt.strftime('%Y')}_{dt.strftime('%j')}*.nc"
    return _remap(date_str, input_dir, output_dir, GRID_1_12, pattern)


def remap_cmems(date_str, input_dir, output_dir, processes=4):
    """04_cdo_cmems: CMEMS cropped with sellonlatbox (non-recursive)."""
    files = glob.glob(os.path.join(input_dir, f"*{date_str}*.nc"))
    if not files:
        return f"SKIP {date_str}"
    infile = files[0]
    outfile = os.path.join(output_dir, f"{date_str}.nc")
    os.makedirs(output_dir, exist_ok=True)
    cdo.sellonlatbox("-180,180,-80,90", input=infile, output=outfile,
                     options=f"-P {processes}")
    return f"OK {date_str}"


# ---------- Parallel worker ----------
def _seq_worker(name, func, start, end, input_dir, output_dir):
    """Process the complete date range sequentially for one data source (called in parallel by the outer Pool)."""
    for d in date_range(start, end):
        print(f"{name} {func(d.strftime('%Y%m%d'), input_dir, output_dir)}")


def _cmems_worker(start, end, input_dir, output_dir, processes):
    for d in date_range(start, end):
        print(f"CMEMS {remap_cmems(d.strftime('%Y%m%d'), input_dir, output_dir, processes)}")


# ============================================================
# Fusion (corresponding to 06_NcToNpySimplewithsss_temp.py)
# ============================================================
def _read_nc(path):
    if not path or not os.path.exists(path):
        logging.info(f"File not found: {path}")
        return None
    try:
        return nc.Dataset(path)
    except Exception as e:
        logging.error(f"Error reading file {path}: {e}")
        return None


def _read_var(var, dtype=np.float16):
    """Read an NC variable or ndarray slice -> convert dtype -> fill masked values."""
    data = var[:]
    data = data.astype(dtype)
    if isinstance(data, np.ma.MaskedArray):
        data.set_fill_value(-32767)
        data = data.filled()
    return data


def fuse_one(current_date, folders, output_dir):
    """Daily fusion: AVISO + SST + CMEMS (two u/v levels) + SSS (SMAP or climatology)."""
    date_str = current_date.strftime("%Y%m%d")

    ghrsst_path = os.path.join(folders['ghrsst'], f"{date_str}.nc")
    cmems_path  = os.path.join(folders['cmems'],  f"{date_str}.nc")
    aviso_path  = os.path.join(folders['aviso'],  f"{date_str}.nc")

    sss_smap_path = os.path.join(folders['sss'], f"{date_str}.nc")
    if os.path.exists(sss_smap_path):
        sss_path = sss_smap_path
        use_smap = True

    logging.info(f"Date: {current_date.strftime('%Y-%m-%d')}")
    logging.info(f"GHRSST PATH: {ghrsst_path}")
    logging.info(f"CMEMS  PATH: {cmems_path}")
    logging.info(f"AVISO  PATH: {aviso_path}")
    logging.info(f"SSS    PATH: {sss_path}")

    ghrsst_ds = _read_nc(ghrsst_path)
    cmems_ds  = _read_nc(cmems_path)
    aviso_ds  = _read_nc(aviso_path)
    sss_ds    = _read_nc(sss_path)

    if any(ds is None for ds in (ghrsst_ds, cmems_ds, aviso_ds, sss_ds)):
        logging.info("file has none")
        return

    # SST
    sst_data = _read_var(ghrsst_ds.variables['analysed_sst'])
    ghrsst_ds.close()

    # SLA
    allsat_data = _read_var(aviso_ds.variables['sla'])
    aviso_ds.close()

    # Current velocity (first two levels)
    u0 = _read_var(cmems_ds.variables['uo'])
    v0 = _read_var(cmems_ds.variables['vo'])
    uv1 = np.stack((u0[0, 0], v0[0, 0]), axis=0)
    uv2 = np.stack((u0[0, 1], v0[0, 1]), axis=0)
    cmems_ds.close()

    # SSS
    if use_smap:
        if 'sss_smap' in sss_ds.variables:
            sss_data = _read_var(sss_ds.variables['sss_smap'])
        elif 'sss' in sss_ds.variables:
            logging.warning(f"SMAP missing 'sss_smap', using 'sss': {sss_path}")
            sss_data = _read_var(sss_ds.variables['sss'])
        else:
            logging.warning(f"SMAP variable not found, using first: {sss_path}")
            sss_data = _read_var(sss_ds.variables[list(sss_ds.variables.keys())[0]])
    #else:
        #if 'so' in sss_ds.variables:
            #sss_data = _read_var(sss_ds.variables['so'][:, 0, :, :])
        #else:
            #logging.warning(f"Climatic missing 'so', using first: {sss_path}")
            #sss_data = _read_var(sss_ds.variables[list(sss_ds.variables.keys())[0]])
    sss_ds.close()

    fusion_data = np.full(shape=(7, 680, 1440), fill_value=np.nan, dtype=np.float16)
    fusion_data[0]     = allsat_data
    fusion_data[1]     = sst_data
    fusion_data[2:4]   = uv1
    fusion_data[4:6]   = uv2
    fusion_data[6]     = sss_data

    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, f"{date_str}.npy"), fusion_data)


# ============================================================
# Main workflow
# ============================================================
def main():
    p = argparse.ArgumentParser(description="Observation CDO resampling + NC-to-NPY fusion (combined 01-04 & 06)")
    p.add_argument("--start-date", required=True, help="Start date (YYYY-MM-DD)")
    p.add_argument("--end-date",   required=True, help="End date (YYYY-MM-DD)")

    # raw input directory
    p.add_argument("--aviso-dir",    required=True, help="AVISO raw input directory")
    p.add_argument("--ghrsst-dir",   required=True, help="GHRSST raw input directory")
    p.add_argument("--smap-dir",     required=True, help="SMAP raw input directory")
    p.add_argument("--cmems-dir",    required=True, help="CMEMS raw input directory")

    # resampled output directory（equivalent to RECON_*_OUT in the original shell script）
    p.add_argument("--aviso-out", required=True, help="AVISO resampled output directory")
    p.add_argument("--sst-out",   required=True, help="GHRSST resampled output directory")
    p.add_argument("--smap-out",  required=True, help="SMAP resampled output directory")
    p.add_argument("--cmems-out", required=True, help="CMEMS resampled output directory")

    # fusion output directory
    p.add_argument("--output-dir", required=True, help="Fusion NPY output directory")
    p.add_argument("--processes", type=int, default=4, help="Number of parallel CDO processes (default: 4)")
    p.add_argument("--skip-preprocess", action="store_true",
                   help="Skip CDO resampling and perform fusion only (for already preprocessed data)")
    args = p.parse_args()

    start = datetime.strptime(args.start_date, "%Y-%m-%d")
    end   = datetime.strptime(args.end_date,   "%Y-%m-%d")

    os.makedirs(args.output_dir, exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(
            args.output_dir,
            f'data_fusion_{args.start_date}_{args.end_date}_with_sss.log'),
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
    )

    # ---------- 1) CDO resampling (01-04 in parallel) ----------
    if not args.skip_preprocess:
        ensure_grid_file(GRID_1_1_4)
        ensure_grid_file(GRID_1_12)
        print("==== 1) CDO resampling: AVISO / SST / SMAP / CMEMS in parallel ====")
        with Pool(processes=4) as pool:
            pool.apply_async(_seq_worker,
                             ("AVISO", remap_aviso, start, end,
                              args.aviso_dir, args.aviso_out))
            pool.apply_async(_seq_worker,
                             ("SST", remap_sst, start, end,
                              args.ghrsst_dir, args.sst_out))
            pool.apply_async(_seq_worker,
                             ("SMAP", remap_smap, start, end,
                              args.smap_dir, args.smap_out))
            pool.apply_async(_cmems_worker,
                             (start, end, args.cmems_dir,
                              args.cmems_out, args.processes))
            pool.close()
            pool.join()
        print("1) CDO resampling completed")

    # ---------- 2) NC→NPY fusion（06） ----------
    print("==== 2) NC -> NPY fusion ====")
    folders = {
        'aviso':    args.aviso_out,
        'ghrsst':   args.sst_out,
        'cmems':    args.cmems_out,
        'sss':      args.smap_out,
    }
    for d in date_range(start, end):
        fuse_one(d, folders, args.output_dir)
    print("2) Fusion completed")


if __name__ == "__main__":
    main()
