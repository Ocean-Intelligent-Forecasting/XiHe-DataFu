#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Ocean data processing: variable extraction -> shallow/deep NC -> shallow/deep NPY.

Install dependencies: python -m pip install numpy netCDF4 tqdm
The runtime environment must also have the CDO command-line tool installed and be able to run cdo -V.
Example command:
    python process_ocean.py --input-dir ./raw_data --example-dir ./example_data --output-dir ./outputs --start-date 20210101 --end-date 20211231

Output directories (created automatically):
    get_variables/YYYYMMDD/                 the original four groups of variable NC files
    get_depth/surface/YYYYMMDD/             NC files for 12 shallow levels
    get_depth/deep/YYYYMMDD/                NC files for 11 deep levels
    npy_output/input_surface_data/mra5_YYYYMMDD.npy
    npy_output/input_deep_data/mra5_YYYYMMDD.npy

NPY data use float16, and channel indexing starts from 0:
    surface: (1, 52, 2041, 4320)
      0=zos，1=0，2=0，3=a copy of channel 4。
      Starting from channel 4, each depth is ordered as thetao, so, uo, vo.
    deep: (1, 44, 2041, 4320)
      Starting from channel 0, each depth is ordered as thetao, so, uo, vo.

ERA5/GHRSST are not read, and no SST unit conversion is performed.
Step 2 continues to use cdo -P 20 sellevel,<depth-list> to extract shallow and deep NC files separately.
Steps 1 and 3 read and write in row chunks; NPY uses memory mapping and writes only one large array at a time.
Masked missing values, NaN, Inf, and values greater than 9e36 are written as -32768.
This matches the actual stored value produced by converting -32767 to float16 in the legacy code.
An error is raised if valid values overflow when converted to float16; outputs containing Inf are not generated.

Convention: the first level of the input directory contains one NC file per day, using the original filename date rule.
The date range is inclusive; input and template files must both contain one time step and use the same grid, depth levels, and axis order.
3-D variables use dimensions (time, depth, latitude, longitude); zos uses (time, latitude, longitude).
NC level selection is performed by CDO; NPY data follow the configured depth order, with a 0.001 floating-point tolerance when validating coordinates.
Existing outputs with the same name are overwritten; final filenames are updated only after both NPY files are fully written.
"""

import argparse
from contextlib import contextmanager
from datetime import datetime
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

import netCDF4 as nc
import numpy as np
from tqdm import tqdm


VARIABLES = {"thetao": "3D-thetao", "so": "3D-so", "uovo": "3D-uovo", "zos": "2D"}
CHANNELS = {"thetao": 0, "so": 1, "uo": 2, "vo": 3}
LEVELS = {
    "surface": [
        0.4940, 2.6457, 5.0782, 7.9296, 11.4050, 15.8101,
        21.5988, 29.4447, 40.3441, 55.7643, 77.8539, 92.3261,
    ],
    "deep": [
        109.7293, 130.6660, 155.8507, 186.1256, 222.4752, 266.0403,
        318.1274, 380.2130, 453.9377, 541.0889, 643.5668,
    ],
}
GRID_SHAPE = (2041, 4320)
NPY_DTYPE = np.float16
MISSING_VALUE = np.float16(-32768)


@contextmanager
def staged_file(destination):
    """Write a temporary file first, replace the final file after success, and clean up the temporary file on errors."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=destination.name + ".", suffix=".tmp", dir=destination.parent)
    os.close(fd)
    temporary = Path(name)
    try:
        yield temporary
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def blocks(shape, chunk_rows):
    """Read only several rows from one time/depth slice at a time."""
    if len(shape) < 2:
        yield tuple(slice(None) for _ in shape)
        return
    for index in np.ndindex(shape[:-2]):
        prefix = tuple(slice(i, i + 1) for i in index)
        for row in range(0, shape[-2], chunk_rows):
            yield prefix + (slice(row, min(row + chunk_rows, shape[-2])), slice(None))


def depth_selection(dataset, variable_name, levels):
    """Validate CDO output and determine the NPY depth order; this function does not select NC levels."""
    variable = dataset.variables[variable_name]
    if variable.ndim != 4 or variable.shape[0] != 1 or variable.shape[-2:] != GRID_SHAPE:
        raise ValueError(f"{variable_name} should be (1, depth, {GRID_SHAPE[0]}, {GRID_SHAPE[1]}); actual: {variable.shape}")
    dimension = variable.dimensions[1]
    if dimension not in dataset.variables:
        raise ValueError(f"Depth coordinate variable not found: {dimension}")
    coordinate = dataset.variables[dimension]
    if coordinate.dimensions != (dimension,):
        raise ValueError(f"Depth coordinate {dimension} must be a one-dimensional coordinate")
    values = np.ma.asarray(coordinate[:], dtype=np.float64).filled(np.nan)
    indices = []
    for level in levels:
        found = np.flatnonzero(np.isclose(values, level, rtol=0, atol=0.001))
        if len(found) != 1:
            raise ValueError(f"{variable_name} depth {level:.4f} matched {len(found)} levels; exactly one match is required")
        indices.append(int(found[0]))
    return dimension, np.asarray(indices, dtype=int)


def write_nc(template_path, output_path, chunk_rows, replacements=None, when=None):
    """Step 1: copy the NC file from the template in chunks and replace data for the specified variables."""
    replacements = replacements or {}
    with nc.Dataset(template_path) as template, staged_file(output_path) as temporary:
        if template.groups:
            raise ValueError("This script requires variables to be in the NC root group; grouped NC files are not supported")
        for name, source in replacements.items():
            if name not in template.variables or source.shape != template.variables[name].shape:
                raise ValueError(f"Source variable {name} does not match template {template_path.name} in shape")
        sizes = {name: len(dim) for name, dim in template.dimensions.items()}
        with nc.Dataset(temporary, "w", format=template.data_model) as output:
            output.set_fill_off()  # All variables are fully written later, avoiding repeated pre-filling of large files.
            output.setncatts({name: template.getncattr(name) for name in template.ncattrs()})
            for name, dim in template.dimensions.items():
                output.createDimension(name, None if dim.isunlimited() else sizes[name])
            for name, reference in template.variables.items():
                shape = tuple(sizes[dim] for dim in reference.dimensions)
                attributes = {key: reference.getncattr(key) for key in reference.ncattrs()}
                options = {}
                if "_FillValue" in attributes:
                    options["fill_value"] = attributes.pop("_FillValue")
                if template.data_model.startswith("NETCDF4") and reference.ndim > 0:
                    filters = reference.filters() or {}
                    for key in ("zlib", "complevel", "shuffle", "fletcher32"):
                        if key in filters:
                            options[key] = filters[key]
                    if len(shape) >= 2 and all(shape):
                        options["chunksizes"] = (1,) * (len(shape) - 2) + (min(chunk_rows, shape[-2]), shape[-1])
                target = output.createVariable(name, reference.datatype, reference.dimensions, **options)
                target.setncatts(attributes)
                target.set_auto_chartostring(False)

                if name == "time" and when is not None:
                    if reference.shape != (1,) or "units" not in attributes:
                        raise ValueError("Template time must contain exactly one time point and a valid units attribute")
                    target[:] = nc.date2num([when], attributes["units"], attributes.get("calendar", "standard"))
                    continue

                source = replacements.get(name, reference)
                source.set_auto_chartostring(False)
                if name not in replacements:
                    # Copy template coordinates and related data unchanged to avoid rounding from repeated scale_factor decoding/encoding.
                    source.set_auto_maskandscale(False)
                    target.set_auto_maskandscale(False)
                for destination_slice in blocks(shape, chunk_rows):
                    target[destination_slice] = source[destination_slice]


def encode_npy(block):
    """Handle missing values only within the current chunk and convert to float16."""
    values = np.asarray(np.ma.getdata(block))
    missing = np.ma.getmaskarray(block) | ~np.isfinite(values) | (values > 9e36)
    with np.errstate(over="ignore", invalid="ignore"):
        encoded = values.astype(NPY_DTYPE)
    encoded[missing] = MISSING_VALUE
    if not np.isfinite(encoded).all():
        raise ValueError("Valid values that cannot be represented by float16 were found; check the data values or units")
    return encoded


def write_npy(output_path, layer, layer_files, zos_path, chunk_rows):
    """Write a group of NC files to NPY using the defined channel order, processing only one chunk at a time."""
    base = 4 if layer == "surface" else 0
    nlevels = len(LEVELS[layer])
    shape = (1, base + 4 * nlevels, *GRID_SHAPE)
    array = np.lib.format.open_memmap(output_path, mode="w+", dtype=NPY_DTYPE, shape=shape)
    try:
        if layer == "surface":
            with nc.Dataset(zos_path) as dataset:
                zos = dataset.variables["zos"]
                if zos.shape != (1, *GRID_SHAPE):
                    raise ValueError(f"zos should be (1, {GRID_SHAPE[0]}, {GRID_SHAPE[1]}); actual: {zos.shape}")
                for row in range(0, GRID_SHAPE[0], chunk_rows):
                    rows = slice(row, min(row + chunk_rows, GRID_SHAPE[0]))
                    array[0, 0, rows, :] = encode_npy(zos[0, rows, :])
                    array[0, 1:3, rows, :] = 0  # u10、v10 are set to zero everywhere, including land areas.

        for group in ("thetao", "so", "uovo"):
            with nc.Dataset(layer_files[group]) as dataset:
                names = ("uo", "vo") if group == "uovo" else (group,)
                for name in names:
                    variable = dataset.variables[name]
                    _, indices = depth_selection(dataset, name, LEVELS[layer])
                    if variable.shape[1] != nlevels:
                        raise ValueError(f"{layer}/{name} should be {nlevels} depth levels; actual: {variable.shape[1]}")
                    for k, index in enumerate(indices):
                        channel = base + 4 * k + CHANNELS[name]
                        for row in range(0, GRID_SHAPE[0], chunk_rows):
                            rows = slice(row, min(row + chunk_rows, GRID_SHAPE[0]))
                            tile = encode_npy(variable[0, int(index), rows, :])
                            array[0, channel, rows, :] = tile
                            if layer == "surface" and name == "thetao" and k == 0:
                                array[0, 3, rows, :] = tile  # Use the same data block as channel 4 so the values are identical.
        array.flush()
    finally:
        del array


def process_day(input_file, date_str, origin_date, example_dir, output_dir, chunk_rows, cdo="cdo"):
    """Complete the three steps sequentially for each day; use only files generated in the current run to avoid mixing other forecast batches."""
    variable_files = {}
    when = datetime.strptime(date_str, "%Y%m%d").replace(hour=12)
    with nc.Dataset(input_file) as source:
        for group, suffix in VARIABLES.items():
            names = ("uo", "vo") if group == "uovo" else (group,)
            output_name = f"glo12_rg_1d-m_{date_str}-{date_str}_{suffix}_fcst_R{origin_date}.nc"
            output_path = output_dir / "get_variables" / date_str / output_name
            replacements = {name: source.variables[name] for name in names}
            write_nc(example_dir / f"{group}.nc", output_path, chunk_rows, replacements=replacements, when=when)
            variable_files[group] = output_path

    layer_files = {layer: {} for layer in LEVELS}
    for group in ("thetao", "so", "uovo"):
        variable_path = variable_files[group]
        names = ("uo", "vo") if group == "uovo" else (group,)
        for layer, levels in LEVELS.items():
            output_path = output_dir / "get_depth" / layer / date_str / variable_path.name
            level_text = ",".join(f"{level:.4f}" for level in levels)
            with staged_file(output_path) as temporary:
                # Let CDO create the output to avoid overwrite prompts triggered by an empty temporary file.
                temporary.unlink()
                subprocess.run(
                    [cdo, "-P", "20", f"sellevel,{level_text}", str(variable_path), str(temporary)],
                    check=True,
                )
                # CDO may warn about missing levels; verify that all levels are present before saving.
                with nc.Dataset(temporary) as dataset:
                    for name in names:
                        depth_selection(dataset, name, levels)
                        if dataset.variables[name].shape[1] != len(levels):
                            raise ValueError(f"CDO output {layer}/{name} should contain {len(levels)}")
            layer_files[layer][group] = output_path

    npy_name = f"mra5_{date_str}.npy"
    surface_path = output_dir / "npy_output" / "input_surface_data" / npy_name
    deep_path = output_dir / "npy_output" / "input_deep_data" / npy_name
    # Update the final filenames only after both temporary files have been written.
    with staged_file(surface_path) as surface_tmp, staged_file(deep_path) as deep_tmp:
        write_npy(surface_tmp, "surface", layer_files["surface"], variable_files["zos"], chunk_rows)
        write_npy(deep_tmp, "deep", layer_files["deep"], variable_files["zos"], chunk_rows)


def parse_date(value):
    if len(value) != 8 or not value.isascii() or not value.isdigit():
        raise ValueError(f"Date must be an 8-digit YYYYMMDD value: {value!r}")
    return datetime.strptime(value, "%Y%m%d")


def main():
    parser = argparse.ArgumentParser(description="Ocean-variable extraction, CDO shallow/deep level extraction, and NPY conversion")
    parser.add_argument("--input-dir", type=Path, required=True, help="Directory containing raw NC files")
    parser.add_argument("--example-dir", type=Path, required=True, help="Directory containing template NC files")
    parser.add_argument("--output-dir", type=Path, required=True, help="Unified output root directory")
    parser.add_argument("--start-date", default="20210101", help="Start date YYYYMMDD, default: 20210101")
    parser.add_argument("--end-date", default="20211231", help="End date YYYYMMDD, default: 20211231")
    parser.add_argument("--chunk-rows", type=int, default=128, help="Number of latitude rows read/written per chunk in Step 1 and NPY conversion; default: 128")
    args = parser.parse_args()
    try:
        start_date, end_date = parse_date(args.start_date), parse_date(args.end_date)
    except ValueError as exc:
        parser.error(str(exc))
    if start_date > end_date or args.chunk_rows < 1:
        parser.error("The start date cannot be later than the end date, and chunk-rows must be greater than 0")
    input_dir = args.input_dir.expanduser().resolve()
    example_dir = args.example_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not input_dir.is_dir():
        parser.error(f"Input directory does not exist: {input_dir}")
    for group in VARIABLES:
        if not (example_dir / f"{group}.nc").is_file():
            parser.error(f"Missing template file: {example_dir / f'{group}.nc'}")
    cdo = shutil.which("cdo")
    if cdo is None:
        parser.error("The cdo command was not found; make sure the current environment can run cdo -V")

    jobs = {}
    failed = skipped = 0
    for path in sorted(input_dir.glob("*.nc")):
        if not path.is_file():
            continue
        try:
            date_str, origin_date = path.name[-21:-13], path.name[-11:-3]
            day = parse_date(date_str)
            if not start_date <= day <= end_date:
                skipped += 1
                continue
            parse_date(origin_date)
        except ValueError as exc:
            failed += 1
            print(f"Invalid date in filename: {path.name}; reason: {exc}")
            continue
        if date_str in jobs:
            parser.error(f"{date_str} has multiple raw files: {jobs[date_str][0].name}、{path.name}; only one input per day is allowed to avoid NPY filename collisions")
        jobs[date_str] = (path, origin_date)
    if not jobs:
        parser.error("No processable raw NC files were found in the specified date range")

    nc.set_chunk_cache(size=8 * 1024 * 1024, nelems=1009, preemption=0.75)
    success = 0
    for date_str, (path, origin_date) in tqdm(sorted(jobs.items()), desc="Variables -> levels -> NPY", unit="day"):
        try:
            process_day(path, date_str, origin_date, example_dir, output_dir, args.chunk_rows, cdo)
            success += 1
        except Exception as exc:
            failed += 1
            tqdm.write(f"Processing failed: {path.name}; reason: {exc}")
    print(f"Processing complete: succeeded for {success} days, failed for {failed} files, skipped {skipped} files outside the date range.")
    print(f"Shallow NPY: {output_dir / 'npy_output' / 'input_surface_data'}")
    print(f"Deep NPY: {output_dir / 'npy_output' / 'input_deep_data'}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
