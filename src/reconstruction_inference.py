from datetime import datetime
import onnxruntime as ort
import numpy as np
import time
import os
import argparse
import pathlib
from reconstruction_obs_preprocess_step import process_data
import traceback
import threading
import torch
import gc

# Base path configuration
file_path = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.dirname(file_path)
# Default root paths: background-field root and observation-field root
DEFAULT_INPUT = os.path.join(project_path,  'reconstruction_workspace', 'input')
DEFAULT_OUTPUT = os.path.join(project_path,  'reconstruction_workspace', 'output')

# Core inference class (dual-input: background field + observation field)
class inference_gpu():
    def __init__(self, ort_session, layer):
        super().__init__()
        self.ort_session = ort_session
        #print(f"[---init---]: layer={layer}")
        self.data_process = process_data(
            os.path.join(file_path, 'reconstruction_config.yaml'),
            f'variables_{layer}',
            f'input_{layer}',
            f'output_{layer}'
        )
        self.denorm = self.data_process.get_denormalize()

    def inference(self, input_data):
        # Data normalization
        x = self.data_process.read_data(input_data)
        ort_inputs = {'input': x}
        
        # Inference
        start = time.time()
        pred = self.ort_session.run(None, ort_inputs)[0]
        pred = self.denorm(torch.from_numpy(pred))
        return pred

# Inference function (dual data loading: background + observation)
def infer_on_gpu(model_cfg, session):
    # Read files: filter by date + sort
    model_name =model_cfg["layer"]
    def load_files(data_path, is_background=False):
        files = []
        for r, d, f in os.walk(data_path):
            for name in f:
                date = name[-12:-4]
                if is_background:
                    files.append((os.path.join(r, name), date))
                else:
                    if args.start_date <= date <= args.end_date:
                        files.append((os.path.join(r, name), date))
        return sorted(files, key=lambda x: x[1])

    # Read background-field and observation-field files for the current model
    input_files = load_files(model_cfg["input_path"],is_background=False)
    mask = np.load(model_cfg["mask_path"])
    mask = torch.tensor(mask)
    inferencer = inference_gpu(session, model_cfg["layer"])

    for input_path, target_date in input_files:
        try:
            input = np.load(input_path, mmap_mode='r')

            pred = inferencer.inference(input)
            pred[mask] = np.nan
            save_path = os.path.join(model_cfg["out_path"], f"pred_mra5_{target_date}.npy")
            np.save(save_path, pred.numpy().astype(np.float16))

        except Exception as e:
            print(f"Processing failed: {e}")
            traceback.print_exc()

if __name__ == "__main__":
    # Command-line arguments
    parser = argparse.ArgumentParser(description="Reconstruction ONNX inference")
    parser.add_argument("--input_root", type=str, default=DEFAULT_INPUT, help="Observation-field root directory")
    parser.add_argument("--start_date", type=str, default="20211101", help="Start date YYYYMMDD")
    parser.add_argument("--end_date", type=str, default="20211130", help="End date YYYYMMDD")
    parser.add_argument("--mode", type=str, default="cpu", choices=["cpu", "gpu"])
    parser.add_argument("--gpu_devices", nargs="+", type=int, default=[0])
    parser.add_argument("--save_root", type=str, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    # Core configuration: bind shallow/deep models to their background/observation paths
    MODEL_CONFIGS = [
        # Shallow model (1to22)
        {
            "input_path": args.input_root,
            "out_path": os.path.join(args.save_root, "surface"),
            "mask_path": os.path.join(file_path, "maske_surface_recon_48.npy"),
            "layer": "1to22",
            "onnx": "reconstruction_best_onnx_1to22.onnx"
        },
        
        # Deep model (23to33)
        {
            "input_path": args.input_root,
            "out_path": os.path.join(args.save_root, "deep"),
            "mask_path": os.path.join(file_path, "mask_deep.npy"),
            "layer": "23to33",
            "onnx": "reconstruction_best_onnx_23to33.onnx"
        }
        
    ]

    # Create output directory
    for cfg in MODEL_CONFIGS:
        pathlib.Path(cfg["out_path"]).mkdir(parents=True, exist_ok=True)

    # Load ONNX models
    sessions = []
    opts = ort.SessionOptions()
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    opts.intra_op_num_threads = 2
    opts.add_session_config_entry("session.use_device_allocator_for_initializers", "1")
    opts.add_session_config_entry("memory.arena.enable", "0")

    # CPU mode
    if args.mode == "cpu":
        for cfg in MODEL_CONFIGS:
            onnx_path = os.path.join(project_path,  "reconstruction_models", cfg["onnx"])
            sess = ort.InferenceSession(onnx_path, sess_options=opts, providers=["CPUExecutionProvider"])
            for inp in sess.get_inputs():
                sessions.append(sess)
    # GPU mode
    else:
        if len(args.gpu_devices) >= 2:
            for i, g in enumerate(args.gpu_devices[:2]):
                provider = [("CUDAExecutionProvider", {"device_id": g})]
                onnx_path = os.path.join(project_path, "reconstruction_models", MODEL_CONFIGS[i]["onnx"])
                sessions.append(ort.InferenceSession(onnx_path, sess_options=opts, providers=provider))
        else:
            provider = [("CUDAExecutionProvider", {"device_id": args.gpu_devices[0]})]
            for cfg in MODEL_CONFIGS:
                onnx_path = os.path.join(project_path, "reconstruction_models", cfg["onnx"])
                sessions.append(ort.InferenceSession(onnx_path, sess_options=opts, providers=provider))
    
    for i, s in enumerate(sessions):
        infer_on_gpu(MODEL_CONFIGS[i], s)
