import glob
import os.path

import numpy as np

os.makedirs("data/Schubert_Winterreise/pitch_hs512_overtones", exist_ok=True)

shifts = [12, 19, 24, 28, 31, 34, 36, 38, 40]
strengths = np.power(1.0/3.0, np.arange(1, len(shifts) + 1))

for npy_file_name in glob.glob("data/Schubert_Winterreise/pitch_hs512_nooverl/*.npy"):
    base_name = os.path.basename(npy_file_name)
    output_file_path = f"data/Schubert_Winterreise/pitch_hs512_overtones/{base_name}"

    wo_overtones = np.load(npy_file_name).astype(np.float64)
    w_overtones = np.copy(wo_overtones).astype(np.float64)

    for shift, strength in zip(shifts, strengths):
        w_overtones[shift:] += strength * wo_overtones[:-shift]

    w_overtones = np.clip(w_overtones, 0.0, 1.0)

    np.save(output_file_path, w_overtones)
