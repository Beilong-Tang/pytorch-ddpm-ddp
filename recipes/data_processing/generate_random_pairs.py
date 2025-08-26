import os.path as op 
import fire
import os
from pathlib import Path
import tqdm
import numpy as np

def write_dict_to_scp(path, val_dict):
    os.makedirs(Path(path).parent.absolute(), exist_ok=True)
    with open(path, "w") as f:
        for _k, _v in val_dict.items():
            f.write(f"{_k} {_v}\n")
    pass

def read_scp(scp_path):
    res = {}
    with open(scp_path, "r") as f:
        for l in f.readlines():
            l = l.replace("\n", "")
            uttid, val = l.split(" ")
            res[uttid] = val
    return res

def generate(scp, output, img_size: 32, ch = 3):
    res = read_scp(scp)
    ids = list(res.keys())
    os.makedirs(op.join(output, "noise"), exist_ok=True)
    output_res={}
    for _i in tqdm.tqdm(ids):
        noise = np.random.normal(0.0, 1.0, (ch, img_size, img_size))
        _p = op.join(output, "noise", f"{_i}.npy")
        np.save(_p, noise)
        output_res[_i] = _p
    write_dict_to_scp(op.join(output, "noise.scp"), output_res)


if __name__ == "__main__":
    fire.Fire(generate)
    
