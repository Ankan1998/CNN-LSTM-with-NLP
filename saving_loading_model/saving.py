import torch
import shutil

def save_checkpoint(ckp_dict, is_best, ckpt_path, best_model_path):
    f_path = ckpt_path
    torch.save(ckp_dict, f_path)
    if is_best:
        best_fpath = best_model_path
        shutil.copyfile(f_path, best_fpath)
