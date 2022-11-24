# In this file, we define download_model
# It runs during container build time to get model weights built into the container

from diffusers import StableDiffusionPipeline, LMSDiscreteScheduler
import os
# import importlib
# import torch
from omegaconf import OmegaConf

# def instantiate_from_config(config):
#     if not "target" in config:
#         if config == '__is_first_stage__':
#             return None
#         elif config == "__is_unconditional__":
#             return None
#         raise KeyError("Expected key `target` to instantiate.")
#     return get_obj_from_str(config["target"])(**config.get("params", dict()))


# def get_obj_from_str(string, reload=False):
#     module, cls = string.rsplit(".", 1)
#     if reload:
#         module_imp = importlib.import_module(module)
#         importlib.reload(module_imp)
#     return getattr(importlib.import_module(module, package=None), cls)

# def load_model_from_config(config, ckpt, verbose=False):
#     print(f"Loading model from {ckpt}")
#     pl_sd = torch.load(ckpt, map_location="cpu")
#     if "global_step" in pl_sd:
#         print(f"Global Step: {pl_sd['global_step']}")
#     sd = pl_sd["state_dict"]
#     model = instantiate_from_config(config.model)
#     m, u = model.load_state_dict(sd, strict=False)
#     if len(m) > 0 and verbose:
#         print("missing keys:")
#         print(m)
#     if len(u) > 0 and verbose:
#         print("unexpected keys:")
#         print(u)

#     model.cuda()
#     model.eval()
#     return model

def download_model():
    # do a dry run of loading the huggingface model, which will download weights at build time
    #Set auth token which is required to download stable diffusion model weights
    HF_AUTH_TOKEN = os.getenv("HF_AUTH_TOKEN")

    config = OmegaConf.load(f"v2-inference.yml")
    # model = load_model_from_config(config, f"{opt.ckpt}")


    lms = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear")

    model = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2", config=config, scheduler=lms, use_auth_token=HF_AUTH_TOKEN)

if __name__ == "__main__":
    download_model()
