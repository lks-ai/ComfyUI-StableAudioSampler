import os, sys, json
import glob
import torch
import torchaudio
from einops import rearrange
from stable_audio_tools import get_pretrained_model
from stable_audio_tools.inference.generation import generate_diffusion_cond
import numpy as np
from safetensors.torch import load_file
from .util_config import get_model_config
# from stable_audio_tools.models.factory import create_model_from_config
# from stable_audio_tools.models.utils import load_ckpt_state_dict
from stable_audio_tools import get_pretrained_model, create_model_from_config
from stable_audio_tools.inference.generation import generate_diffusion_cond
from stable_audio_tools.models.utils import load_ckpt_state_dict

# Test current setup
# Add in Audio2Audio

# Comfy libs
def add_comfy_path():
    current_path = os.path.dirname(os.path.abspath(__file__))
    comfy_path = os.path.abspath(os.path.join(current_path, '../../../comfy'))
    if comfy_path not in sys.path:
        sys.path.insert(0, comfy_path)

add_comfy_path()

import folder_paths # type: ignore

device = "cuda" if torch.cuda.is_available() else "cpu"
MAX_FP32 = np.iinfo(np.int32).max
SCHEDULERS = ["dpmpp-3m-sde", "dpmpp-2m-sde", "k-heun", "k-lms", "k-dpmpp-2s-ancestral", "k-dpm-2", "k-dpm-fast"]

class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False

base_path = os.path.dirname(os.path.realpath(__file__))
os.makedirs("models/audio_checkpoints", exist_ok=True)

# Our any instance wants to be a wildcard string
any = AnyType("audio")
def get_cpkt_path(ckpt_name):
    return f"models/audio_checkpoints/{ckpt_name}"

model_files = [os.path.basename(file) for file in glob.glob(f"models/audio_checkpoints/*.safetensors")] + [os.path.basename(file) for file in glob.glob("models/audio_checkpoints/*.ckpt")]
if len(model_files) == 0:
    model_files.append("Put models in models/audio_checkpoints")

def repo_path(repo, filename):
    path = os.path.join(repo, filename)
    instance_path = os.path.normpath(path)
    if sys.platform == 'win32':
        instance_path = instance_path.replace('\\', "/")
    return instance_path

def generate_audio(cond_batch, steps, cfg_scale, sample_size, sigma_min, sigma_max, sampler_type, device, save, save_prefix, modelinfo, batch_size=1, seed=-1, after_generate="randomize", counter=0):
    model, sample_rate, sample_size = modelinfo
    b_pos, b_neg = cond_batch
    p_conditioning, p_batch_size = b_pos
    n_conditioning, n_batch_size = b_neg

    output = generate_diffusion_cond(
        model,
        steps=steps,
        cfg_scale=cfg_scale,
        conditioning=p_conditioning,
        negative_conditioning=n_conditioning,
        sample_size=sample_size,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        sampler_type=sampler_type,
        device=device,
        seed=seed,
        batch_size=p_batch_size,
        return_latents=True,
    )

    output = rearrange(output, "b d n -> d (b n)")

    output = output.to(torch.float32).div(torch.max(torch.abs(output))).clamp(-1, 1).mul(32767).to(torch.int16).cpu()
    
    if save:
        save_audio_files(output, sample_rate, save_prefix, counter)
    
    # Convert to bytes
    audio_bytes = output.numpy().tobytes()
    
    return audio_bytes, sample_rate

def get_model(model_filename=None, repo=None):
    if model_filename:
        model_path = get_cpkt_path(model_filename) #f"models/audio_checkpoints/{model_filename}"
        if model_filename.endswith(".safetensors") or model_filename.endswith(".ckpt"):
            model_config = get_model_config()
            model = create_model_from_config(model_config)
            print(model_path)
            model.load_state_dict(load_ckpt_state_dict(model_path))
        else:
            model, model_config = get_pretrained_model("stabilityai/stable-audio-open-1.0")
        sample_rate = model_config["sample_rate"]
        sample_size = model_config["sample_size"]
    elif repo:
        if repo == "stabilityai/stable-audio-open-1.0":
            model, model_config = get_pretrained_model(repo)
        else:
            json_path = repo_path(repo, "model_config.json")
            model_path = repo_path(repo, "model.safetensors")
            with open(json_path) as f:
                model_config = json.load(f)
            model = create_model_from_config(model_config)
            model.load_state_dict(load_ckpt_state_dict(model_path), strict=False)
        sample_rate = model_config["sample_rate"]
        sample_size = model_config["sample_size"]

    return (model.to(device), sample_rate, sample_size)

def save_audio_files(output, sample_rate, filename_prefix, counter):
    filename_prefix += ""
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    
    for i, audio in enumerate(output):
        file_path = os.path.join(output_dir, f"{filename_prefix}_{counter:05}.wav")
        torchaudio.save(file_path, audio.unsqueeze(0), sample_rate)
        counter += 1

class StableAudioSampler:
    def __init__(self):
        self.counter = 0
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "audio_model": ("SAOMODEL", {"forceInput": True}),
                "positive": ("SAOCOND", {"forceInput": True}),
                "negative": ("SAOCOND", {"forceInput": True}),
                "seed": ("INT", {"default": -1, "min": -1, "max": MAX_FP32}),
                "steps": ("INT", {"default": 100, "min": 1, "max": 10000}),
                "cfg_scale": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "sample_size": ("INT", {"default": 65536, "min": 1, "max": 1000000}),
                "sigma_min": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1000.0, "step": 0.01}),
                "sigma_max": ("FLOAT", {"default": 500.0, "min": 0.0, "max": 1000.0, "step": 0.01}),
                "sampler_type": (SCHEDULERS, {"default": "dpmpp-3m-sde"}),
                "save": ("BOOLEAN", {"default": True}),
                "save_prefix": ("STRING", {"default": "StableAudio"}),
            }
        }

    RETURN_TYPES = ("VHS_AUDIO", "INT")
    RETURN_NAMES = ("audio", "sample_rate")
    FUNCTION = "sample"
    OUTPUT_NODE = True

    CATEGORY = "audio/samplers"

    def sample(self, audio_model, positive, negative, seed, steps, cfg_scale, sample_size, sigma_min, sigma_max, sampler_type, save, save_prefix):
        for key, value in locals().items():
            print(f"{key}: {value}")
        audio_bytes, sample_rate = generate_audio((positive, negative), steps, cfg_scale, sample_size, sigma_min, sigma_max, sampler_type, device, save, save_prefix, audio_model, seed=seed, counter=self.counter)
        return (audio_bytes, sample_rate)

class StableLoadAudioModel:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_filename": (model_files, ),
            },
            "optional": {
                "repo": ("STRING", {"default": "stabilityai/stable-audio-open-1.0"})
            }
        }

    RETURN_TYPES = ("SAOMODEL", )
    RETURN_NAMES = ("audio_model", )
    FUNCTION = "load"

    CATEGORY = "audio/loaders"

    def load(self, model_filename, repo=None):
        modelinfo = get_model(model_filename=model_filename, repo=repo)
        return (modelinfo,)
    
class StableAudioPrompt:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "conditioning": ("SAOCOND", {"forceInput": True}),
                "prompt": ("STRING", {"default": "", "multiline": True}),
            }
        }
 
    RETURN_TYPES = ("SAOCOND", )
    RETURN_NAMES = ("conditioning", )
    FUNCTION = "go"

    CATEGORY = "audio/loaders"

    def go(self, conditioning, prompt):
        cond, batch_size = conditioning
        cond[0]['prompt'] = prompt
        #c = conditioning[0]
        # conditioning = [{
        #     "prompt": prompt,
        #     "seconds_start": seconds_start,
        #     "seconds_total": seconds_total
        # }]
        return ((cond, batch_size), )

class StableAudioConditioning:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "seconds_start": ("INT", {"default": 0, "min": 0, "max": 60, "step": 1, "display": "number"}),
                "seconds_total": ("INT", {"default": 30, "min": 0, "max": 60, "step": 1, "display": "number"}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 50, "step": 1, "display": "number"}),
            }
        }
 
    RETURN_TYPES = ("SAOCOND", )
    RETURN_NAMES = ("conditioning", )
    FUNCTION = "go"

    CATEGORY = "audio/loaders"

    def go(self, seconds_start, seconds_total, batch_size):
        conditioning = [{
            "prompt": None,
            "seconds_start": seconds_start,
            "seconds_total": seconds_total
        }]
        return ((conditioning, batch_size), )

NODE_CLASS_MAPPINGS = {
    "StableAudioSampler": StableAudioSampler,
    "StableAudioLoadModel": StableLoadAudioModel,
    "StableAudioPrompt": StableAudioPrompt,
    "StableAudioConditioning": StableAudioConditioning,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "StableAudioSampler": "Stable Audio Sampler",
    "StableAudioLoadModel": "Load Stable Audio Model",
    "StableAudioPrompt": "Stable Audio Prompt",
    "StableAudioConditioning": "Stable Audio Pre-Conditioning"
}
