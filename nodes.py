import os, sys, json, gc
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
# from stable_audio_tools.inference.generation import generate_diffusion_cond
from stable_audio_tools.models.utils import load_ckpt_state_dict

from stable_audio_tools.inference.generation import generate_diffusion_cond, generate_diffusion_uncond
from stable_audio_tools.inference.utils import prepare_audio
from stable_audio_tools.training.utils import copy_state_dict
from aeiou.viz import audio_spectrogram_image

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
ACKPT_FOLDER = "models/audio_checkpoints/"

class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False

base_path = os.path.dirname(os.path.realpath(__file__))
os.makedirs(ACKPT_FOLDER, exist_ok=True)

# Our any instance wants to be a wildcard string
any = AnyType("audio")
def get_models_path(ckpt_name):
    if not ckpt_name:
        return None
    return f"{ACKPT_FOLDER}{ckpt_name}"

model_files = [os.path.basename(file) for file in glob.glob(f"{ACKPT_FOLDER}*.safetensors")] + [os.path.basename(file) for file in glob.glob("models/audio_checkpoints/*.ckpt")]
config_files = [os.path.basename(file) for file in glob.glob(f"{ACKPT_FOLDER}*.json")]
if len(model_files) == 0:
    model_files.append(f"Put models in {ACKPT_FOLDER}")

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
        #return_latents=True,
    )

    # Rearrange audio batch to a single sequence
    output = rearrange(output, "b d n -> d (b n)")
    # Peak normalize, clip, convert to int16
    output = output.to(torch.float32).div(torch.max(torch.abs(output))).clamp(-1, 1).mul(32767).to(torch.int16).cpu()
    
    # Save Audio to output
    if save:
        save_audio_files(output, sample_rate, save_prefix, counter)

    # Generate the spectrogram
    spectrogram = audio_spectrogram_image(output, sample_rate=sample_rate)
    
    # Convert to bytes
    audio_bytes = output.numpy().tobytes()

    # GC and memory Cleanup
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    
    return audio_bytes, sample_rate, spectrogram

def get_model(model_filename=None, config=None, repo=None, half_precision=False):
    #print(model_filename, config, repo, half_precision)
    if model_filename:
        model_path = get_models_path(model_filename) #f"models/audio_checkpoints/{model_filename}"
        if model_filename.endswith(".safetensors") or model_filename.endswith(".ckpt"):
            if not config:
                model_config = get_model_config()
            else:
                with open(config, 'r') as f:
                    model_config = json.load(f)
            model = create_model_from_config(model_config)
            print(model_path)
            model.load_state_dict(load_ckpt_state_dict(model_path))
        else:
            repo_id = "stabilityai/stable-audio-open-1.0" if not repo else repo
            print(f"Loading pretrained model {repo_id}")
            model, model_config = get_pretrained_model(repo_id)
        sample_rate = model_config["sample_rate"]
        sample_size = model_config["sample_size"]
    elif repo:
        if repo == "stabilityai/stable-audio-open-1.0":
            print(f"Loading pretrained model {repo}")
            model, model_config = get_pretrained_model(repo)
        else:
            json_path = config or repo_path(repo, "model_config.json")
            model_path = repo_path(repo, "model.safetensors")
            with open(json_path) as f:
                model_config = json.load(f)
            model = create_model_from_config(model_config)
            model.load_state_dict(load_ckpt_state_dict(model_path), strict=False)
        sample_rate = model_config["sample_rate"]
        sample_size = model_config["sample_size"]
    else:
        raise ValueError("You must specify an Audio Checkpoint or a Repo to load from.")
    
    model = model.to(device) #.eval().requires_grad_(False)
    
    if half_precision:
        model.to(torch.float16)
    
    return (model, sample_rate, sample_size)

def load_model(model_config=None, model_ckpt_path=None, pretrained_name=None, pretransform_ckpt_path=None, device="cuda", model_half=False):
    global model, sample_rate, sample_size
    
    if pretrained_name is not None:
        print(f"Loading pretrained model {pretrained_name}")
        model, model_config = get_pretrained_model(pretrained_name)

    elif model_config is not None and model_ckpt_path is not None:
        print(f"Creating model from config")
        model = create_model_from_config(model_config)

        print(f"Loading model checkpoint from {model_ckpt_path}")
        # Load checkpoint
        copy_state_dict(model, load_ckpt_state_dict(model_ckpt_path))
        #model.load_state_dict(load_ckpt_state_dict(model_ckpt_path))

    sample_rate = model_config["sample_rate"]
    sample_size = model_config["sample_size"]

    if pretransform_ckpt_path is not None:
        print(f"Loading pretransform checkpoint from {pretransform_ckpt_path}")
        model.pretransform.load_state_dict(load_ckpt_state_dict(pretransform_ckpt_path), strict=False)
        print(f"Done loading pretransform")

    model.to(device).eval().requires_grad_(False)

    if model_half:
        model.to(torch.float16)
        
    print(f"Done loading model")

    return model, model_config

def save_audio_files(output, sample_rate, filename_prefix, counter):
    filename_prefix += ""
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    
    for i, audio in enumerate(output):
        file_path = os.path.join(output_dir, f"{filename_prefix}_{counter:05}.wav")
        torchaudio.save(file_path, audio.unsqueeze(0), sample_rate)
        counter += 1

from aeiou.viz import spectrogram_image

def create_image_batch(spectrograms, batch_size):
    images = []
    for spec in spectrograms:
        im = spec.convert("RGB")  # Ensure image is in RGB format
        im_tensor = torch.tensor(np.array(im))  # Convert to tensor, keeping the dimensions as (height, width, channels)
        images.append(im_tensor)
    batch_tensor = torch.stack(images)  # Stack images into a batch
    return batch_tensor

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

    RETURN_TYPES = ("VHS_AUDIO", "INT", "IMAGE")
    RETURN_NAMES = ("audio", "sample_rate", "image")
    FUNCTION = "sample"
    OUTPUT_NODE = True

    CATEGORY = "audio/samplers"

    def sample(self, audio_model, positive, negative, seed, steps, cfg_scale, sample_size, sigma_min, sigma_max, sampler_type, save, save_prefix):
        for key, value in locals().items():
            print(f"{key}: {value}")
        audio_bytes, sample_rate, spectrogram = generate_audio((positive, negative), steps, cfg_scale, sample_size, sigma_min, sigma_max, sampler_type, device, save, save_prefix, audio_model, seed=seed, counter=self.counter)
        spectrograms = create_image_batch([spectrogram], 1)
        return (audio_bytes, sample_rate, spectrograms)

class StableLoadAudioModel:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_filename": (model_files, ),
            },
            "optional": {
                "model_config": (config_files, ),
                "repo": ("STRING", {"default": "stabilityai/stable-audio-open-1.0"}),
                "half_precision": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("SAOMODEL", )
    RETURN_NAMES = ("audio_model", )
    FUNCTION = "load"

    CATEGORY = "audio/loaders"

    def load(self, model_filename, model_config=None, repo=None, half_precision=None):
        mpath = get_models_path(model_config)
        modelinfo = get_model(model_filename=model_filename, config=mpath, repo=repo, half_precision=half_precision)
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

    CATEGORY = "audio/conditioning"

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

    CATEGORY = "audio/conditioning"

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
