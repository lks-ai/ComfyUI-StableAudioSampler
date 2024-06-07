import os
import glob
import torch
import torchaudio
from einops import rearrange
from stable_audio_tools import get_pretrained_model
from stable_audio_tools.inference.generation import generate_diffusion_cond
import numpy as np

from safetensors.torch import load_file
from .util_config import get_model_config
from stable_audio_tools.models.factory import create_model_from_config
from stable_audio_tools.models.utils import load_ckpt_state_dict
import folder_paths
import comfy.samplers

device = "cuda" if torch.cuda.is_available() else "cpu"

class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False
base_path = os.path.dirname(os.path.realpath(__file__))
os.makedirs("models/audio_checkpoints", exist_ok=True)

# Our any instance wants to be a wildcard string
any = AnyType("audio")

if "audio_checkpoints" not in folder_paths.folder_names_and_paths:
    folder_paths.folder_names_and_paths["audio_checkpoints"] = ([os.path.join(folder_paths.models_dir, "audio_checkpoints")], [".ckpt", ".safetensors"])
    models_dir = os.path.join(folder_paths.models_dir, "audio_checkpoints")
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
model_files = folder_paths.get_filename_list("audio_checkpoints")

if len(model_files) == 0:
    model_files.append("Put models in models/audio_checkpoints")


def generate_audio(prompt, steps, cfg_scale, sample_size, sigma_min, sigma_max, sampler_type, device, save, model_filename):
    model_path = folder_paths.get_full_path("audio_checkpoints", model_filename)
    if model_filename.endswith(".safetensors") or model_filename.endswith(".ckpt"):
        model_config = get_model_config()
        model = create_model_from_config(model_config)
        model.load_state_dict(load_ckpt_state_dict(model_path))
        sample_rate = model_config["sample_rate"]
        sample_size = model_config["sample_size"]
    else:
        model, model_config = get_pretrained_model("stabilityai/stable-audio-open-1.0")
        sample_rate = model_config["sample_rate"]
        sample_size = model_config["sample_size"]
    
    model = model.to(device)

    conditioning = [{
        "prompt": prompt,
        "seconds_start": 0,
        "seconds_total": 30
    }]
    
    seed = np.random.randint(0, np.iinfo(np.int32).max)

    output = generate_diffusion_cond(
        model,
        steps=steps,
        cfg_scale=cfg_scale,
        conditioning=conditioning,
        sample_size=sample_size,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        sampler_type=sampler_type,
        device=device,
        seed=seed,
    )

    output = rearrange(output, "b d n -> d (b n)")

    output = output.to(torch.float32).div(torch.max(torch.abs(output))).clamp(-1, 1).mul(32767).to(torch.int16).cpu()

    if save:
        save_path = folder_paths.get_output_directory()
    else:
        save_path = folder_paths.get_temp_directory()

    save_path = os.path.join(save_path, "audio")
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    file_counter = len([name for name in os.listdir(save_path) if os.path.isfile(os.path.join(save_path, name))])
    counter = file_counter + 1
    file_name = f"{counter:05}.wav"
    save_path = os.path.join(save_path, file_name)

    torchaudio.save(save_path, output, sample_rate)
    
    # Convert to bytes
    audio_bytes = output.numpy().tobytes()
    
    return audio_bytes, sample_rate

class StableAudioSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "prompt": ("STRING", {"default": "128 BPM tech house drum loop"}),
                "model_filename": (model_files, ),
                "steps": ("INT", {"default": 100, "min": 1, "max": 10000}),
                "cfg_scale": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "sample_size": ("INT", {"default": 65536, "min": 1, "max": 1000000}),
                "sigma_min": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1000.0, "step": 0.01}),
                "sigma_max": ("FLOAT", {"default": 500.0, "min": 0.0, "max": 1000.0, "step": 0.01}),
                #"sampler_type": (comfy.samplers.KSampler.SAMPLERS, {"default": "dpmpp_3m_sde",}),
                "sampler_type": ("STRING", {"default": "dpmpp-3m-sde"}),
                "save": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = (any, "INT")
    FUNCTION = "sample"
    OUTPUT_NODE = True

    CATEGORY = "audio"

    def sample(self, prompt, steps, cfg_scale, sample_size, sigma_min, sigma_max, sampler_type, save, model_filename):
        audio_bytes, sample_rate = generate_audio(prompt, steps, cfg_scale, sample_size, sigma_min, sigma_max, sampler_type, device, save, model_filename)
        return (audio_bytes, sample_rate)

NODE_CLASS_MAPPINGS = {
    "StableAudioSampler": StableAudioSampler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "StableAudioSampler": "Stable Diffusion Audio Sampler",
}
