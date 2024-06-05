import os
import glob
import torch
import torchaudio
from einops import rearrange
from stable_audio_tools import get_pretrained_model
from stable_audio_tools.inference.generation import generate_diffusion_cond

from safetensors.torch import load_file
from .util_config import get_model_config
from stable_audio_tools.models.factory import create_model_from_config
from stable_audio_tools.models.utils import load_ckpt_state_dict

device = "cuda" if torch.cuda.is_available() else "cpu"

class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False
base_path = os.path.dirname(os.path.realpath(__file__))
os.makedirs("models/audio_checkpoints", exist_ok=True)

# Our any instance wants to be a wildcard string
any = AnyType("audio")
model_files = [os.path.basename(file) for file in glob.glob("models/audio_checkpoints/*.safetensors")] + [os.path.basename(file) for file in glob.glob("models/audio_checkpoints/*.ckpt")]
if len(model_files) == 0:
    model_files.append("Put models in models/audio_checkpoints")


def generate_audio(prompt, steps, cfg_scale, sample_size, sigma_min, sigma_max, sampler_type, device, save, save_path, model_filename):
    model_path = f"models/audio_checkpoints/{model_filename}"
    if model_filename.endswith(".safetensors") or model_filename.endswith(".ckpt"):
        model = create_model_from_config(get_model_config())
        model.load_state_dict(load_ckpt_state_dict(model_path))
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

    output = generate_diffusion_cond(
        model,
        steps=steps,
        cfg_scale=cfg_scale,
        conditioning=conditioning,
        sample_size=sample_size,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        sampler_type=sampler_type,
        device=device
    )

    output = rearrange(output, "b d n -> d (b n)")

    output = output.to(torch.float32).div(torch.max(torch.abs(output))).clamp(-1, 1).mul(32767).to(torch.int16).cpu()
    
    if save:
        torchaudio.save("output/" + save_path, output, sample_rate)
    
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
                "sampler_type": ("STRING", {"default": "dpmpp-3m-sde"}),
                "save": ("BOOLEAN", {"default": True}),
                "save_path": ("STRING", {"default": "output.wav"}),
            }
        }

    RETURN_TYPES = (any, "INT")
    FUNCTION = "sample"
    OUTPUT_NODE = True

    CATEGORY = "audio"

    def sample(self, prompt, steps, cfg_scale, sample_size, sigma_min, sigma_max, sampler_type, save, save_path, model_filename):
        audio_bytes, sample_rate = generate_audio(prompt, steps, cfg_scale, sample_size, sigma_min, sigma_max, sampler_type, device, save, save_path, model_filename)
        return (audio_bytes, sample_rate)

NODE_CLASS_MAPPINGS = {
    "StableAudioSampler": StableAudioSampler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "StableAudioSampler": "Stable Diffusion Audio Sampler",
}
