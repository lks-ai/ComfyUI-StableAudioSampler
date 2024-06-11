import os
import json
import torch
import torchaudio
import folder_paths

from einops import rearrange
from stable_audio_tools.inference.generation import generate_diffusion_cond

from .util_config import get_model_config
from stable_audio_tools.models.factory import create_model_from_config
from stable_audio_tools.models.utils import load_ckpt_state_dict

class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False

any = AnyType("*")

SAMPLERS = ["k-heun", "k-lms", "k-dpmpp-2s-ancestral", "k-dpm-2", "k-dpm-fast", "k-dpm-adaptive", "dpmpp-2m-sde", "dpmpp-3m-sde"]

if "audio_configs" not in folder_paths.folder_names_and_paths:
    folder_paths.folder_names_and_paths["audio_configs"] = ([os.path.join(folder_paths.models_dir, "audio_checkpoints")], [".json"])
if "audio_checkpoints" not in folder_paths.folder_names_and_paths:
    folder_paths.folder_names_and_paths["audio_checkpoints"] = ([os.path.join(folder_paths.models_dir, "audio_checkpoints")], [".ckpt", ".safetensors"])
    models_dir = os.path.join(folder_paths.models_dir, "audio_checkpoints")
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
model_files = folder_paths.get_filename_list("audio_checkpoints")
config_files = folder_paths.get_filename_list("audio_configs")

if len(model_files) == 0:
    model_files.append("Put models in models/audio_checkpoints")

class StableAudioSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "audio_model": ("SAOMODEL", {"forceInput": True}),
                "positive": ("SAOCOND", {"forceInput": True}),
                "negative": ("SAOCOND", {"forceInput": True}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 100, "min": 1, "max": 10000}),
                "cfg_scale": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "sigma_min": ("FLOAT", {"default": 0.3, "min": 0.01, "max": 1000.0, "step": 0.01}),
                "sigma_max": ("FLOAT", {"default": 500.0, "min": 0.0, "max": 1000.0, "step": 0.01}),
                "sampler_type": (SAMPLERS, {"default": "dpmpp-3m-sde"}),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "save": ("BOOLEAN", {"default": True}),
                "save_prefix": ("STRING", {"default": "StableAudio"}),
            },
            "optional": {
                "audio": (any, )
            }
        }

    RETURN_TYPES = (any, "INT")
    FUNCTION = "sample"
    OUTPUT_NODE = True

    CATEGORY = "audio"

    def sample(self, audio_model, positive, negative, seed, steps, cfg_scale, sigma_min, sigma_max, sampler_type, denoise, save, save_prefix, audio=None):

        model, sample_rate, sample_size, = audio_model

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        b_pos = positive
        b_neg = negative
        p_conditioning, p_batch_size = b_pos
        n_conditioning, n_batch_size = b_neg
        sample_size = p_conditioning[0]['seconds_total'] * sample_rate

        init_audio = audio
        if init_audio is not None:
            in_sr, init_audio = init_audio
            # Turn into torch tensor, converting from int16 to float32
            init_audio = torch.from_numpy(init_audio).float().div(32767)
            
            if init_audio.dim() == 1:
                init_audio = init_audio.unsqueeze(0) # [1, n]
            elif init_audio.dim() == 2:
                init_audio = init_audio.transpose(0, 1) # [n, 2] -> [2, n]

            if in_sr != sample_rate:
                resample_tf = torchaudio.transforms.Resample(in_sr, sample_rate).to(init_audio.device)
                init_audio = resample_tf(init_audio)

            init_audio = (sample_rate, init_audio)

        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

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
            seed=seed,
            device=device,
            batch_size=p_batch_size,
            init_noise_level=denoise, 
            init_audio=init_audio
        )
        
        gendata = locals()
        gendata['prompt'] = p_conditioning[0]['prompt']
        gendata['negative_prompt'] = n_conditioning[0]['prompt']
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
        file_name = f"{save_prefix}_{counter:05}.wav"
        save_path = os.path.join(save_path, file_name)

        torchaudio.save(save_path, output, sample_rate)
        
        # Convert to bytes
        audio_bytes = output.numpy().tobytes()
        sample_rate = sample_rate
        
        return (audio_bytes, sample_rate)

class StableLoadAudioModel:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_filename": (model_files, ),
            },
            "optional": {
                "model_config": (config_files, ),
            }
        }

    RETURN_TYPES = ("SAOMODEL", )
    RETURN_NAMES = ("audio_model", )
    FUNCTION = "load"

    CATEGORY = "audio/loaders"

    def load(self, model_filename, model_config=None):
        model_path = folder_paths.get_full_path("audio_checkpoints", model_filename)
        config = folder_paths.get_full_path("audio_checkpoints", model_config)
        if not config:
            model_config = get_model_config()
        else:
            with open(config, 'r') as f:
                model_config = json.load(f)
        model = create_model_from_config(model_config)
        print(model_path)
        model.load_state_dict(load_ckpt_state_dict(model_path))
        sample_rate = model_config["sample_rate"]
        sample_size = model_config["sample_size"]

        modelinfo = (model, sample_rate, sample_size)

        return (modelinfo, )

class StableAudioPrompt:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "conditioning": ("SAOCOND", {"forceInput": True}),
                "prompt": ("STRING", {"multiline": True}),
            }
        }
 
    RETURN_TYPES = ("SAOCOND", )
    RETURN_NAMES = ("conditioning", )
    FUNCTION = "go"

    CATEGORY = "audio/conditioning"

    def go(self, conditioning, prompt):
        print("PROMPT", prompt)
        cond, batch_size = conditioning
        print(cond, batch_size)
        o = []
        for v in cond:
            v['prompt'] = prompt
            o.append(v.copy())
        print(o, batch_size)
        return ((o, batch_size), )

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