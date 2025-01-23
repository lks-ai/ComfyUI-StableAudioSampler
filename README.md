# ComfyUI-StableAudioSampler
The New Stable Audio Open 1.0 Sampler In a ComfyUI Node. Make some beats!
![image](https://github.com/lks-ai/ComfyUI-StableAudioSampler/assets/163685473/477272f3-46c5-46e5-8de9-d74a93e91716)

## An Example I Pasted Together
In this [workflow](https://github.com/lks-ai/ComfyUI-StableAudioSampler/blob/main/workflows/audio-space-exploration.json), I got random `cfg_scale`, `sigma_min` and `step` values making variations on the same 16 beats; same `prompt` and `seed`. **VOLUME WARNING!**

https://github.com/lks-ai/ComfyUI-StableAudioSampler/assets/163685473/5f43db75-cc35-47f3-999b-6f65f91420eb

# Caveats
- The longer your audio, the more VRAM you need to stitch it together
- on a 3060, we've tried up to 10 seconds so far

## Installation

### Download the Model and Config
1. Go to [Stable Audio Open on HuggingFace](https://huggingface.co/stabilityai/stable-audio-open-1.0/tree/main/) and download the `model.safetensors` and `model.config.json` files.
2. Place the files in the `models/audio_checkpoints` folder. If you don't have one, make one in your comfy folder.
3. Open Comfy and StableAudioLoader will see your model and config

### With a HuggingFace Token
1. Make sure you have your `HF_TOKEN` environment variable for hugging face because model loading doesn't work just yet directly from a saved file
2. Go ahead and download model from here for when we fix that [Stable Audio Open on HuggingFace](https://huggingface.co/stabilityai/stable-audio-open-1.0/tree/main/model.safetensors)
3. Make sure to run `pip install -r requirements.txt` inside the repo folder if you're not using Manager
4. It should just run if you've got your environment variable set up

There will definitely be issues because this is so new and it was coded quickly so we couldn't test it out.

This is not an official StableAudioOpen repository.

# Bonus Installation Tips

For users experiencing installation difficulties, thanks to @marius-jopen, additional information can be found [here](https://github.com/lks-ai/ComfyUI-StableAudioSampler/issues/30#issuecomment-2609436644).

Thanks to @camenduru, stable-audio-open-1.0 can also be obtained [here](https://huggingface.co/audo/stable-audio-open-1.0/tree/main) without the need to be logged in to your HuggingFace account.

## Current Features
- Load your own models!
- Runs in half precision optional
- Nodes
  - A Sampler Node: now with seed control, positive and negative prompts
  - A Pre-Conditioning Node: kind of like empty latent audio with batch option
  - A Prompt Node: Pipes conditioning
  - A Model Loading Node: Includes repo options and scans `models/audio_checkpoints` for models and config.json files
- `control_after_generate` option
- Audio to Audio (like in the Gradio Example) **Still working on fix for this**
- Can still use HF env key if you want
- Generates audio and outputs raw bytes and a sample rate for use with VHS
- Includes all of the original Stable Audio Open parameters
- Sampler outputs a Spectrogram image (experimental)
- Can save audio to file
- New Prefix Templates for save file naming
- Outputs a temporary `wav` to `temp/stableaudiosampler.wav` you can use for looping like in [this video](https://www.youtube.com/watch?v=_eR6tP-c8W4).

### Example Workflows
#### [Exploring Same Prompt and Seed](https://github.com/lks-ai/ComfyUI-StableAudioSampler/blob/main/workflows/audio-space-exploration.json)

The part I use AnyNode for is just getting random values within a range for `cfg_scale`, `steps` and `sigma_min` thanks to feedback from the community and some tinkering, I think I found a way in this workflow to just get endless sequences of the same seed/prompt in any key (because I mentioned what key the synth lead needed to be in).

With the new save prefix templating, it makes it easy to just look at the file and know what settings (since wav doesn't have PNGinfo)

## Roadmap and Requested Features
Keeping track of requests and ideas as they come in:
- Stereo output
- Nodes
  - A Mixer Node (mix your audio outputs with some sort of mastering)
  - A Tiling Sampler (concatenate the audios)
- More Sampler Node Options
  - Gain
  - Possibly Clipping at some dB
  - Cleaning up some of the current options with selectors, etc.
- Upfi (upscaling fidelity)
- Audio Preview Node

## Error: `progressbar`
If you get the `progressbar` error, you can use our new utility from the latest update.
```bash
cd ComfyUI/custom_modules/ComfyUI-StableAudioSampler
python util_discrepancies.py progressbar
```
You will see something like this...
![Screenshot from 2024-06-13 13-02-30](https://github.com/lks-ai/ComfyUI-StableAudioSampler/assets/163685473/5ce10eb3-d841-4d21-bd48-93d697cff3d8)
In this screenshot, you see `protobuf` but that is only because I don't have version issues with `progressbar`.
**Note**: If I install one of those version suggestions, StableAudioSampler should work, but at the same time, it might make other packages not work.

# Contributions   
We are very open to anyone who wants to contribute from the open source community. Make your forks and pull requests. We will build something cool. If it's already on the roadmap, chances are we're already working on it, so check in with us. We will start a dev branch.

# Feature Requests
If you have a request for a feature, open an issue about it and it will be seen.

Happy Diffusing!
