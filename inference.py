import torch
import argparse
from hi_diffusers import HiDreamImagePipeline
from hi_diffusers import HiDreamImageTransformer2DModel
from hi_diffusers.schedulers.fm_solvers_unipc import FlowUniPCMultistepScheduler
from hi_diffusers.schedulers.flash_flow_match import FlashFlowMatchEulerDiscreteScheduler
from transformers import LlamaForCausalLM, PreTrainedTokenizerFast
from accelerate import Accelerator
from accelerate.utils import set_seed
import os

MODEL_PREFIX = "HiDream-ai"
# LLAMA_MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"
LLAMA_MODEL_NAME = "mlabonne/Meta-Llama-3.1-8B-Instruct-abliterated"
# Model configurations
MODEL_CONFIGS = {
    "dev": {
        "path": f"{MODEL_PREFIX}/HiDream-I1-Dev",
        "guidance_scale": 0.0,
        "num_inference_steps": 28,
        "shift": 6.0,
        "scheduler": FlashFlowMatchEulerDiscreteScheduler
    },
    "full": {
        "path": f"{MODEL_PREFIX}/HiDream-I1-Full",
        "guidance_scale": 5.0,
        "num_inference_steps": 50,
        "shift": 3.0,
        "scheduler": FlowUniPCMultistepScheduler
    },
    "fast": {
        "path": f"{MODEL_PREFIX}/HiDream-I1-Fast",
        "guidance_scale": 0.0,
        "num_inference_steps": 16,
        "shift": 3.0,
        "scheduler": FlashFlowMatchEulerDiscreteScheduler
    }
}

# Resolution options
RESOLUTION_OPTIONS = [
    "1024 × 1024 (Square)",
    "768 × 1360 (Portrait)",
    "1360 × 768 (Landscape)",
    "880 × 1168 (Portrait)",
    "1168 × 880 (Landscape)",
    "1248 × 832 (Landscape)",
    "832 × 1248 (Portrait)"
]

# Load models
def load_models(model_type, device_map):
    config = MODEL_CONFIGS[model_type]
    pretrained_model_name_or_path = config["path"]
    scheduler = MODEL_CONFIGS[model_type]["scheduler"](num_train_timesteps=1000, shift=config["shift"], use_dynamic_shifting=False)
    
    print(f"Using device mapping strategy: {device_map}")
    print(f"Available GPU count: {torch.cuda.device_count()}")
    
    # Load tokenizer (doesn't need to be on GPU)
    tokenizer_4 = PreTrainedTokenizerFast.from_pretrained(
        LLAMA_MODEL_NAME,
        use_fast=False)
    
    # Use device_map to distribute text encoder across multiple GPUs
    text_encoder_4 = LlamaForCausalLM.from_pretrained(
        LLAMA_MODEL_NAME,
        output_hidden_states=True,
        output_attentions=True,
        device_map=device_map,
        torch_dtype=torch.bfloat16)

    # Use device_map to distribute transformer model across multiple GPUs
    transformer = HiDreamImageTransformer2DModel.from_pretrained(
        pretrained_model_name_or_path, 
        subfolder="transformer",
        device_map=device_map,
        torch_dtype=torch.bfloat16)

    # Load pipeline and configure device_map
    pipe = HiDreamImagePipeline.from_pretrained(
        pretrained_model_name_or_path, 
        scheduler=scheduler,
        tokenizer_4=tokenizer_4,
        text_encoder_4=text_encoder_4,
        device_map=device_map,
        torch_dtype=torch.bfloat16
    )
    pipe.transformer = transformer
    
    return pipe, config

# Parse resolution string to get height and width
def parse_resolution(resolution_str):
    if "1024 × 1024" in resolution_str:
        return 1024, 1024
    elif "768 × 1360" in resolution_str:
        return 768, 1360
    elif "1360 × 768" in resolution_str:
        return 1360, 768
    elif "880 × 1168" in resolution_str:
        return 880, 1168
    elif "1168 × 880" in resolution_str:
        return 1168, 880
    elif "1248 × 832" in resolution_str:
        return 1248, 832
    elif "832 × 1248" in resolution_str:
        return 832, 1248
    else:
        return 1024, 1024  # Default fallback

# Generate image function
def generate_image(pipe, model_type, prompt, resolution, seed):
    # Get current model configuration
    config = MODEL_CONFIGS[model_type]
    guidance_scale = config["guidance_scale"]
    num_inference_steps = config["num_inference_steps"]
    
    # Parse resolution
    height, width = parse_resolution(resolution)
    
    # Handle random seed
    if seed == -1:
        seed = torch.randint(0, 1000000, (1,)).item()
    
    # All available GPUs should already be used by the model, no need to manually specify generator's device
    generator = torch.Generator().manual_seed(seed)
    set_seed(seed)  # Set global seed to ensure consistency of results
    
    # Execute inference
    with torch.inference_mode():
        images = pipe(
            prompt,
            height=height,
            width=width,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            num_images_per_prompt=1,
            generator=generator
        ).images
    
    return images[0], seed

if __name__ == "__main__":
    # os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    os.environ['NCCL_P2P_DISABLE'] = '1' # for old NVIDIA driver
    os.environ['NCCL_IB_DISABLE'] = '1'

    import monkey_patch_cat
    monkey_patch_cat.apply_patch()

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default="dev")
    parser.add_argument("--device_map", type=str, default="balanced", help="Device map strategy: 'auto', 'balanced', 'balanced_low_0', etc.")
    args = parser.parse_args()
    model_type = args.model_type
    device_map = args.device_map

    # Initialize Accelerator
    accelerator = Accelerator()

    # Initialize default model
    print(f"Loading {model_type} model to multiple GPUs...")
    pipe, _ = load_models(model_type, device_map)
    print("Model loaded successfully!")

    # Print current model module device mapping
    if hasattr(pipe, "device_map") and pipe.device_map:
        print("\nModel deployment device mapping:")
        for name, device in pipe.device_map.items():
            print(f"- {name}: {device}")

    prompt = "A cat holding a sign that says \"Hi-Dreams.ai\"."
    prompt = "A detailed and futuristic illustration of a computer processor hardware design process. Show a close-up of a high-tech semiconductor chip with intricate circuitry on a blue background. Highlight a vulnerability in the design, such as a glowing red area or a crack in the circuitry. Next to the chip, depict engineers or AI systems analyzing the design on holographic screens, identifying and fixing the issue. Include symbols of security, such as a shield icon or a lock, to represent the threat mitigation. Emphasize a before-and-after effect, with the flawed design on one side and the repaired, secure design on the other. Use a modern, tech-inspired color palette with blues, silvers, and reds to convey innovation and urgency."
    resolution = "1024 × 1024 (Square)"
    seed = -1
    print(f"Generating image, prompt: '{prompt}'")
    image, used_seed = generate_image(pipe, model_type, prompt, resolution, seed)
    print(f"Image generation completed! Seed used: {used_seed}")
    output_path = "output.png"
    image.save(output_path)
    print(f"Image saved to: {output_path}")