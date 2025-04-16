import torch
import gradio as gr
from hi_diffusers import HiDreamImagePipeline
from hi_diffusers import HiDreamImageTransformer2DModel
from hi_diffusers.schedulers.fm_solvers_unipc import FlowUniPCMultistepScheduler
from hi_diffusers.schedulers.flash_flow_match import FlashFlowMatchEulerDiscreteScheduler
from transformers import LlamaForCausalLM, PreTrainedTokenizerFast
from accelerate import Accelerator
from accelerate.utils import set_seed
import argparse
import os

# os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['NCCL_P2P_DISABLE'] = '1' # for old NVIDIA driver
os.environ['NCCL_IB_DISABLE'] = '1'

import monkey_patch_cat
monkey_patch_cat.apply_patch()

# Add command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--device_map", type=str, default="balanced", help="Device map strategy: 'auto', 'balanced', 'balanced_low_0', etc.")
args = parser.parse_args()
device_map = args.device_map

# Initialize Accelerator
accelerator = Accelerator()

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
def load_models(model_type):
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
def generate_image(model_type, prompt, resolution, seed):
    global pipe, current_model
    
    # Reload model if needed
    if model_type != current_model:
        del pipe
        torch.cuda.empty_cache()
        print(f"Loading {model_type} model...")
        pipe, config = load_models(model_type)
        current_model = model_type
        print(f"{model_type} model loaded successfully!")
    
    # Get configuration for current model
    config = MODEL_CONFIGS[model_type]
    guidance_scale = config["guidance_scale"]
    num_inference_steps = config["num_inference_steps"]
    
    # Parse resolution
    height, width = parse_resolution(resolution)
    
    # Handle seed
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

# Initialize with default model

current_model = "full"
print(f"Loading {current_model} model...")
pipe, _ = load_models(current_model)
print("Model loaded successfully!")

# Print current model module device mapping
if hasattr(pipe, "device_map") and pipe.device_map:
    print("\nModel deployment device mapping:")
    for name, device in pipe.device_map.items():
        print(f"- {name}: {device}")

# Create Gradio interface
with gr.Blocks(title="HiDream Image Generator") as demo:
    gr.Markdown("# HiDream Image Generator")
    
    with gr.Row():
        with gr.Column():
            model_type = gr.Radio(
                choices=list(MODEL_CONFIGS.keys()),
                value="full",
                label="Model Type",
                info="Select model variant"
            )
            
            prompt = gr.Textbox(
                label="Prompt", 
                placeholder="A cat holding a sign that says \"Hi-Dreams.ai\".", 
                lines=3
            )
            
            resolution = gr.Radio(
                choices=RESOLUTION_OPTIONS,
                value=RESOLUTION_OPTIONS[0],
                label="Resolution",
                info="Select image resolution"
            )
            
            seed = gr.Number(
                label="Seed (use -1 for random)", 
                value=-1, 
                precision=0
            )
            
            generate_btn = gr.Button("Generate Image")
            seed_used = gr.Number(label="Seed Used", interactive=False)
            
        with gr.Column():
            output_image = gr.Image(label="Generated Image", type="pil")
    
    generate_btn.click(
        fn=generate_image,
        inputs=[model_type, prompt, resolution, seed],
        outputs=[output_image, seed_used]
    )

# Launch app
if __name__ == "__main__":
    demo.launch()
