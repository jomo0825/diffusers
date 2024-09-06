import argparse
import itertools
import math
import os
from pathlib import Path
from typing import Optional
import subprocess
import sys
import json
import copy

import gc
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.utils.data import Dataset
from transformers import AutoTokenizer, PretrainedConfig
import bitsandbytes as bnb

from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, set_seed
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from contextlib import nullcontext
from peft import LoraConfig, set_peft_model_state_dict
from peft.utils import get_peft_model_state_dict

from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
    FluxPipeline,
    FluxTransformer2DModel,
)

from diffusers.optimization import get_scheduler
from diffusers.training_utils import (
    _set_state_dict_into_text_encoder,
    cast_training_params,
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
)


from huggingface_hub import HfFolder, Repository, whoami
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from safetensors.torch import load_file, save_file
from transformers import CLIPTextModel, CLIPTokenizer, CLIPTextConfig, CLIPTextModelWithProjection, T5TokenizerFast
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from diffusers.utils.torch_utils import is_compiled_module

from lora_flx import *

logger = get_logger(__name__)




def import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path: str, revision: str, subfolder: str = "text_encoder"
):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder=subfolder, revision=revision
    )
    model_class = text_encoder_config.architectures[0]
    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "T5EncoderModel":
        from transformers import T5EncoderModel

        return T5EncoderModel
    else:
        raise ValueError(f"{model_class} is not supported.")


        
def load_text_encoders(args, class_one, class_two):
    text_encoder_one = class_one.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", device_map="auto"
    )
    text_encoder_two = class_two.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder_2", device_map="auto"
    )
    return text_encoder_one, text_encoder_two
        
        

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--instance_data_dir",
        type=str,
        default=None,
        required=True,
        help="A folder containing the training data of instance images.",
    )

    parser.add_argument(
        "--instance_prompt",
        type=str,
        default=None,
        help="The prompt with identifier specifying the instance",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop", action="store_true", help="Whether to center crop images before resizing to resolution"
    )
    parser.add_argument("--train_text_encoder", action="store_true", help="Whether to train the text encoder")
    parser.add_argument(
        "--train_batch_size", type=int, default=1, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--sample_batch_size", type=int, default=1, help="Batch size (per device) for sampling images."
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-6,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="no",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose"
            "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
            "and an Nvidia Ampere GPU."
        ),
    )

  
 
    parser.add_argument(
        "--Session_dir",
        type=str,
        default="",     
        help="Current session directory",
    )    

       
    parser.add_argument(
        "--dim",
        type=int,
        default=64,        
        help="LoRa dimension",
    )
    

    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=1,
        help="the FLUX.1 dev variant is a guidance distilled model",
    )    
    
    parser.add_argument(
        "--max_sequence_length",
        type=int,
        default=512,
        help="Maximum sequence length to use with with the T5 text encoder",
    )    
    
    parser.add_argument(
        "--saves",
        type=str,
        default="[]",
    )
    
    
    args = parser.parse_args()
    
    return args



class DreamBoothDataset(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images and the tokenizes prompts.
    """

    def __init__(
        self,
        instance_data_root,
        args,
        tokenizers,        
        text_encoders,
        size=512,
        center_crop=False,
    ):
        self.size = size
        self.tokenizers=tokenizers
        self.text_encoders=text_encoders
        self.center_crop = center_crop

        self.instance_data_root = Path(instance_data_root)
        if not self.instance_data_root.exists():
            raise ValueError("Instance images root doesn't exists.")

        self.instance_images_path = list(Path(instance_data_root).iterdir())
        self.num_instance_images = len(self.instance_images_path)
        self._length = self.num_instance_images
        
        self.image_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return self._length

    def __getitem__(self, index, args=parse_args()):
        example = {}
        path = self.instance_images_path[index % self.num_instance_images]
        instance_image = Image.open(path)
        if not instance_image.mode == "RGB":
            instance_image = instance_image.convert("RGB")

        filename = Path(path).stem

        pt=''.join([i for i in filename if not i.isdigit()])
        pt=pt.replace("_"," ")
        pt=pt.replace("(","")
        pt=pt.replace(")","")
        pt=pt.replace("-","")
        instance_prompt = pt
        #print('[1;32m'+instance_prompt)
        
        example["instance_images"] = self.image_transforms(instance_image)
        with torch.no_grad():
            example["instance_prompt_ids"], example["pooled_prompt_embeds"], example["text_ids"] = compute_text_embeddings(instance_prompt,  self.text_encoders,  self.tokenizers)
        return example

            
class PromptDataset(Dataset):
    "A simple dataset to prepare the prompts to generate class images on multiple GPUs."

    def __init__(self, prompt, num_samples):
        self.prompt = prompt
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        example = {}
        example["prompt"] = self.prompt
        example["index"] = index
        return example


def _encode_prompt_with_t5(
    text_encoder,
    tokenizer,
    max_sequence_length=512,
    prompt=None,
    num_images_per_prompt=1,
    device=None,
    text_input_ids=None,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    if tokenizer is not None:
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            return_length=False,
            return_overflowing_tokens=False,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
    else:
        if text_input_ids is None:
            raise ValueError("text_input_ids must be provided when the tokenizer is not specified")

    prompt_embeds = text_encoder(text_input_ids.to(device))[0]

    dtype = text_encoder.dtype
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

    _, seq_len, _ = prompt_embeds.shape

    # duplicate text embeddings and attention mask for each generation per prompt, using mps friendly method
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

    return prompt_embeds


def _encode_prompt_with_clip(
    text_encoder,
    tokenizer,
    prompt: str,
    device=None,
    text_input_ids=None,
    num_images_per_prompt: int = 1,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    if tokenizer is not None:
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_overflowing_tokens=False,
            return_length=False,
            return_tensors="pt",
        )

        text_input_ids = text_inputs.input_ids
    else:
        if text_input_ids is None:
            raise ValueError("text_input_ids must be provided when the tokenizer is not specified")

    prompt_embeds = text_encoder(text_input_ids.to(device), output_hidden_states=False)

    # Use pooled output of CLIPTextModel
    prompt_embeds = prompt_embeds.pooler_output
    prompt_embeds = prompt_embeds.to(dtype=text_encoder.dtype, device=device)

    # duplicate text embeddings for each generation per prompt, using mps friendly method
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, -1)

    return prompt_embeds



def encode_prompt(
    text_encoders,
    tokenizers,
    prompt: str,
    max_sequence_length=512,
    device=None,
    num_images_per_prompt: int = 1,
    text_input_ids_list=None,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)
    dtype = text_encoders[0].dtype

    pooled_prompt_embeds = _encode_prompt_with_clip(
        text_encoder=text_encoders[0],
        tokenizer=tokenizers[0],
        prompt=prompt,
        device=device if device is not None else text_encoders[0].device,
        num_images_per_prompt=num_images_per_prompt,
        text_input_ids=text_input_ids_list[0] if text_input_ids_list else None,
    )

    prompt_embeds = _encode_prompt_with_t5(
        text_encoder=text_encoders[1],
        tokenizer=tokenizers[1],
        max_sequence_length=max_sequence_length,
        prompt=prompt,
        num_images_per_prompt=num_images_per_prompt,
        device=device if device is not None else text_encoders[1].device,
        text_input_ids=text_input_ids_list[1] if text_input_ids_list else None,
    )

    text_ids = torch.zeros(batch_size, prompt_embeds.shape[1], 3).to(device=device, dtype=dtype) #3
    text_ids = text_ids.repeat(num_images_per_prompt, 1,1)

    return prompt_embeds, pooled_prompt_embeds, text_ids

    

def collate_fn(examples):

    input_ids = [example["instance_prompt_ids"] for example in examples]
    pixel_values = [example["instance_images"] for example in examples]        
    pooled_prompt_embeds = [example["pooled_prompt_embeds"] for example in examples]
    text_ids = [example["text_ids"] for example in examples]

    pixel_values = torch.stack(pixel_values)
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).half()

    input_ids = torch.cat(input_ids, dim=0)

    batch = {
        "input_ids": input_ids,
        "pixel_values": pixel_values,
        "pooled_prompt_embeds": pooled_prompt_embeds,
        "text_ids":text_ids
    }

    return batch



def compute_text_embeddings(prompt, text_encoders, tokenizers):
    with torch.no_grad():
        prompt_embeds, pooled_prompt_embeds, text_ids = encode_prompt(
            text_encoders, tokenizers, prompt, 512
        )
        prompt_embeds = prompt_embeds.to('cuda')
        pooled_prompt_embeds = pooled_prompt_embeds.to('cuda')
        text_ids = text_ids.to('cuda')
    return prompt_embeds, pooled_prompt_embeds, text_ids


    
class LatentsDataset(Dataset):
    def __init__(self, latents_cache, text_encoder_cache, pooled_cache, text_ids):
        self.latents_cache = latents_cache
        self.text_encoder_cache = text_encoder_cache
        self.pooled_cache = pooled_cache
        self.text_ids = text_ids

    def __len__(self):
        return len(self.latents_cache)

    def __getitem__(self, index):
        return self.latents_cache[index], self.text_encoder_cache[index], self.pooled_cache[index], self.text_ids[index]
    
    

def main():
    args = parse_args()
    logging_dir = Path(args.output_dir, args.logging_dir)
          
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with="tensorboard",
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs],
    )
  
        
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
            
    
    # Load scheduler and models

    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )

    noise_scheduler_copy = copy.deepcopy(noise_scheduler)    
    

    weight_dtype = torch.float32
    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif args.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        
        
    model_path = os.path.join(args.Session_dir, os.path.basename(args.Session_dir) + ".safetensors")


    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model          
    
    
    optimizer_class = bnb.optim.AdamW8bit 
    
    
    # Load the tokenizers
    tokenizer_one = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=None
    )
    tokenizer_two = T5TokenizerFast.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer_2",
        revision=None        
    )


    # import correct text encoder classes
    text_encoder_cls_one = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, revision=None
    )
    text_encoder_cls_two = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, subfolder="text_encoder_2", revision=None
    )


    text_encoder_one, text_encoder_two = load_text_encoders(args, 
        text_encoder_cls_one, text_encoder_cls_two
    )
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae", revision=None
    )


    vae.requires_grad_(False)
    text_encoder_one.requires_grad_(False)
    text_encoder_two.requires_grad_(False)
    vae.eval()
    text_encoder_one.eval()
    text_encoder_two.eval()


    text_encoder_one.to("cuda", dtype=torch.bfloat16)
    text_encoder_two.to("cuda", dtype=torch.bfloat16)
    vae.to("cuda", dtype=torch.bfloat16) 

    tokenizers = [tokenizer_one, tokenizer_two]
    text_encoders = [text_encoder_one, text_encoder_two]


    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )


    train_dataset = DreamBoothDataset(
        instance_data_root=args.instance_data_dir,
        tokenizers=tokenizers,
        text_encoders=text_encoders,
        size=args.resolution,
        center_crop=args.center_crop,
        args=args
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=lambda examples: collate_fn(examples),
    )


    # Scheduler and math around the number of training steps.

    latents_cache= []
    text_encoder_cache= []
    pooled_cache= []
    text_ids= []
    for batch in train_dataloader:
        with torch.no_grad():

            batch["input_ids"] = batch["input_ids"].to("cuda", non_blocking=True)
            batch["pooled_prompt_embeds"] = batch["pooled_prompt_embeds"]
            batch["text_ids"]=batch["text_ids"][0][0]

            batch["pixel_values"]=((vae.encode(batch["pixel_values"].to("cuda", dtype=torch.bfloat16, non_blocking=True)).latent_dist.sample() - vae.config.shift_factor) * vae.config.scaling_factor)

            latents_cache.append(batch["pixel_values"])
            text_encoder_cache.append(batch["input_ids"])
            pooled_cache.append(batch["pooled_prompt_embeds"])
            text_ids.append(batch["text_ids"])


    train_dataset = LatentsDataset(latents_cache, text_encoder_cache, pooled_cache, text_ids)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=1, collate_fn=lambda x: x, shuffle=True)
    vae_scale_factor = 2 ** (len(vae.config.block_out_channels))

    del vae, tokenizers, text_encoder_one, text_encoder_two
    gc.collect()
    torch.cuda.empty_cache()
    

    
    transformer = FluxTransformer2DModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="transformer",  device_map="auto", torch_dtype=torch.bfloat16
    )
    
    
    transformer.requires_grad_(False)
    network = create_network(1, args.dim, args.dim, transformer)


    network.apply_to(transformer, True)
   
    network.requires_grad_(True)
    
    trainable_params=network.parameters()


    if not args.use_8bit_adam:

        import prodigyopt
        optimizer_class = prodigyopt.Prodigy

        if args.learning_rate <= 0.1:
            logger.warning(
                "Learning rate is too low. When using prodigy, it's generally better to set learning rate around 1.0"
            )

        optimizer = optimizer_class(
            trainable_params,
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            beta3=None,
            weight_decay=0,
            eps=args.adam_epsilon,
            decouple=True,
            use_bias_correction=True,
            safeguard_warmup=True
        )
    
    else:
        
        optimizer = optimizer_class(
            trainable_params,
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=0,
            eps=1e-14
        )

    
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True


    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps= args.max_train_steps * args.gradient_accumulation_steps,
    )    
    
    network, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
       network, optimizer, train_dataloader, lr_scheduler
    )

    if args.gradient_checkpointing:
        transformer.train()
        transformer.enable_gradient_checkpointing()
 
        
    transformer.to("cuda", dtype=weight_dtype)
     


    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("dreambooth", config=vars(args))

    def bar(prg):
       br='|'+'█' * prg + ' ' * (25-prg)+'|'
       return br
    
    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    text_enc_context = nullcontext() if args.train_text_encoder else torch.no_grad()
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    global_step = 0
    

    def append_dims(x, target_dims):
        """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
        dims_to_append = target_dims - x.ndim
        if dims_to_append < 0:
            raise ValueError(
                f"input has {x.ndim} dims but target_dims is {target_dims}, which is less"
            )
        return x[(...,) + (None,) * dims_to_append]
    
    
    def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
        sigmas = noise_scheduler_copy.sigmas.to(device="cuda", dtype=dtype)
        schedule_timesteps = noise_scheduler_copy.timesteps.to("cuda")
        timesteps = timesteps.to("cuda")
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma



    for epoch in range(args.num_train_epochs):
                   
        for step, batch in enumerate(train_dataloader):
            network.train()
            with accelerator.accumulate(network):
                
                with torch.no_grad():
                    model_input = batch[0][0]
                
                latent_image_ids = FluxPipeline._prepare_latent_image_ids(
                    model_input.shape[0],
                    model_input.shape[2],
                    model_input.shape[3],
                    "cuda",
                    weight_dtype,
                )                
                
                noise = torch.randn_like(model_input)
                bsz = model_input.shape[0]

               
                u = compute_density_for_timestep_sampling(
                    weighting_scheme='logit_normal',
                    batch_size=bsz,
                    logit_mean=0.0,
                    logit_std=1.0,
                    mode_scale=1.29,
                )                

                indices = (u * noise_scheduler_copy.config.num_train_timesteps).long()

                timesteps = noise_scheduler_copy.timesteps[indices].to(device=model_input.device)

                sigmas = get_sigmas(timesteps, n_dim=model_input.ndim, dtype=model_input.dtype)
                noisy_model_input = (1.0 - sigmas) * model_input + sigmas * noise

                packed_noisy_model_input = FluxPipeline._pack_latents(
                    noisy_model_input,
                    batch_size=model_input.shape[0],
                    num_channels_latents=model_input.shape[1],
                    height=model_input.shape[2],
                    width=model_input.shape[3],
                )                
                
                if transformer.config.guidance_embeds:
                    guidance = torch.tensor([args.guidance_scale], device="cuda")
                    guidance = guidance.expand(model_input.shape[0])
                else:
                    guidance = None          


                # Predict the noise residual
                with accelerator.autocast():
                    model_pred = transformer(
                        hidden_states=packed_noisy_model_input,
                        timestep=timesteps/1000,
                        guidance=guidance,
                        encoder_hidden_states=batch[0][1],
                        pooled_projections=batch[0][2][0],
                        txt_ids=batch[0][3],
                        img_ids=latent_image_ids,
                        return_dict=False,
                    )[0]


                    model_pred = FluxPipeline._unpack_latents(
                        model_pred,
                        height=int(model_input.shape[2] * vae_scale_factor / 2),
                        width=int(model_input.shape[3] * vae_scale_factor / 2),
                        vae_scale_factor=vae_scale_factor,
                    )

                weighting = compute_loss_weighting_for_sd3(weighting_scheme='logit_normal', sigmas=sigmas)

                # Get the target for loss depending on the prediction type
                target = noise - model_input

                loss = torch.mean(
                    (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1),
                    1,
                )
                loss = loss.mean()

                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=False)


            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                
            fll=round((global_step*100)/args.max_train_steps)
            fll=round(fll/4)
            pr=bar(fll)
            

            Epochs="[0;32mEpoch" if step==0 else "Epoch"
                       
            logs = {Epochs: str(epoch+1)+'('+str(step+1)+'/'+str(len(train_dataloader))+')[0m', "loss": loss.item(), "lr": lr_scheduler.get_last_lr()[0]}
            
            progress_bar.set_postfix(logs)
            progress_bar.set_description_str("Progress")
            accelerator.log(logs, step=global_step)
            
            
            if epoch+1 in json.loads(args.saves) and step+1==len(train_dataloader):
                unwrapped=accelerator.unwrap_model(network)
                intrmdr= os.path.join(args.Session_dir, os.path.basename(args.Session_dir)+'_epoch_'+str(epoch+1)+'.safetensors')
                unwrapped.save_weights(intrmdr, torch.bfloat16, None)                
                print("[1;33mIntermediary LoRAs saved[0m")


            if global_step >= args.max_train_steps:
                break

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
         network = accelerator.unwrap_model(network)
    accelerator.end_training()
    
    network.save_weights(model_path, torch.bfloat16, None)
    
if __name__ == "__main__":
    main()
