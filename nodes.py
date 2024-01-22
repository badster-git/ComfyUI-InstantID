import copy
import os
import torch
from safetensors.torch import load_file
from torchvision.transforms import ToTensor
from comfy.model_management import get_torch_device
from .pipeline_stable_diffusion_xl_instantid import StableDiffusionXLInstantIDPipeline, draw_kps
from .infer import resize_img
import folder_paths
import cv2
import numpy as np
from PIL import Image

from diffusers.utils import load_image
from diffusers.models import ControlNetModel
from diffusers import StableDiffusionXLPipeline

from insightface.app import FaceAnalysis

CUSTOM_NODE_CWD = os.path.dirname(os.path.realpath(__file__))

class InstantIDSampler:
    def __init__(self):
        self.torch_device = get_torch_device()
        self.tmp_dir = folder_paths.get_temp_directory()
        self.face_app = None
        self.controlnet = None
        self.previous_ckpt = None
        self.pipe = None
        
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ckpt_name": (folder_paths.get_filename_list("checkpoints"), ),
                "positive": ("STRING", {"multiline": True}),
                "negative": ("STRING", {"multiline": True}),
                "steps": ("INT", {"default": 30, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 5, "min": 0.0, "max": 100.0}),
                "strength": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "image" : ("IMAGE", )
            }
        }

    RETURN_TYPES = ("IMAGE",)

    FUNCTION = "sample"

    CATEGORY = "Instant ID"


    def sample(self, ckpt_name, positive, negative, steps, cfg, strength, seed, image):

        print("Instant ID Current Working Dir:", CUSTOM_NODE_CWD)
    
        if self.face_app is None:
            app = FaceAnalysis(name='antelopev2', root=CUSTOM_NODE_CWD, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
            app.prepare(ctx_id=0, det_size=(640, 640))
            self.face_app = app
            print("Instant ID Insightface App Loaded.")

        # Path to InstantID models
        face_adapter = os.path.join(CUSTOM_NODE_CWD, f'./checkpoints/ip-adapter.bin')
        controlnet_path = os.path.join(CUSTOM_NODE_CWD, f'./checkpoints/ControlNetModel')

        # Load pipeline
        if self.controlnet is None:
            self.controlnet = ControlNetModel.from_pretrained(controlnet_path, torch_dtype=torch.float16, use_safetensors=True)
            print("Instant ID Controlnet Loaded.")

        # prepare ckpt for diffusers

        if self.previous_ckpt != ckpt_name:
            ckpt_cache_path = os.path.join(self.tmp_dir, ckpt_name)
            StableDiffusionXLPipeline.from_single_file(
                pretrained_model_link_or_path=folder_paths.get_full_path("checkpoints", ckpt_name),
                torch_dtype=torch.float16,
                cache_dir=self.tmp_dir,
            ).save_pretrained(ckpt_cache_path, safe_serialization=True)
            
            self.previous_ckpt = ckpt_name
            self.pipe = StableDiffusionXLInstantIDPipeline.from_pretrained(
                ckpt_cache_path,
                controlnet=self.controlnet,
                torch_dtype=torch.float16,
            )
            self.pipe.cuda()
            self.pipe.load_ip_adapter_instantid(face_adapter)
            print("Instant ID Ckpt Reloaded.")


        prompt = positive
        n_prompt = negative

        face_image = Image.fromarray(np.clip(255. * image[0].cpu().numpy(), 0, 255).astype(np.uint8))
        face_image = resize_img(face_image)

        face_info = self.face_app.get(cv2.cvtColor(np.array(face_image), cv2.COLOR_RGB2BGR))
        if len(face_info) < 1:
            raise ValueError("Cannot find any face.")
        face_info = sorted(face_info, key=lambda x:(x['bbox'][2]-x['bbox'][0])*x['bbox'][3]-x['bbox'][1])[-1] # only use the maximum face
        face_emb = face_info['embedding']
        face_kps = draw_kps(face_image, face_info['kps'])
        print("Instant ID Face Info Updated.")


        self.pipe.set_ip_adapter_scale(strength)

        g_cpu = torch.Generator()
        g_cpu.manual_seed(seed)
        result = self.pipe(
            prompt=prompt,
            negative_prompt=n_prompt,
            image_embeds=face_emb,
            image=face_kps,
            controlnet_conditioning_scale=strength,
            num_inference_steps=steps,
            guidance_scale=cfg,
            generator=g_cpu
        ).images

        def convert_images_to_tensors(images: list[Image.Image]):
            return torch.stack([np.transpose(ToTensor()(image), (1, 2, 0)) for image in images])

        return (convert_images_to_tensors(result),)
