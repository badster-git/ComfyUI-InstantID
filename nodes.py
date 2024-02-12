import copy
import os
import torch
from safetensors.torch import load_file
from torchvision.transforms import ToTensor
from comfy.model_management import get_torch_device
import comfy.controlnet
from .pipeline_stable_diffusion_xl_instantid import (
    StableDiffusionXLInstantIDPipeline,
    draw_kps,
)
from .infer import resize_img
import folder_paths
import cv2
import numpy as np
from PIL import Image

import diffusers
from diffusers.utils import load_image
from diffusers.models import ControlNetModel
from diffusers import StableDiffusionXLPipeline
from huggingface_hub import hf_hub_download

from diffusers.pipelines.controlnet.multicontrolnet import MultiControlNetModel
from insightface.app import FaceAnalysis

from .controlnet_util import openpose, get_depth_map, get_canny_image

CUSTOM_NODE_CWD = os.path.dirname(os.path.realpath(__file__))

SCHEDULER_KWARGS = hf_hub_download(
    repo_id="SG161222/RealVisXL_V3.0_Turbo",
    subfolder="scheduler",
    filename="scheduler_config.json",
)
SCHEDULER = diffusers.EulerDiscreteScheduler.from_config(SCHEDULER_KWARGS)


class InstantIDSampler:
    def __init__(self):
        self.torch_device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tmp_dir = folder_paths.get_temp_directory()
        self.face_app = None
        self.controlnet = None
        self.previous_ckpt = None
        self.pipe = None
        self.scheduler = SCHEDULER 

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ckpt_name": (folder_paths.get_filename_list("checkpoints"),),
                "controlnet": ("MODEL",),
                "positive": ("STRING", {"multiline": True}),
                "negative": ("STRING", {"multiline": True}),
                "steps": ("INT", {"default": 25, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 2.5, "min": 0.0, "max": 100.0}),
                "ip_adapter_strength": (
                    "FLOAT",
                    {"default": 0.8, "min": 0.0, "max": 1.5},
                ),
                "pose_strength": ("FLOAT", {"default": 0.4, "min": 0.0, "max": 1.5}),
                "canny_strength": ("FLOAT", {"default": 0.4, "min": 0.0, "max": 1.5}),
                "depth_strength": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.5}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "image": ("IMAGE",),
                "enhance_face_region": (
                    "BOOLEAN",
                    {"default": False, "label_on": "on", "label_off": "off"},
                ),
                # "scheduler": (
                #     [
                #         "DEISMultistepScheduler",
                #         "HeunDiscreteScheduler",
                #         "EulerDiscreteScheduler",
                #         "DPMSolverMultistepScheduler",
                #         "DPMSolverMultistepScheduler-Karras",
                #         "DPMSolverMultistepScheduler-Karras-SDE",
                #     ],
                #     {"default": "EulerDiscreteScheduler"},
                # ),
            },
            "optional": {
                "pose_image_optional": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE",)

    FUNCTION = "sample"

    CATEGORY = "Instant ID"

    def sample(
        self,
        ckpt_name,
        controlnet,
        positive,
        negative,
        steps,
        cfg,
        ip_adapter_strength,
        pose_strength,
        canny_strength,
        depth_strength,
        seed,
        image,
        enhance_face_region,
        # scheduler,
        pose_image_optional=None,
    ):

        # if self.scheduler is None:
        #     self.scheduler = scheduler

        if self.face_app is None:
            app = FaceAnalysis(
                name="antelopev2",
                root=CUSTOM_NODE_CWD,
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            )
            app.prepare(ctx_id=0, det_size=(640, 640))
            self.face_app = app
            print("Instant ID Insightface App Loaded.")

        # Path to InstantID models
        face_adapter = os.path.join(CUSTOM_NODE_CWD, f"./checkpoints/ip-adapter.bin")

        # Load pipeline
        if self.controlnet is None:
            self.controlnet = controlnet

        # prepare ckpt for diffusers
        if self.previous_ckpt != ckpt_name:
            ckpt_cache_path = os.path.join(self.tmp_dir, ckpt_name)
            StableDiffusionXLPipeline.from_single_file(
                pretrained_model_link_or_path=folder_paths.get_full_path(
                    "checkpoints", ckpt_name
                ),
                torch_dtype=torch.float16,
                cache_dir=self.tmp_dir,
            ).save_pretrained(ckpt_cache_path, safe_serialization=True)

            self.previous_ckpt = ckpt_name
            self.pipe = StableDiffusionXLInstantIDPipeline.from_pretrained(
                ckpt_cache_path,
                controlnet=self.controlnet,
                torch_dtype=torch.float16,
                scheduler=self.scheduler,
            )

            # self.pipe.scheduler = diffusers.EulerDiscreteScheduler.from_config(
            #     self.pipe.scheduler.config
            # )
            self.pipe.cuda()
            self.pipe.load_ip_adapter_instantid(face_adapter)

        prompt = positive
        n_prompt = negative

        # scheduler_class_name = scheduler.split("-")[0]

        add_kwargs = {}
        # if len(scheduler.split("-")) > 1:
        #     add_kwargs["use_karras_sigmas"] = True
        # if len(scheduler.split("-")) > 2:
        #     add_kwargs["algorithm_type"] = "sde-dpmsolver++"


        # self.scheduler = getattr(diffusers, scheduler_class_name)
        self.pipe.scheduler = self.scheduler.from_config(
            self.pipe.scheduler.config, **add_kwargs
        )


        face_image = Image.fromarray(
            np.clip(255.0 * image[0].cpu().numpy(), 0, 255).astype(np.uint8)
        )
        face_image = resize_img(face_image)

        face_info = self.face_app.get(
            cv2.cvtColor(np.array(face_image), cv2.COLOR_RGB2BGR)
        )
        if len(face_info) < 1:
            raise ValueError("Cannot find any face.")
        face_info = sorted(
            face_info,
            key=lambda x: (x["bbox"][2] - x["bbox"][0]) * x["bbox"][3] - x["bbox"][1],
        )[
            -1
        ]  # only use the maximum face

        face_emb = face_info["embedding"]
        face_kps = draw_kps(face_image, face_info["kps"])
        width, height = face_kps.size
        print("Instant ID Face Info Updated.")
        img_controlnet = face_image

        if pose_image_optional is not None:
            pose_image = Image.fromarray(
                np.clip(255.0 * pose_image_optional[0].cpu().numpy(), 0, 255).astype(
                    np.uint8
                )
            )
            pose_image = resize_img(pose_image, max_side=1024)
            img_controlnet = pose_image
            face_info = self.face_app.get(
                cv2.cvtColor(np.array(pose_image), cv2.COLOR_RGB2BGR)
            )
            if len(face_info) == 0:
                raise Exception(
                    f"Cannot find any face in the reference image! Please upload another person image"
                )

            face_info = face_info[-1]
            face_kps = draw_kps(pose_image, face_info["kps"])

            width, height = face_kps.size

            # load midas
            # midas = MidasDetector.from_pretrained("lllyasviel/Annotators")
            # use depth control
            # processed_image_midas = midas(pose_image)
            # processed_image_midas = processed_image_midas.resize(pose_image.size)
            # final_image.append(processed_image_midas)

        controlnet_map_fn = {
            "pose": openpose,
            "canny": get_canny_image,
            "depth": get_depth_map,
        }

        if enhance_face_region:
            control_mask = np.zeros([height, width, 3])
            x1, y1, x2, y2 = face_info["bbox"]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            control_mask[y1:y2, x1:x2] = 255
            control_mask = Image.fromarray(control_mask.astype(np.uint8))
        else:
            control_mask = None

        control_scales = [
            ip_adapter_strength,
            pose_strength,
            canny_strength,
            depth_strength,
        ]

        control_images = [face_kps] + [
            controlnet_map_fn[key](img_controlnet).resize((width, height))
            for key in list(controlnet_map_fn.keys())
        ]

        # g_cpu = torch.Generator()
        # g_cpu.manual_seed(seed)
        generator = torch.Generator(device=self.torch_device).manual_seed(seed)

        self.pipe.set_ip_adapter_scale(ip_adapter_strength)

        # result = self.pipe(
        #     prompt=prompt,
        #     negative_prompt=n_prompt,
        #     image_embeds=face_emb,
        #     # image=face_kps,
        #     image=control_images,
        #     control_mask=control_mask,
        #     controlnet_conditioning_scale=control_scales,
        #     num_inference_steps=steps,
        #     guidance_scale=cfg,
        #     generator=generator,
        #     return_dict=False
        # )
        result = self.pipe(
            prompt=prompt,
            negative_prompt=n_prompt,
            image_embeds=face_emb,
            # image=face_kps,
            image=control_images,
            control_mask=control_mask,
            controlnet_conditioning_scale=control_scales,
            num_inference_steps=steps,
            guidance_scale=cfg,
            generator=generator,
        ).images

        # # 检查输出类型并相应处理
        # if isinstance(result, tuple):
        #     # 当返回的是元组时，第一个元素是图像列表
        #     images_list = result[0]
        # else:
        #     # 如果返回的是 StableDiffusionXLPipelineOutput，需要从中提取图像
        #     images_list = result.images

        # # 转换图像为 torch.Tensor，并调整维度顺序为 NHWC
        # images_tensors = []
        # for img in images_list:
        #     # 将 PIL.Image 转换为 numpy.ndarray
        #     img_array = np.array(img)
        #     # 转换 numpy.ndarray 为 torch.Tensor
        #     img_tensor = torch.from_numpy(img_array).float() / 255.
        #     # 转换图像格式为 CHW (如果需要)
        #     if img_tensor.ndim == 3 and img_tensor.shape[-1] == 3:
        #         img_tensor = img_tensor.permute(2, 0, 1)
        #     # 添加批次维度并转换为 NHWC
        #     img_tensor = img_tensor.unsqueeze(0).permute(0, 2, 3, 1)
        #     images_tensors.append(img_tensor)

        # if len(images_tensors) > 1:
        #     output_image = torch.cat(images_tensors, dim=0)
        # else:
        #     output_image = images_tensors[0]

        # return (output_image,)

        def convert_images_to_tensors(images: list[Image.Image]):
            return torch.stack(
                [np.transpose(ToTensor()(image), (1, 2, 0)) for image in images]
            )

        return (convert_images_to_tensors(result),)
        # return result[0]


class InstantIDMultiControlNetNode:
    def __init__(self):
        self.controlnet_path = os.path.join(
            CUSTOM_NODE_CWD, f"./checkpoints/controlnet"
        )
        self.face_app = None
        self.controlnet = None
        self.controlnets = []

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                # "pose_path": (
                #     [
                #         x
                #         for x in folder_paths.get_filename_list("controlnet")
                #         if "pose" in x and "xl" in x
                #     ],
                # )
                # or None,
                # "canny_path": (
                #     [
                #         x
                #         for x in folder_paths.get_filename_list("controlnet")
                #         if "canny" in x and "xl" in x
                #     ],
                # )
                # or None,
                # "depth_path": (
                #     [
                #         x
                #         for x in folder_paths.get_filename_list("controlnet")
                #         if "depth" in x and "xl" in x
                #     ],
                # )
                # or None,
                "pose_path": (
                    "STRING",
                    {"default": "thibaud/controlnet-openpose-sdxl-1.0"},
                ),
                "canny_path": (
                    "STRING",
                    {"default": "diffusers/controlnet-canny-sdxl-1.0"},
                ),
                "depth_path": (
                    "STRING",
                    {"default": "diffusers/controlnet-depth-sdxl-1.0-small"},
                ),
            },
        }

    RETURN_TYPES = ("MODEL",)

    FUNCTION = "load_controlnets"

    CATEGORY = "Instant ID"

    def load_controlnets(s, pose_path, canny_path, depth_path):
        # Load pipeline
        if s.controlnet is None:
            s.controlnet = ControlNetModel.from_pretrained(
                s.controlnet_path, torch_dtype=torch.float16
            )
            s.controlnets.append(s.controlnet)


        for controlnet_path in [
            pose_path,
            canny_path,
            depth_path,
        ]:
            # controlnet_path = folder_paths.get_full_path("controlnet", controlnet_name)
            # controlnet = comfy.controlnet.load_controlnet(controlnet_path)
            controlnet = ControlNetModel.from_pretrained(
                controlnet_path, torch_dtype=torch.float16
            )
            s.controlnets.append(controlnet)
        s.controlnet = MultiControlNetModel(s.controlnets)

        return [s.controlnet]
