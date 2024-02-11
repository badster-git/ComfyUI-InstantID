import cv2
import torch
import numpy as np
from PIL import Image

from diffusers.utils import load_image
from diffusers.models import ControlNetModel
from diffusers import StableDiffusionXLPipeline

from insightface.app import FaceAnalysis
from .pipeline_stable_diffusion_xl_instantid import StableDiffusionXLInstantIDPipeline, draw_kps

def resize_img(input_image, max_side=1280, min_side=1024, size=None, 
               pad_to_max_side=False, mode=Image.BILINEAR, base_pixel_number=64):

    w, h = input_image.size
    if size is not None:
        w_resize_new, h_resize_new = size
    else:
        ratio = min_side / min(h, w)
        w, h = round(ratio*w), round(ratio*h)
        ratio = max_side / max(h, w)
        input_image = input_image.resize([round(ratio*w), round(ratio*h)], mode)
        w_resize_new = (round(ratio * w) // base_pixel_number) * base_pixel_number
        h_resize_new = (round(ratio * h) // base_pixel_number) * base_pixel_number
    input_image = input_image.resize([w_resize_new, h_resize_new], mode)

    if pad_to_max_side:
        res = np.ones([max_side, max_side, 3], dtype=np.uint8) * 255
        offset_x = (max_side - w_resize_new) // 2
        offset_y = (max_side - h_resize_new) // 2
        res[offset_y:offset_y+h_resize_new, offset_x:offset_x+w_resize_new] = np.array(input_image)
        input_image = Image.fromarray(res)
    return input_image


if __name__ == "__main__":
    pass

    # app = FaceAnalysis(name='antelopev2', root='./', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    # app.prepare(ctx_id=0, det_size=(640, 640))

    # # Path to InstantID models
    # face_adapter = f'./checkpoints/ip-adapter.bin'
    # controlnet_path = f'./checkpoints/ControlNetModel'

    # # Load pipeline
    # controlnet = ControlNetModel.from_pretrained(controlnet_path, torch_dtype=torch.float16, use_safetensors=True)

    # base_model_path = '/nieta_fs/ops/models/checkpoint/sd_xl_base_1.0.safetensors'
    # ckpt_cache_path =  './tmp/tmpckpt.safetensors'
    # StableDiffusionXLPipeline.from_single_file(
    #     pretrained_model_link_or_path=base_model_path,
    #     torch_dtype=torch.float16,
    #     cache_dir='./tmp',
    # ).save_pretrained(ckpt_cache_path, safe_serialization=True)
       

    # pipe = StableDiffusionXLInstantIDPipeline.from_pretrained(
    #     ckpt_cache_path,
    #     controlnet=controlnet,
    #     torch_dtype=torch.float16,
    # )
    # pipe.cuda()
    # pipe.load_ip_adapter_instantid(face_adapter)

    # prompt = "anime film screenshot of a man, masterpiece, best quality"
    # n_prompt = "(lowres, low quality, worst quality:1.2), (text:1.2), watermark, painting, drawing, illustration, glitch, deformed, mutated, cross-eyed, ugly, disfigured (lowres, low quality, worst quality:1.2), (text:1.2), watermark, painting, drawing, illustration, glitch,deformed, mutated, cross-eyed, ugly, disfigured"

    # face_image = load_image("./examples/yann-lecun_resize.jpg")
    # face_image = resize_img(face_image)

    # face_info = app.get(cv2.cvtColor(np.array(face_image), cv2.COLOR_RGB2BGR))
    # face_info = sorted(face_info, key=lambda x:(x['bbox'][2]-x['bbox'][0])*x['bbox'][3]-x['bbox'][1])[-1] # only use the maximum face
    # face_emb = face_info['embedding']
    # face_kps = draw_kps(face_image, face_info['kps'])

    # pipe.set_ip_adapter_scale(0.8)
    # image = pipe(
    #     prompt=prompt,
    #     negative_prompt=n_prompt,
    #     image_embeds=face_emb,
    #     image=face_kps,
    #     controlnet_conditioning_scale=0.8,
    #     num_inference_steps=30,
    #     guidance_scale=5,
    # ).images[0]

    # image.save('result.jpg')