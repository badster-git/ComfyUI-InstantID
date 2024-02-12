# ComfyUI InstantID MultiControlNet

This is a custom node for [Instant ID](https://github.com/InstantID/InstantID). Added functionality for modifying scale values for 3 different ControlNets; Pose, Canny, & Depth. Similar to the [InstantID Space](https://huggingface.co/spaces/InstantX/InstantID).

I mainly made this to learn a little bit more about custom nodes within ComfyUI but decided I'd share it. 

It's important to note that this workflow uses a lot of VRAM, keep that in mind before attempting to use this. Typically I see usage of around 18-19 GB. 

## Install 

Just as other custom nodes:
```
cd ComfyUI/custom_nodes/
git clone https://github.com/badster-git/ComfyUI-InstantID-MultiControlNet.git
pip install -r requirements.txt
```

## Download Models

You can directly download the model from [Huggingface](https://huggingface.co/InstantX/InstantID).
You also can download the model in python script:

```python
from huggingface_hub import hf_hub_download
hf_hub_download(repo_id="InstantX/InstantID", filename="ControlNetModel/config.json", local_dir="./checkpoints")
hf_hub_download(repo_id="InstantX/InstantID", filename="ControlNetModel/diffusion_pytorch_model.safetensors", local_dir="./checkpoints")
hf_hub_download(repo_id="InstantX/InstantID", filename="ip-adapter.bin", local_dir="./checkpoints")
```

If you cannot access to Huggingface, you can use [hf-mirror](https://hf-mirror.com/) to download models.
```python
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download --resume-download InstantX/InstantID --local-dir checkpoints
```

For face encoder, you need to manually download via this [URL](https://github.com/deepinsight/insightface/issues/1896#issuecomment-1023867304) to `models/antelopev2` as the default link is invalid. Once you have prepared all models, the folder tree should be like:

```
  .
  ├── models
  ├── checkpoints
  ├── ip_adapter
  ├── pipeline_stable_diffusion_xl_instantid.py
  └── README.md
```

## Usage

Choose an SDXL base ckpt. You can also try SDXL Turbo with minimal steps which is very efficient for fast testing.

Model used in the examples can be found [here](https://huggingface.co/SG161222/RealVisXL_V3.0_Turbo).

First time loading usually takes more than 60s, but the node will try its best to cache models.
 

<img width="1292" alt="Workflow Example 1" src="./assets/Screenshot 2024-02-12 013622.png">

<img width="1248" alt="Workflow Example 2" src="./assets/Screenshot 2024-02-12 020559.png">

<img width="1248" alt="Workflow Example 2" src="./assets/Screenshot 2024-02-12 024826.png">

<img width="1248" alt="Workflow Example 2" src="./assets/Screenshot 2024-02-12 030305.png">

# Original Project 
<a href='https://instantid.github.io/'><img src='https://img.shields.io/badge/Project-Page-green'></a> 
<a href='https://arxiv.org/abs/2401.07519'><img src='https://img.shields.io/badge/Technique-Report-red'></a> 
<a href='https://huggingface.co/papers/2401.07519'><img src='https://img.shields.io/static/v1?label=Paper&message=Huggingface&color=orange'></a> 
<a href='https://huggingface.co/spaces/InstantX/InstantID'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue'></a> 

**InstantID : Zero-shot Identity-Preserving Generation in Seconds**

InstantID is a new state-of-the-art tuning-free method to achieve ID-Preserving generation with only single image, supporting various downstream tasks.

<img src='assets/applications.png'>

## Release
- [2024/1/22] 🔥 We release the [pre-trained checkpoints](https://huggingface.co/InstantX/InstantID), [inference code](https://github.com/InstantID/InstantID/blob/main/infer.py) and [gradio demo](https://huggingface.co/spaces/InstantX/InstantID)!
- [2024/1/15] 🔥 We release the technical report.
- [2023/12/11] 🔥 We launch the project page.

## Demos

### Stylized Synthesis

<p align="center">
  <img src="assets/0.png">
</p>

### Comparison with Previous Works

<p align="center">
  <img src="assets/compare-a.png">
</p>

Comparison with existing tuning-free state-of-the-art techniques. InstantID achieves better fidelity and retain good text editability (faces and styles blend better).

<p align="center">
  <img src="assets/compare-c.png">
</p>

Comparison with pre-trained character LoRAs. We don't need multiple images and still can achieve competitive results as LoRAs without any training.

<p align="center">
  <img src="assets/compare-b.png">
</p>

Comparison with InsightFace Swapper (also known as ROOP or Refactor). However, in non-realistic style, our work is more flexible on the integration of face and background.

## Usage Tips
- For higher similarity, increase the weight of controlnet_conditioning_scale (IdentityNet) and ip_adapter_scale (Adapter).
- For over-saturation, decrease the ip_adapter_scale. If not work, decrease controlnet_conditioning_scale.
- For higher text control ability, decrease ip_adapter_scale.
- For specific styles, choose corresponding base model makes differences.

## Acknowledgements
- Our work is highly inspired by [IP-Adapter](https://github.com/tencent-ailab/IP-Adapter) and [ControlNet](https://github.com/lllyasviel/ControlNet). Thanks for their great works!
- Thanks to the HuggingFace team for their generous GPU support!

## Disclaimer
This project is released under [Apache License](https://github.com/InstantID/InstantID?tab=Apache-2.0-1-ov-file#readme) and aims to positively impact the field of AI-driven image generation. Users are granted the freedom to create images using this tool, but they are obligated to comply with local laws and utilize it responsibly. The developers will not assume any responsibility for potential misuse by users.

## Cite
If you find InstantID useful for your research and applications, please cite us using this BibTeX:

```bibtex
@article{wang2024instantid,
  title={InstantID: Zero-shot Identity-Preserving Generation in Seconds},
  author={Wang, Qixun and Bai, Xu and Wang, Haofan and Qin, Zekui and Chen, Anthony},
  journal={arXiv preprint arXiv:2401.07519},
  year={2024}
}
