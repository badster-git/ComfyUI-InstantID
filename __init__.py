import os
from .nodes import InstantIDSampler, InstantIDMultiControlNetNode

NODE_CLASS_MAPPINGS = {
    "Instant ID Sampler": InstantIDSampler, 
    "Instant ID MultiControlNet": InstantIDMultiControlNetNode
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "Instant ID Sampler": "Instant ID Sampler For SDXL",
    "Instant ID MultiControlNet": "Instant ID MultiControlNet For SDXL",
}
