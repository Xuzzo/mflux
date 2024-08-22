import sys 
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))
from flux_1.weights.weight_handler import LoraWeightHandler


LoraWeightHandler("XLabs-AI/flux-lora-collection", "realism_lora")