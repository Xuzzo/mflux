import sys 
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from flux_1.config.config import Config
from flux_1.flux import Flux1Lora, Flux1
from flux_1.post_processing.image_util import ImageUtil

flux_lora = Flux1Lora(repo_id="black-forest-labs/FLUX.1-dev", lora_repo_id="XLabs-AI/flux-lora-collection", lora_id="anime_lora")
# flux = Flux1(repo_id="black-forest-labs/FLUX.1-dev")

image = flux_lora.generate_image(
    seed=3,
    prompt="A cute corgi lives in a house made out of sushi, anime",
    config=Config(
            num_inference_steps=10,
            height=256,
            width=512,
        )
)
ImageUtil.save_image(image, "image.png")