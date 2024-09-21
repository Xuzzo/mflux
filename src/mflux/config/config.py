import logging

import mlx.core as mx

log = logging.getLogger(__name__)


class Config:
    precision: mx.Dtype = mx.bfloat16

    def __init__(
            self,
            num_inference_steps: int = 4,
            width: int = 1024,
            height: int = 1024,
            guidance: float = 4.0,
    ):
        if width % 16 != 0 or height % 16 != 0:
            log.warning("Width and height should be multiples of 16. Rounding down.")
        self.width = 16 * (width // 16)
        self.height = 16 * (height // 16)
        self.num_inference_steps = num_inference_steps
        self.num_denoising_steps = num_inference_steps
        self.inference_steps = list(range(num_inference_steps))
        self.guidance = guidance


class ConfigImg2Img(Config):
    def __init__(
            self,
            num_inference_steps: int = 4,
            width: int = 1024,
            height: int = 1024,
            guidance: float = 4.0,
            controlnet_strength: float = 1.0,
            img2img_strength: float = 0.7,
    ):
        super().__init__(num_inference_steps, width, height, guidance)
        self.controlnet_strength = controlnet_strength

        self.img2img_strength = img2img_strength
        if img2img_strength <= 0.0 or img2img_strength >= 1.0:
            raise ValueError("Strength should be a float between 0 and 1.")

        self.num_total_denoising_steps = int(self.num_inference_steps / (1-img2img_strength))
        self.init_timestep = int(self.num_total_denoising_steps - self.num_inference_steps)
        self.inference_steps = list(range(self.num_total_denoising_steps))[self.init_timestep:]
