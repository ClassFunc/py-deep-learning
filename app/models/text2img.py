from pydantic import BaseModel

class Text2ImageModels(BaseModel):
    prompt: str
    negative_prompt: str = '""'
    seed: int = 0
    count: int = 1
    num_inference_steps: int = 50
    guidance_scale: float = 7.5
    eta: float = 0.0
    height: int = 512
    width: int = 512
