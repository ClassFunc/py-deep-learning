from fastapi import APIRouter, HTTPException
from fastapi.responses import PlainTextResponse
from fastapi.responses import JSONResponse
from app.text2img import _generate
from app.models.text2img import Text2ImageModels

router = APIRouter(
    tags=["/v1/text2img"],
    responses={400: {'description': 'Bad Request'}}
)

@router.post("/v1/text2img")
async def stable_text2img(request: Text2ImageModels):
          prompt = request.prompt or ""
          negative_prompt = request.negative_prompt
          seed = request.seed
          count = request.count
          num_inference_steps = request.num_inference_steps
          guidance_scale = request.guidance_scale
          eta = request.eta
          width = request.width
          height = request.height

          if len(prompt) == 0:
            raise HTTPException(status_code=404, detail="Prompt not found")
          return JSONResponse(_generate(prompt, negative_prompt, seed, count, num_inference_steps, guidance_scale, eta, width, height))
