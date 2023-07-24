from fastapi import APIRouter
from fastapi.responses import PlainTextResponse
from fastapi.responses import JSONResponse
from app.text2img import _generate

router = APIRouter(
    tags=["/v1/text2img"],
    responses={400: {'description': 'Bad Request'}}
)

@router.post("/v1/text2img")
async def stable_text2img():
          return JSONResponse(_generate('test new image'))
