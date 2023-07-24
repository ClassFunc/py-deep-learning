from fastapi import APIRouter

router = APIRouter(
    tags=["/v1/health"],
    responses={400: {'description': 'Bad Request'}}
)

@router.get("/v1/health")
async def perfom_healthcheck():
    return {"health":"application OK"}
