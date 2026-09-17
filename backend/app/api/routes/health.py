from fastapi import APIRouter, Request
import torch

router = APIRouter()

@router.get("/health", summary="Health check + model status")
async def health(request: Request):
    loader = getattr(request.app.state, "model_loader", None)
    models = loader.models if loader else None
    return {
        "status": "ok",
        "device": models.device if models else "unknown",
        "models": {
            "efficientnet": models.efficientnet is not None if models else False,
            "vit":          models.vit is not None if models else False,
            "xception":     models.xception is not None if models else False,
            "rawnet2":      models.rawnet2 is not None if models else False,
            "face_detector":models.face_detector is not None if models else False,
        },
        "cuda_available": torch.cuda.is_available(),
    }
