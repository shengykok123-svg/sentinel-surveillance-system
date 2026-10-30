from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field
from app.api.deps import get_current_user, get_auth_service
from app.services.settings_service import RecognitionSettings
from app.services.auth_service import AuthError

router = APIRouter(prefix="/api", tags=["settings"])

@router.get("/settings")
def get_settings(request: Request, user=Depends(get_current_user)):
    service = request.app.state.settings_service
    return {"recognition": service.current(), "defaults": service.defaults}

@router.put("/settings/recognition")
def save_recognition(payload: RecognitionSettings, request: Request, user=Depends(get_current_user)):
    if "admin" not in {role.lower() for role in user.roles}:
        raise HTTPException(403, "Administrator role is required")
    return request.app.state.settings_service.save(payload)

class PasswordChange(BaseModel):
    current_password: str = Field(min_length=1, max_length=1024)
    new_password: str = Field(min_length=8, max_length=128)

@router.post("/auth/change-password")
def change_password(payload: PasswordChange, user=Depends(get_current_user), service=Depends(get_auth_service)):
    try:
        service.change_password(user, payload.current_password, payload.new_password)
    except AuthError as exc:
        raise HTTPException(400, str(exc)) from exc
    return {"message": "Password updated successfully"}
