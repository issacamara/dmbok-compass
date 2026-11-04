"""Admin-only download for the versioned corpus structure template."""

from pathlib import Path

from fastapi import APIRouter, Depends
from fastapi.responses import FileResponse

from app.contracts import UserProfile
from app.identity import get_admin_user


router = APIRouter(prefix="/api/admin/corpus-staging", tags=["corpus staging"])
TEMPLATE_PATH = Path(__file__).parents[1] / "templates" / "dmbok-structure-template-v1.json"


@router.get("/template", response_class=FileResponse)
def download_structure_template(
    _: UserProfile = Depends(get_admin_user),
) -> FileResponse:
    return FileResponse(
        TEMPLATE_PATH,
        media_type="application/json",
        filename="dmbok-structure-template-v1.json",
    )
