"""
Face search and embedding statistics API routes for the Sentinel backend.

Router prefix: ``/api``

This module provides endpoints for querying the face recognition index:

* **POST /api/search**           -- Upload a face image and search the FAISS
  index for the closest matching enrolled persons.
* **GET  /api/embeddings/stats** -- Retrieve statistics about the current
  state of the FAISS vector index (total embeddings, dimensionality, etc.).
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile

from app.api.deps import get_recognition_service, get_current_user
from app.services.recognition_service import RecognitionService


router = APIRouter(prefix="/api", tags=["recognition"], dependencies=[Depends(get_current_user)])


@router.post("/search")
async def search_face(
    image: UploadFile = File(...),
    top_k: int = Query(default=5, ge=1, le=20),
    recognition_service: RecognitionService = Depends(get_recognition_service),
):
    """Search the enrolled person index using a face image.

    **HTTP Method:** POST
    **Path:** /api/search
    **Auth required:** Yes (Bearer token)
    **Content-Type:** multipart/form-data

    The uploaded image is processed to detect and extract a face embedding,
    which is then compared against all enrolled persons in the FAISS index.
    The ``top_k`` closest matches are returned, ranked by similarity.

    Args:
        image: A face image file (JPEG/PNG) to use as the search query.
        top_k: Number of closest matches to return (1--20, default 5).

    Returns:
        dict: Search results containing a list of matched persons with
            similarity scores and metadata.

    Raises:
        HTTPException 400: If no face is detected in the uploaded image,
            or the image cannot be decoded.
    """
    try:
        # Read the raw image bytes and delegate to the recognition service,
        # which handles face detection, embedding extraction, and FAISS search.
        result = recognition_service.search_image(await image.read(), top_k=top_k)
    except ValueError as exc:
        # Raised when the image is invalid or contains no detectable face
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return result.model_dump()


# Plain ``def`` so FastAPI runs it in the threadpool: the synchronous database
# work must not stall the event loop that sends the camera frame streams.
@router.get("/embeddings/stats")
def get_embedding_stats(recognition_service: RecognitionService = Depends(get_recognition_service)):
    """Return statistics about the FAISS face embedding index.

    **HTTP Method:** GET
    **Path:** /api/embeddings/stats
    **Auth required:** Yes (Bearer token)

    Provides a summary of the current index state, including the total
    number of stored embeddings, the embedding dimensionality, and the
    number of enrolled persons.

    Returns:
        dict: Index statistics serialised from the service model.
    """
    return recognition_service.get_embedding_stats().model_dump()
