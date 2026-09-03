from __future__ import annotations

import hashlib
import importlib
import io
import sys
import threading
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from PIL import Image

from app.models.schemas import (
    AnalyzeTextRequest,
    AnalyzeTextResponse,
    DatasetMatch,
    DatasetStatusResponse,
    EncodeImageResponse,
)
from app.services import EmbeddingService, FAISSService, ImageProcessor, DatasetLoader

REPO_ROOT = Path(__file__).resolve().parents[3]
PROJECT_DIR = REPO_ROOT / "project"
_SYSTEM_LOCK = threading.Lock()

# Define semantic search models
class SearchQuery(BaseModel):
    query: str
    top_k: int = 5
    confidence_threshold: float = 0.7

class SearchResult(BaseModel):
    image_id: str
    image_path: str
    thumbnail_url: str
    similarity_score: float
    metadata: dict

class SemanticSearchResponse(BaseModel):
    query: str
    total_results: int
    results: list[SearchResult]
    processing_time_ms: float

class DatasetStatsResponse(BaseModel):
    total_images: int
    indexed_images: int
    embedding_dimension: int
    last_updated: str


def _ensure_project_path() -> None:
    project_path = str(PROJECT_DIR)
    if project_path not in sys.path:
        sys.path.insert(0, project_path)


@lru_cache(maxsize=1)
def _project_api() -> dict[str, Any]:
    _ensure_project_path()
    project_main = importlib.import_module("main")
    file_loader = importlib.import_module("utils.file_loader")
    return {
        "system_cls": getattr(project_main, "SatelliteImageIntelligenceSystem"),
        "dataset_error": getattr(file_loader, "DatasetNotFoundError"),
        "discover_images": getattr(file_loader, "discover_images"),
        "dataset_preview_paths": getattr(file_loader, "dataset_preview_paths"),
    }


@lru_cache(maxsize=1)
def get_system() -> Any:
    api = _project_api()
    system_cls = api["system_cls"]
    return system_cls(project_root=PROJECT_DIR, offline_processing=False)


def _stable_geo_from_text(value: str) -> tuple[float, float]:
    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()
    lat_raw = int(digest[:8], 16) / 0xFFFFFFFF
    lng_raw = int(digest[8:16], 16) / 0xFFFFFFFF
    lat = round((lat_raw * 170.0) - 85.0, 6)
    lng = round((lng_raw * 360.0) - 180.0, 6)
    return lat, lng


app = FastAPI(title="FusionX Bridge API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/dataset-status", response_model=DatasetStatusResponse)
def dataset_status() -> DatasetStatusResponse:
    api = _project_api()
    system = get_system()

    discover_images = api["discover_images"]
    preview_paths = api["dataset_preview_paths"]

    image_paths = discover_images(system.dataset_root)
    samples = preview_paths(image_paths, sample_size=5)
    indexed = bool(system.faiss_manager and system.faiss_manager.has_loaded_index())

    return DatasetStatusResponse(
        dataset_root=str(system.dataset_root),
        total_images=len(image_paths),
        sample_paths=samples,
        indexed=indexed,
    )


@app.post("/analyze-text", response_model=AnalyzeTextResponse)
def analyze_text(payload: AnalyzeTextRequest) -> AnalyzeTextResponse:
    query = payload.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    api = _project_api()
    dataset_error_type = api["dataset_error"]
    system = get_system()

    try:
        with _SYSTEM_LOCK:
            matches = system._search_local_with_text(query=query, top_k=payload.top_k)
            detected_topic = system.detect_topic(query)
            detected_task = system.detect_task(query)
            caption_targets = [path for path, _ in matches[:3]]
            caption_map = system._generate_captions(caption_targets) if caption_targets else {}
    except dataset_error_type as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Failed to process semantic query: {exc}") from exc

    response_matches: list[DatasetMatch] = []
    for idx, (path, similarity) in enumerate(matches, start=1):
        lat, lng = _stable_geo_from_text(str(path))
        clipped_score = max(min(float(similarity), 1.0), 0.0)
        response_matches.append(
            DatasetMatch(
                id=f"patch-{idx}",
                file_path=str(path),
                similarity=clipped_score,
                confidence=clipped_score,
                lat=lat,
                lng=lng,
                label=path.stem,
                caption=caption_map.get(str(path)),
            )
        )

    if response_matches:
        status_message = f"Found {len(response_matches)} local matches for '{query}'."
    else:
        status_message = "No local matches found in the indexed dataset."

    return AnalyzeTextResponse(
        query=query,
        detected_topic=detected_topic,
        detected_task=detected_task,
        dataset_root=str(system.dataset_root),
        total_matches=len(response_matches),
        matches=response_matches,
        status_message=status_message,
    )


@app.post("/encode-image", response_model=EncodeImageResponse)
async def encode_image(file: UploadFile = File(...)) -> EncodeImageResponse:
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="No image bytes provided.")

    try:
        with Image.open(io.BytesIO(content)) as img:
            image = img.convert("RGB")
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Invalid image file: {exc}") from exc

    system = get_system()
    try:
        with _SYSTEM_LOCK:
            embedding = system._get_clip_model().encode_pil_image(image)[0]
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Image embedding failed: {exc}") from exc

    vector = np.asarray(embedding, dtype=np.float32)
    digest = hashlib.sha1(vector.tobytes()).hexdigest()[:16].upper()
    code = f"VX-{digest}"
    preview = [round(float(v), 6) for v in vector[:12]]

    return EncodeImageResponse(
        unique_math_code=code,
        embedding_dim=int(vector.shape[0]),
        vector_preview=preview,
    )


# ============================================================================
# SEMANTIC SATELLITE IMAGERY SEARCH ENDPOINTS
# ============================================================================

# Global semantic search services (initialized lazily)
_embedding_service: Optional[EmbeddingService] = None
_faiss_service: Optional[FAISSService] = None
_image_processor: Optional[ImageProcessor] = None
_dataset_loader: Optional[DatasetLoader] = None


def _init_semantic_services():
    """Initialize semantic search services"""
    global _embedding_service, _faiss_service, _image_processor, _dataset_loader
    
    if _embedding_service is not None:
        return  # Already initialized
    
    try:
        dataset_path = Path("/Users/mahesharunaladi/Downloads/DeepQuery/project/dataset")
        index_path = Path("/Users/mahesharunaladi/Downloads/DeepQuery/FusionX-main/backend/app/indices")
        
        _embedding_service = EmbeddingService(device="cpu")
        _faiss_service = FAISSService(
            embedding_dim=_embedding_service.get_embedding_dim(),
            index_path=index_path
        )
        _image_processor = ImageProcessor()
        _dataset_loader = DatasetLoader(dataset_path)
        
        # Load or build index
        if _faiss_service.index_exists():
            _faiss_service.load_index()
        else:
            image_paths = _dataset_loader.get_all_images()
            if image_paths:
                _build_semantic_index(image_paths)
    
    except Exception as e:
        import logging
        logging.error(f"Failed to initialize semantic services: {e}")
        raise


def _build_semantic_index(image_paths: list[Path]):
    """Build FAISS index from satellite images"""
    import logging
    logger = logging.getLogger(__name__)
    
    logger.info(f"Building semantic index for {len(image_paths)} images...")
    
    embeddings_list = []
    valid_paths = []
    
    for idx, image_path in enumerate(image_paths):
        try:
            if (idx + 1) % max(1, len(image_paths) // 10) == 0:
                logger.info(f"Processing {idx + 1}/{len(image_paths)}...")
            
            img = _image_processor.load_image(image_path)
            if img is None:
                continue
            
            img = _image_processor.preprocess(img)
            embedding = _embedding_service.encode_image(img)
            embeddings_list.append(embedding)
            valid_paths.append(image_path)
            
        except Exception as e:
            logger.warning(f"Failed to process {image_path}: {e}")
            continue
    
    if embeddings_list:
        embeddings_array = np.vstack(embeddings_list).astype('float32')
        _faiss_service.add_embeddings(embeddings_array, valid_paths)
        _faiss_service.save_index()
        logger.info(f"✓ Semantic index built with {len(valid_paths)} images")
    else:
        logger.warning("No images were successfully processed")


@app.get("/semantic/stats", response_model=DatasetStatsResponse)
def semantic_stats():
    """Get semantic search system statistics"""
    try:
        _init_semantic_services()
        
        return DatasetStatsResponse(
            total_images=_dataset_loader.count_images(),
            indexed_images=_faiss_service.get_index_size(),
            embedding_dimension=_embedding_service.get_embedding_dim(),
            last_updated="2024-05-02"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/semantic/search", response_model=SemanticSearchResponse)
def semantic_search(query_data: SearchQuery):
    """
    Perform semantic search over satellite imagery
    
    Example queries:
    - "flooded roads"
    - "new construction"
    - "deforestation"
    - "urban development"
    - "water bodies"
    """
    try:
        _init_semantic_services()
        
        start_time = time.time()
        
        # Generate query embedding
        query_embedding = _embedding_service.encode_text(query_data.query)
        
        # Search FAISS index
        similarities, indices, result_paths = _faiss_service.search(
            query_embedding,
            k=query_data.top_k
        )
        
        # Prepare results
        results = []
        for score, path in zip(similarities, result_paths):
            if score >= query_data.confidence_threshold:
                image_id = path.stem
                
                results.append(SearchResult(
                    image_id=image_id,
                    image_path=str(path),
                    thumbnail_url=f"/api/thumbnail/{image_id}",
                    similarity_score=float(score),
                    metadata={
                        "filename": path.name,
                        "size_bytes": path.stat().st_size if path.exists() else 0
                    }
                ))
        
        processing_time_ms = (time.time() - start_time) * 1000
        
        return SemanticSearchResponse(
            query=query_data.query,
            total_results=len(results),
            results=results,
            processing_time_ms=processing_time_ms
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/semantic/reindex")
def semantic_reindex():
    """Rebuild semantic index from scratch"""
    try:
        _init_semantic_services()
        
        image_paths = _dataset_loader.refresh()
        _faiss_service.clear()
        _build_semantic_index(image_paths)
        
        return {
            "status": "success",
            "message": f"Semantic index rebuilt with {_faiss_service.get_index_size()} images"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/semantic/images")
def semantic_images(limit: int = Query(20, ge=1, le=100)):
    """List available images with their metadata"""
    try:
        _init_semantic_services()
        
        all_paths = _dataset_loader.get_all_images()[:limit]
        
        images = []
        for path in all_paths:
            images.append({
                "id": path.stem,
                "name": path.name,
                "size_bytes": path.stat().st_size if path.exists() else 0,
                "url": f"/semantic/thumbnail/{path.stem}"
            })
        
        return {
            "total": _dataset_loader.count_images(),
            "returned": len(images),
            "images": images
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/semantic/thumbnail/{image_id}")
async def semantic_thumbnail(image_id: str):
    """Get thumbnail for an image"""
    try:
        _init_semantic_services()
        
        all_paths = _dataset_loader.get_all_images()
        for path in all_paths:
            if path.stem == image_id:
                return FileResponse(path, media_type="image/jpeg")
        
        raise HTTPException(status_code=404, detail="Image not found")
    except FileResponse:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/semantic/upload")
async def semantic_upload(file: UploadFile = File(...)):
    """Upload and index a new satellite image"""
    try:
        _init_semantic_services()
        
        if not ImageProcessor.is_supported_format(file.filename):
            raise HTTPException(status_code=400, detail="Invalid image format")
        
        # Save uploaded file
        upload_dir = Path("/Users/mahesharunaladi/Downloads/DeepQuery/project/dataset/images")
        upload_dir.mkdir(parents=True, exist_ok=True)
        
        file_path = upload_dir / file.filename
        
        content = await file.read()
        with open(file_path, "wb") as f:
            f.write(content)
        
        # Generate embedding
        img = _image_processor.load_image(file_path)
        if img is None:
            raise HTTPException(status_code=400, detail="Failed to process image")
        
        img = _image_processor.preprocess(img)
        embedding = _embedding_service.encode_image(img)
        
        # Add to index
        _faiss_service.add_embeddings(
            np.array([embedding]).astype('float32'),
            [file_path]
        )
        _faiss_service.save_index()
        
        return {
            "status": "success",
            "message": f"Image '{file.filename}' uploaded and indexed",
            "image_id": file_path.stem
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

