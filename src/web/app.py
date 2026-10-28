"""HTTP API and static HTML for the inspection workbench."""

from __future__ import annotations

from base64 import b64encode
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from src.hardware.lighting_config import validate_lighting_values
from src.roi.quality_profiles import ROI_QUALITY_PROFILES, linked_roi_settings, roi_level, roi_settings_summary
from src.roi.threshold_config import validate_roi_threshold

from .service import Workbench
from .diagnostics import attach_diagnostic_routes


STATIC_DIR = Path(__file__).parent / "static"


class PixelFormatRequest(BaseModel):
    pixel_format: str


class RoiRequest(BaseModel):
    threshold: float


class LightingRequest(BaseModel):
    values: list[float]
    group: int | None = None


class ClearRequest(BaseModel):
    confirmed: bool


def _images_to_urls(payload: dict) -> dict:
    payload["images"] = {key: "data:image/png;base64," + b64encode(data).decode("ascii")
                         for key, data in payload["images"].items()}
    return payload


def create_app(workbench: Workbench) -> FastAPI:
    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        workbench.start()
        try:
            yield
        finally:
            workbench.stop()

    app = FastAPI(title="Vision Workbench", lifespan=lifespan)
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    attach_diagnostic_routes(app, workbench, STATIC_DIR)

    @app.get("/")
    def index():
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/api/bootstrap")
    def bootstrap():
        return workbench.bootstrap()

    @app.get("/api/status")
    def status():
        return workbench.status()

    @app.get("/api/preview")
    def preview():
        data = workbench.preview()
        if data is None:
            return Response(status_code=204, headers={"Cache-Control": "no-store"})
        return Response(data, media_type="image/jpeg", headers={"Cache-Control": "no-store"})

    @app.post("/api/camera/reconnect")
    def reconnect():
        return workbench.reconnect()

    @app.post("/api/camera/pixel-format")
    def pixel_format(request: PixelFormatRequest):
        try:
            return workbench.set_pixel_format(request.pixel_format)
        except (ValueError, OSError, RuntimeError) as error:
            raise HTTPException(400, str(error)) from error

    @app.get("/api/roi")
    def load_roi():
        return workbench.roi_settings()

    @app.get("/api/roi/preview")
    def preview_roi(threshold: float | None = None, level: str | None = None):
        try:
            if level is not None:
                if level not in ROI_QUALITY_PROFILES:
                    raise ValueError("낮음·중간·높음 중 ROI 검증 강도를 선택하세요.")
                threshold = ROI_QUALITY_PROFILES[level]["quality"]["min_score"]
            if threshold is None:
                raise ValueError("ROI 기준을 지정하세요.")
            config = linked_roi_settings(validate_roi_threshold(threshold))
            return {"threshold": threshold, "level": roi_level(config),
                    "details": roi_settings_summary(config)}
        except ValueError as error:
            raise HTTPException(400, str(error)) from error

    @app.post("/api/roi")
    def save_roi(request: RoiRequest):
        try:
            return workbench.save_roi(request.threshold)
        except (ValueError, OSError) as error:
            raise HTTPException(400, str(error)) from error

    @app.get("/api/lighting")
    def load_lighting():
        return workbench.load_lighting()

    @app.post("/api/lighting/config")
    def save_lighting(request: LightingRequest):
        return workbench.save_lighting(request.values)

    @app.post("/api/lighting/apply")
    def apply_lighting(request: LightingRequest):
        if request.group is not None and not 1 <= request.group <= 4:
            raise HTTPException(400, "조명 번호는 1~4여야 합니다.")
        if len(request.values) != 4:
            raise HTTPException(400, "네 조명의 밝기를 모두 지정하세요.")
        return workbench.apply_lighting(request.values, request.group)

    @app.post("/api/inspect")
    async def inspect(source: str = Form(...), mode: str = Form("normal"),
                      lighting: str = Form("0,0,0,0"), image: UploadFile | None = File(None)):
        try:
            values = validate_lighting_values([float(value) for value in lighting.split(",")])
            frame = None
            if source == "file" and image is not None:
                content = await image.read(30 * 1024 * 1024 + 1)
                if len(content) > 30 * 1024 * 1024:
                    raise ValueError("이미지 파일은 30MB 이하여야 합니다.")
                decoded = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_COLOR)
                if decoded is None:
                    raise ValueError("이미지 파일을 읽을 수 없습니다.")
                frame = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
            return _images_to_urls(workbench.inspect(source, mode, frame, values))
        except (ValueError, OSError, RuntimeError) as error:
            raise HTTPException(400, str(error)) from error
        finally:
            if image is not None:
                await image.close()

    @app.get("/api/history")
    def history():
        from .service import record_data
        return [record_data(r) for r in workbench.history.query(limit=5)]

    @app.get("/api/statistics")
    def statistics(start: str | None = None, end: str | None = None,
                   source: str = "all", page: int = 1):
        try:
            return workbench.statistics(start, end, source, page)
        except (ValueError, OSError) as error:
            raise HTTPException(400, str(error)) from error

    @app.get("/api/statistics/{record_id}")
    def detail(record_id: str):
        try:
            return _images_to_urls(workbench.detail(record_id))
        except (ValueError, OSError, KeyError) as error:
            raise HTTPException(404, str(error)) from error

    @app.get("/api/statistics/export/csv")
    def export_csv(start: str | None = None, end: str | None = None, source: str = "all"):
        try:
            path = workbench.history.export_csv(start, end, source)
            return FileResponse(path, media_type="text/csv", filename=path.name)
        except (ValueError, OSError) as error:
            raise HTTPException(400, str(error)) from error

    @app.post("/api/statistics/clear")
    def clear(request: ClearRequest):
        if not request.confirmed:
            raise HTTPException(400, "삭제 확인을 선택한 뒤 다시 누르세요.")
        try:
            with workbench.lock:
                count = workbench.history.clear_all()
            return {"count": count, "message": f"전체 측정 이력 {count:,}건과 저장 영상·CSV를 초기화했습니다."}
        except (ValueError, OSError) as error:
            raise HTTPException(400, str(error)) from error

    return app
