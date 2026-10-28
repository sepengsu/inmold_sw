"""Camera and six-channel RS232 diagnostic HTTP routes."""

from __future__ import annotations

from base64 import b64encode
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from serial.tools import list_ports

from src.diagnostics.camera import CamTestController, DEFAULT_CAPTURE_DIR
from src.diagnostics.serial import RS232TestController
from src.hardware.led_control import BAUD_RATE, CHANNEL_COUNT, LIGHT_GROUPS
from src.web.service import Workbench, image_bytes


class CameraSettingsRequest(BaseModel):
    serial_number: str = ""
    gain: float
    fps: float


class ExposureRequest(BaseModel):
    exposure_us: float


class ConnectionRequest(BaseModel):
    port: str
    baudrate: int


class CameraLightRequest(ConnectionRequest):
    values: list[float]
    group: int | None = None


class SerialApplyRequest(ConnectionRequest):
    values: list[float]
    channel: int | None = None


class CaptureRequest(BaseModel):
    name: str = ""


def _image_url(image, extension: str = ".png") -> str | None:
    media = "image/jpeg" if extension == ".jpg" else "image/png"
    return None if image is None else f"data:{media};base64," + b64encode(image_bytes(image, extension)).decode("ascii")


def _serial_snapshot(controller: RS232TestController) -> dict:
    status, message, notes, history, connected = controller.snapshot()
    return {"status": status, "message": message, "notes": notes,
            "history": history, "connected": connected,
            "port": controller.light.port, "baudrate": controller.light.baudrate}


def attach_diagnostic_routes(app: FastAPI, workbench: Workbench, static_dir: Path) -> None:
    camera_test = CamTestController(
        workbench.camera.serial_number, workbench.light.port, workbench.light.baudrate,
        settings=workbench.camera.settings, roi_config=workbench.roi_config.path,
        camera_config=workbench.camera_config_path,
        capture_dir=getattr(workbench, "capture_dir", DEFAULT_CAPTURE_DIR),
        camera=workbench.camera, light=workbench.light,
        camera_enabled=workbench.camera_enabled,
    )
    serial_test = RS232TestController(light=workbench.light)

    def replace_camera(camera):
        workbench.camera = camera
        workbench.camera_enabled = True

    def replace_light(light):
        workbench.light = light
        camera_test.light = light
        serial_test.light = light
        camera_test._light_requested = ()
        camera_test._light_acknowledged = ()
        serial_test.last_ack.clear()

    camera_test.on_camera_replace = replace_camera
    camera_test.on_light_replace = replace_light
    serial_test.on_light_replace = replace_light
    app.state.camera_test = camera_test
    app.state.serial_test = serial_test

    @app.get("/camera-test")
    def camera_page():
        return FileResponse(static_dir / "camera-test.html")

    @app.get("/rs232-test")
    def rs232_page():
        return FileResponse(static_dir / "rs232-test.html")

    @app.get("/api/camera-test/bootstrap")
    def camera_bootstrap():
        levels, lighting_message = camera_test.lighting_config.initial_for_ui()
        return {"camera_status": camera_test.camera_status(),
                "light_status": camera_test.light_status(),
                "settings": {"serial_number": camera_test.camera.serial_number or "",
                             "gain": camera_test.settings.gain,
                             "fps": camera_test.settings.frame_rate_fps,
                             "exposure_us": camera_test.settings.exposure_us},
                "lighting": {"values": levels, "message": lighting_message,
                             "groups": [{"name": group.name, "channels": group.channel_label}
                                        for group in LIGHT_GROUPS]},
                "light": {"port": camera_test.light.port,
                          "baudrate": camera_test.light.baudrate,
                          "receipts": list(camera_test.light_receipts())},
                "ports": [port.device for port in list_ports.comports()]}

    @app.get("/api/camera-test/preview")
    def camera_preview(show_roi: bool = True):
        camera_test._camera_enabled = workbench.camera_enabled
        image, camera_status, light_status, frame_info = camera_test.preview(show_roi)
        return {"image": _image_url(image, ".jpg"), "camera_status": camera_status,
                "light_status": light_status, "frame_info": frame_info}

    @app.post("/api/camera-test/capture")
    def camera_capture(request: CaptureRequest):
        paths, message = camera_test.save_image(request.name)
        if paths is None:
            return {"message": message, "files": []}
        root = camera_test.capture_dir
        return {"message": message, "files": [
            {"name": Path(path).name,
             "url": "/api/camera-test/capture/" + Path(path).relative_to(root).as_posix()}
            for path in paths]}

    @app.get("/api/camera-test/capture/{date}/{filename}")
    def camera_download(date: str, filename: str):
        root = camera_test.capture_dir.resolve()
        path = (root / date / filename).resolve()
        if not path.is_relative_to(root) or not path.is_file() or path.suffix.lower() != ".png":
            raise HTTPException(404, "저장 이미지를 찾을 수 없습니다.")
        return FileResponse(path, media_type="image/png", filename=path.name)

    @app.post("/api/camera-test/roi")
    def camera_roi():
        image, message = camera_test.extract_roi()
        return {"image": _image_url(image), "message": message}

    @app.post("/api/camera-test/exposure")
    def camera_exposure(request: ExposureRequest):
        value, status, message = camera_test.apply_exposure(request.exposure_us)
        return {"exposure_us": value, "camera_status": status, "message": message}

    @app.get("/api/camera-test/exposure")
    def camera_exposure_load():
        value, message = camera_test.load_exposure()
        return {"exposure_us": value, "message": message}

    @app.post("/api/camera-test/camera/connect")
    def camera_connect(request: CameraSettingsRequest):
        status = camera_test.apply_camera(request.serial_number, request.gain, request.fps)
        return {"camera_status": status}

    @app.post("/api/camera-test/camera/disconnect")
    def camera_disconnect():
        status = camera_test.disconnect_camera()
        workbench.camera_enabled = False
        return {"camera_status": status}

    @app.post("/api/camera-test/light/connect")
    def camera_light_connect(request: ConnectionRequest):
        message = camera_test.connect_light(request.port, request.baudrate)
        return {"message": message, "light_status": camera_test.light_status(),
                "receipts": list(camera_test.light_receipts())}

    @app.post("/api/camera-test/light/disconnect")
    def camera_light_disconnect():
        message = camera_test.disconnect_light()
        return {"message": message, "light_status": camera_test.light_status(),
                "receipts": list(camera_test.light_receipts())}

    @app.post("/api/camera-test/light/apply")
    def camera_light_apply(request: CameraLightRequest):
        if len(request.values) != len(LIGHT_GROUPS):
            raise HTTPException(400, "네 조명의 밝기를 모두 지정하세요.")
        if request.group is None:
            message = camera_test.send_light_groups(tuple(request.values), request.port, request.baudrate)
        elif 1 <= request.group <= len(LIGHT_GROUPS):
            message = camera_test.send_light_group(request.group, request.values[request.group - 1],
                                                   request.port, request.baudrate)
        else:
            raise HTTPException(400, "조명 번호는 1~4여야 합니다.")
        return {"message": message, "receipts": list(camera_test.light_receipts()),
                "light_status": camera_test.light_status()}

    @app.post("/api/camera-test/light/off")
    def camera_light_off(request: ConnectionRequest):
        result = camera_test.turn_off_lights(request.port, request.baudrate)
        return {"message": result[0], "receipts": list(result[1:5]),
                "reset_values": list(result[5:]), "light_status": camera_test.light_status()}

    @app.get("/api/rs232-test/bootstrap")
    def serial_bootstrap():
        return {**_serial_snapshot(serial_test),
                "ports": [port.device for port in list_ports.comports()],
                "channels": CHANNEL_COUNT}

    @app.get("/api/rs232-test/status")
    def serial_status():
        return _serial_snapshot(serial_test)

    @app.post("/api/rs232-test/connect")
    def serial_connect(request: ConnectionRequest):
        serial_test.connect(request.port, request.baudrate)
        return _serial_snapshot(serial_test)

    @app.post("/api/rs232-test/disconnect")
    def serial_disconnect():
        serial_test.disconnect()
        return _serial_snapshot(serial_test)

    @app.post("/api/rs232-test/apply")
    def serial_apply(request: SerialApplyRequest):
        if request.channel is None:
            if len(request.values) != CHANNEL_COUNT:
                raise HTTPException(400, "여섯 채널의 밝기를 모두 지정하세요.")
            channels = tuple(enumerate(request.values, 1))
        else:
            if not 1 <= request.channel <= CHANNEL_COUNT or len(request.values) != 1:
                raise HTTPException(400, "채널 번호와 밝기 한 개를 지정하세요.")
            channels = ((request.channel, request.values[0]),)
        accepted = serial_test.apply(channels, request.port, request.baudrate)
        return {**_serial_snapshot(serial_test), "acknowledged": accepted}

    @app.post("/api/rs232-test/off")
    def serial_off(request: ConnectionRequest):
        accepted = serial_test.turn_off(request.port, request.baudrate)
        return {**_serial_snapshot(serial_test), "reset_channels": accepted}

    @app.post("/api/rs232-test/history/clear")
    def serial_clear():
        serial_test.clear_history()
        return _serial_snapshot(serial_test)
