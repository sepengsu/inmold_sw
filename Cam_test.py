"""Launch the FastAPI camera diagnostic page without loading a model."""

import argparse
from pathlib import Path
from threading import Timer
import webbrowser

import uvicorn

from src.diagnostics.camera import DEFAULT_CAPTURE_DIR
from src.hardware.camera_config import DEFAULT_CAMERA_CONFIG
from src.hardware.led_control import BAUD_RATE, COM_PORT, LightController
from src.roi.threshold_config import RoiThresholdConfigStore
from src.web.app import create_app
from src.web.service import Workbench


def main() -> None:
    parser = argparse.ArgumentParser(description="Cam_test · FastAPI 카메라 및 조명 진단")
    parser.add_argument("--camera-serial", default="24346077")
    parser.add_argument("--light-port", default=COM_PORT)
    parser.add_argument("--baudrate", type=int, default=BAUD_RATE)
    parser.add_argument("--roi-config", type=Path)
    parser.add_argument("--camera-config", type=Path, default=DEFAULT_CAMERA_CONFIG)
    parser.add_argument("--capture-dir", type=Path, default=DEFAULT_CAPTURE_DIR)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535 or args.baudrate <= 0:
        parser.error("포트는 1~65535, 전송 속도는 양수여야 합니다.")
    root = Path(__file__).resolve().parent
    service = Workbench(
        config_path=root / "config/preprocess.yaml", checkpoint_path=root / "model/best_model.pt",
        device="cpu", batch_size=8, camera_serial=args.camera_serial or None, use_camera=True,
        camera_config_path=args.camera_config, load_model=False,
    )
    service.light = LightController(args.light_port, args.baudrate)
    service.capture_dir = args.capture_dir.expanduser().resolve()
    if args.roi_config:
        service.roi_config = RoiThresholdConfigStore(args.roi_config)
    if not args.no_browser:
        host = "127.0.0.1" if args.host in ("0.0.0.0", "::") else args.host
        Timer(1.5, lambda: webbrowser.open(f"http://{host}:{args.port}/camera-test")).start()
    uvicorn.run(create_app(service), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
