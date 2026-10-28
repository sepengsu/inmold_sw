"""Launch the FastAPI RS232 diagnostic page without a camera or model."""

import argparse
from pathlib import Path
from threading import Timer
import webbrowser

import uvicorn

from src.hardware.camera_config import DEFAULT_CAMERA_CONFIG
from src.hardware.led_control import BAUD_RATE, COM_PORT, LightController
from src.web.app import create_app
from src.web.service import Workbench


def main() -> None:
    parser = argparse.ArgumentParser(description="RS232_test · FastAPI 6채널 조명 진단")
    parser.add_argument("--light-port", default=COM_PORT)
    parser.add_argument("--baudrate", type=int, default=BAUD_RATE)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7861)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535 or args.baudrate <= 0:
        parser.error("포트는 1~65535, 전송 속도는 양수여야 합니다.")
    root = Path(__file__).resolve().parent
    service = Workbench(
        config_path=root / "config/preprocess.yaml", checkpoint_path=root / "model/best_model.pt",
        device="cpu", batch_size=8, camera_serial=None, use_camera=False,
        camera_config_path=DEFAULT_CAMERA_CONFIG, load_model=False, auto_connect_light=False,
    )
    service.light = LightController(args.light_port, args.baudrate)
    if not args.no_browser:
        host = "127.0.0.1" if args.host in ("0.0.0.0", "::") else args.host
        Timer(1.5, lambda: webbrowser.open(f"http://{host}:{args.port}/rs232-test")).start()
    uvicorn.run(create_app(service), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
