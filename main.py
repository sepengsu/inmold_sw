"""Launch the FastAPI and HTML inspection workbench."""

import argparse
from pathlib import Path
from threading import Timer
import webbrowser

from src.hardware.camera_config import DEFAULT_CAMERA_CONFIG


def main() -> None:
    project_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Basler / 이미지 파일 함침율 검사 GUI")
    parser.add_argument("--config", type=Path, default=project_root / "config/preprocess.yaml")
    parser.add_argument("--checkpoint", type=Path, default=project_root / "model/best_model.pt")
    parser.add_argument("--device", default="auto", help="추론 장치 (auto: CUDA 우선, cpu, cuda:0 등)")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--camera-serial", help="Basler 카메라 일련번호")
    parser.add_argument("--camera-config", type=Path, default=DEFAULT_CAMERA_CONFIG, help="카메라 시작 설정 YAML")
    parser.add_argument("--no-camera", action="store_true", help="카메라 없이 이미지 파일 검사")
    parser.add_argument("--no-browser", action="store_true", help="브라우저 자동 열기 해제")
    parser.add_argument("--keep-server", action="store_true", help="모든 검사 UI를 닫아도 서버 유지")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size는 1 이상이어야 합니다.")
    if not 1 <= args.port <= 65535:
        parser.error("--port는 1~65535 범위여야 합니다.")

    import uvicorn
    from src.web.app import create_app
    from src.web.service import Workbench
    from src.web.ui_lifecycle import UiSessions

    service = Workbench(
        config_path=args.config, checkpoint_path=args.checkpoint, device=args.device,
        batch_size=args.batch_size, camera_serial=args.camera_serial,
        use_camera=not args.no_camera, camera_config_path=args.camera_config,
    )
    def request_shutdown() -> None:
        server.should_exit = True

    ui_sessions = None if args.keep_server else UiSessions(request_shutdown, lambda: server.should_exit)
    server = uvicorn.Server(uvicorn.Config(
        create_app(service, ui_sessions=ui_sessions), host=args.host, port=args.port,
    ))
    browser_timer = None
    if not args.no_browser:
        browser_host = "127.0.0.1" if args.host in ("0.0.0.0", "::") else args.host
        browser_timer = Timer(1.5, lambda: webbrowser.open(f"http://{browser_host}:{args.port}/"))
        browser_timer.start()
    try:
        server.run()
    finally:
        if browser_timer is not None:
            browser_timer.cancel()


if __name__ == "__main__":
    main()
