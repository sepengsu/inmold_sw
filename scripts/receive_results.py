"""Standalone example for the receiving PC; prints JSON without saving files."""

import argparse
import json
import socketserver


class ResultHandler(socketserver.StreamRequestHandler):
    def handle(self):
        self.connection.settimeout(5)
        try:
            line = self.rfile.readline(65537)
            if not line:
                return  # The sender also makes empty connections to check the port.
            if len(line) > 65536 or not line.endswith(b"\n"):
                raise ValueError("Expected a JSON line of at most 64 KiB")
            result = json.loads(line.decode("utf-8"))
            print(json.dumps(result, ensure_ascii=False), flush=True)
        except (OSError, UnicodeError, ValueError) as error:
            print(f"Receive error from {self.client_address[0]}: {error}", flush=True)


class ResultServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


def main():
    parser = argparse.ArgumentParser(description="Receive measurement JSON lines over TCP")
    parser.add_argument("--host", default="192.168.50.2")
    parser.add_argument("--port", type=int, default=9997)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    with ResultServer((args.host, args.port), ResultHandler) as server:
        print(f"Listening on {args.host}:{args.port}", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
