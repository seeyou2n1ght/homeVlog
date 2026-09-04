"""HomeVlog 独立审核工作台 Web 服务端 (Audit Web Server)

基于 Python 标准库 http.server，无需安装额外 Web 框架。
使用方法:
  uv run python scripts/audit_tool/app.py [--port 8765] [--no-browser]
"""

import sys
import json
import time
import socket
import logging
import argparse
import webbrowser
from pathlib import Path
from typing import Any
from urllib.parse import urlparse, parse_qs
from http.server import HTTPServer, BaseHTTPRequestHandler

# 引入项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.audit_tool.service import AuditService, CACHE_DIR
from scripts.audit_tool.rerender import ReRenderManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("homevlog.audit.server")

STATIC_DIR = Path(__file__).resolve().parent / "static"
service = AuditService()
rerender_mgr = ReRenderManager()


class AuditHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        # 保持控制台输出整洁
        logger.debug("%s - - [%s] %s", self.address_string(), self.log_date_time_string(), format % args)

    def _send_json(self, data: Any, status: int = 200):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _send_bytes(self, data: bytes, content_type: str, status: int = 200):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(data)

    def _send_file(self, file_path: Path, content_type: str):
        if not file_path.exists():
            self.send_error(404, "File not found")
            return
        try:
            with open(file_path, "rb") as f:
                content = f.read()
            self._send_bytes(content, content_type)
        except Exception as e:
            self.send_error(500, str(e))

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        params = parse_qs(parsed.query)

        # 1. 静态资源路由
        if path in ("/", "/index.html"):
            self._send_file(STATIC_DIR / "index.html", "text/html; charset=utf-8")
            return
        elif path in ("/prototype", "/prototype.html"):
            self._send_file(STATIC_DIR / "prototype.html", "text/html; charset=utf-8")
            return
        elif path == "/prototype.js":
            self._send_file(STATIC_DIR / "prototype.js", "application/javascript; charset=utf-8")
            return
        elif path == "/prototype.css":
            self._send_file(STATIC_DIR / "prototype.css", "text/css; charset=utf-8")
            return
        elif path == "/app.js":
            self._send_file(STATIC_DIR / "app.js", "application/javascript; charset=utf-8")
            return
        elif path == "/style.css":
            self._send_file(STATIC_DIR / "style.css", "text/css; charset=utf-8")
            return

        # 2. REST API 路由
        if path == "/api/overview":
            self._send_json(service.get_overview())
            return
        elif path == "/api/tree":
            self._send_json(service.get_file_tree())
            return
        elif path == "/api/anomalies":
            limit = int(params.get("limit", [60])[0])
            dt = params.get("date", [None])[0]
            cat = params.get("category", ["all"])[0]
            cam = params.get("cam_index", [None])[0]
            cam_int = int(cam) if cam is not None else None
            self._send_json(service.get_anomalies(category=cat, date=dt, cam_index=cam_int, limit=limit))
            return
        elif path == "/api/file_segments":
            fid = params.get("file_id", [None])[0]
            fp = params.get("filepath", [None])[0]
            fid_int = int(fid) if fid else None
            self._send_json(service.get_file_segments(file_id=fid_int, filepath=fp))
            return
        elif path == "/api/frame":
            fp = params.get("filepath", [""])[0]
            t = float(params.get("t", [0.0])[0])
            w = int(params.get("w", [640])[0])
            out = service.extract_frame(fp, t, width=w)
            if out and out.exists():
                self._send_file(out, "image/jpeg")
            else:
                self.send_error(404, "Frame extraction failed")
            return
        elif path == "/api/clip":
            fp = params.get("filepath", [""])[0]
            st = float(params.get("start", [0.0])[0])
            et = float(params.get("end", [st + 4.0])[0])
            out = service.generate_preview_clip(fp, st, et)
            if out and out.exists():
                self._send_file(out, "image/webp")
            else:
                self.send_error(404, "Clip generation failed")
            return
        elif path == "/api/frame_cached":
            name = params.get("name", [""])[0]
            target = CACHE_DIR / Path(name).name
            if target.exists():
                self._send_file(target, "image/jpeg")
            else:
                self.send_error(404, "Cached frame not found")
            return
        elif path == "/api/export":
            fmt = params.get("format", ["csv"])[0].lower()
            content, ctype = service.export_report(fmt=fmt)
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            ts = time.strftime("%Y%m%d_%H%M%S")
            fn = f"homevlog_audit_{ts}.{fmt}"
            self.send_header("Content-Disposition", f'attachment; filename="{fn}"')
            self.send_header("Access-Control-Allow-Origin", "*")

            if fmt == "csv":
                body = b'\xef\xbb\xbf' + content.encode("utf-8")
            else:
                body = content.encode("utf-8")

            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        elif path == "/api/rerender_status":
            dt = params.get("date", [""])[0]
            cam = int(params.get("cam_index", [0])[0])
            st = rerender_mgr.get_status(dt, cam)
            self._send_json(st)
            return
        elif path == "/api/archive_stats":
            self._send_json(service.get_archive_stats())
            return
        elif path == "/api/open_output_dir":
            fp = params.get("filepath", [""])[0]
            target_path = Path(fp) if fp else None
            import subprocess
            if target_path and target_path.exists():
                subprocess.Popen(f'explorer /select,"{target_path}"', shell=True)
                self._send_json({"success": True})
            else:
                self._send_json({"error": "File not found"}, status=404)
            return

        self.send_error(404, "Endpoint not found")

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path
        length = int(self.headers.get("Content-Length", 0))
        post_data = self.rfile.read(length) if length > 0 else b"{}"

        try:
            payload = json.loads(post_data.decode("utf-8"))
        except Exception:
            self._send_json({"error": "Invalid JSON"}, status=400)
            return

        if path == "/api/review":
            seg_id = payload.get("segment_id")
            label = payload.get("manual_label")
            notes = payload.get("notes", "")
            if not seg_id or not label:
                self._send_json({"error": "Missing segment_id or manual_label"}, status=400)
                return
            ok = service.submit_review(int(seg_id), str(label), str(notes))
            self._send_json({"success": ok})
            return

        elif path == "/api/yolo_detect":
            fp = payload.get("filepath", "")
            t = float(payload.get("timestamp", 0.0))
            if not fp:
                self._send_json({"error": "Missing filepath"}, status=400)
                return
            res = service.detect_and_draw_yolo(fp, t)
            self._send_json(res)
            return

        elif path == "/api/rerender":
            dt = payload.get("date", "")
            cam = int(payload.get("cam_index", 0))
            version = payload.get("version", "reviewed")
            if not dt:
                self._send_json({"error": "Missing date"}, status=400)
                return
            res = rerender_mgr.start_rerender(service.db, dt, cam, output_version=version)
            self._send_json(res)
            return

        elif path == "/api/rerender_cancel":
            dt = payload.get("date", "")
            cam = int(payload.get("cam_index", 0))
            ok = rerender_mgr.cancel_task(dt, cam)
            self._send_json({"success": ok})
            return

        elif path == "/api/archive_all":
            force = bool(payload.get("force", False))
            res = service.batch_archive_all(force=force)
            self._send_json(res)
            return

        self.send_error(404, "Endpoint not found")


def find_free_port(start_port: int = 8765) -> int:
    for port in range(start_port, start_port + 50):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return port
    return start_port


def main():
    parser = argparse.ArgumentParser(description="HomeVlog Audit Web Console")
    parser.add_argument("--port", type=int, default=8765, help="HTTP Server port")
    parser.add_argument("--no-browser", action="store_true", help="Do not auto-open browser")
    args = parser.parse_args()

    port = find_free_port(args.port)
    server_addr = ("127.0.0.1", port)
    httpd = HTTPServer(server_addr, AuditHandler)
    url = f"http://127.0.0.1:{port}"

    print("=" * 60)
    print(f" HomeVlog 人工二次审核与质量评估工作台已启动!")
    print(f" 本地访问地址: {url}")
    print(f" 按 Ctrl+C 可停止服务")
    print("=" * 60)

    if not args.no_browser:
        try:
            webbrowser.open(url)
        except Exception:
            pass

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n正在停止服务...")
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()
