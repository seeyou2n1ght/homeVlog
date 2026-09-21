"""HomeVlog 独立审核工作台 Web 服务端 (Audit Web Server)

基于 Python 标准库 http.server，无需安装额外 Web 框架。
使用方法:
  uv run python scripts/audit_tool/app.py [--port 8765] [--no-browser]
"""

import sys
import json
import math
import time
import socket
import secrets
import logging
import argparse
import webbrowser
from http.cookies import SimpleCookie
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
AUDIT_TOKEN = secrets.token_urlsafe(32)


class AuditHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        # 保持控制台输出整洁
        logger.debug("%s - - [%s] %s", self.address_string(), self.log_date_time_string(), format % args)

    def _send_json(self, data: Any, status: int = 200):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_bytes(self, data: bytes, content_type: str, status: int = 200, set_token_cookie: bool = False):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        if set_token_cookie:
            self.send_header(
                "Set-Cookie",
                f"homevlog_audit_token={AUDIT_TOKEN}; HttpOnly; SameSite=Strict; Path=/",
            )
        self.end_headers()
        self.wfile.write(data)

    def _send_file(self, file_path: Path, content_type: str, set_token_cookie: bool = False):
        if not file_path.exists():
            self.send_error(404, "File not found")
            return
        try:
            with open(file_path, "rb") as f:
                content = f.read()
            self._send_bytes(content, content_type, set_token_cookie=set_token_cookie)
        except Exception as e:
            self.send_error(500, str(e))

    def do_OPTIONS(self):
        self.send_error(403, "Cross-origin requests are not allowed")

    def _trusted_request(self, require_token: bool = False) -> bool:
        host = self.headers.get("Host", "")
        if not (host.startswith("127.0.0.1:") or host.startswith("localhost:")):
            return False
        origin = self.headers.get("Origin")
        if origin is not None and origin not in {f"http://{host}", f"https://{host}"}:
            return False
        if not require_token:
            return True
        cookie = SimpleCookie(self.headers.get("Cookie", ""))
        supplied = cookie.get("homevlog_audit_token")
        return supplied is not None and secrets.compare_digest(supplied.value, AUDIT_TOKEN)

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        params = parse_qs(parsed.query)

        # 1. 静态资源路由
        if path in ("/", "/index.html"):
            self._send_file(STATIC_DIR / "index.html", "text/html; charset=utf-8", set_token_cookie=True)
            return
        elif path in ("/prototype", "/prototype.html"):
            self._send_file(STATIC_DIR / "prototype.html", "text/html; charset=utf-8", set_token_cookie=True)
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
        if path.startswith("/api/") and not self._trusted_request():
            self._send_json({"error": "Untrusted request origin"}, status=403)
            return
        if path == "/api/overview":
            self._send_json(service.get_overview())
            return
        elif path == "/api/tree":
            self._send_json(service.get_file_tree())
            return
        elif path == "/api/anomalies":
            limit = int(params.get("limit", [60])[0])
            offset = int(params.get("offset", [0])[0])
            if not 1 <= limit <= 500:
                self.send_error(400, "limit must be between 1 and 500")
                return
            if offset < 0:
                self.send_error(400, "offset must be non-negative")
                return
            dt = params.get("date", [None])[0]
            cat = params.get("category", ["all"])[0]
            cam = params.get("cam_index", [None])[0]
            cam_int = int(cam) if cam is not None else None
            self._send_json(service.get_anomalies(category=cat, date=dt, cam_index=cam_int, limit=limit, offset=offset))
            return
        elif path == "/api/file_segments":
            fid = params.get("file_id", [None])[0]
            fp = params.get("filepath", [None])[0]
            fid_int = int(fid) if fid else None
            self._send_json(service.get_file_segments(file_id=fid_int, filepath=fp))
            return
        elif path == "/api/frame":
            fp = params.get("filepath", [""])[0]
            if not service.is_registered_filepath(fp):
                self.send_error(400, "Unknown filepath")
                return
            t = float(params.get("t", [0.0])[0])
            w = int(params.get("w", [640])[0])
            if not math.isfinite(t) or t < 0 or not 64 <= w <= 1920:
                self.send_error(400, "Invalid frame range")
                return
            out = service.extract_frame(fp, t, width=w)
            if out and out.exists():
                self._send_file(out, "image/jpeg")
            else:
                self.send_error(404, "Frame extraction failed")
            return
        elif path == "/api/clip":
            fp = params.get("filepath", [""])[0]
            if not service.is_registered_filepath(fp):
                self.send_error(400, "Unknown filepath")
                return
            st = float(params.get("start", [0.0])[0])
            et = float(params.get("end", [st + 4.0])[0])
            if not all(map(math.isfinite, (st, et))) or st < 0 or et <= st or et - st > 30:
                self.send_error(400, "Invalid clip range")
                return
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
            only_rev = params.get("only_reviewed", ["false"])[0].lower() in ("true", "1")
            content, ctype = service.export_report(fmt=fmt, only_reviewed=only_rev)
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            ts = time.strftime("%Y%m%d_%H%M%S")
            fn = f"homevlog_audit_{ts}.{fmt}"
            self.send_header("Content-Disposition", f'attachment; filename="{fn}"')

            if fmt == "csv":
                body = b'\xef\xbb\xbf' + content.encode("utf-8")
            else:
                body = content.encode("utf-8")

            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        elif path == "/api/tuning_insights":
            self._send_json(service.get_tuning_insights())
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
            from src.core.config import OUTPUT_DIR
            if (target_path and target_path.exists()
                    and target_path.resolve().is_relative_to(OUTPUT_DIR.resolve())):
                subprocess.Popen(["explorer", "/select,", str(target_path.resolve())])
                self._send_json({"success": True})
            else:
                self._send_json({"error": "File not found"}, status=404)
            return

        self.send_error(404, "Endpoint not found")

    def do_POST(self):
        if not self._trusted_request(require_token=True):
            self._send_json({"error": "Untrusted request origin"}, status=403)
            return
        parsed = urlparse(self.path)
        path = parsed.path
        length = int(self.headers.get("Content-Length", 0))
        if length > 64 * 1024:
            self._send_json({"error": "Request body too large"}, status=413)
            return
        if not self.headers.get("Content-Type", "").lower().startswith("application/json"):
            self._send_json({"error": "Content-Type must be application/json"}, status=415)
            return
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
            scenario = payload.get("scenario", "")
            if not seg_id or not label:
                self._send_json({"error": "Missing segment_id or manual_label"}, status=400)
                return
            ok = service.submit_review(
                int(seg_id),
                str(label),
                notes=str(notes),
                scenario=str(scenario),
            )
            self._send_json({"success": ok})
            return

        elif path == "/api/clear_review":
            seg_id = payload.get("segment_id")
            if not seg_id:
                self._send_json({"error": "Missing segment_id"}, status=400)
                return
            ok = service.clear_review(int(seg_id))
            self._send_json({"success": ok})
            return

        elif path == "/api/save_bbox":
            boxes = payload.get("boxes", [])
            image_name = payload.get("image_name")
            segment_id = payload.get("segment_id")
            if not boxes or not isinstance(boxes, list):
                self._send_json({"error": "boxes must be a non-empty list"}, status=400)
                return
            res = service.save_bounding_box(
                boxes=boxes,
                image_name=str(image_name) if image_name else None,
                segment_id=int(segment_id) if segment_id is not None else None,
            )
            self._send_json(res)
            return

        elif path == "/api/export_yolo_dataset":
            val_ratio = float(payload.get("val_ratio", 0.2))
            out_dir = payload.get("output_dir")
            res = service.export_yolo_dataset(val_ratio=val_ratio, output_dir=out_dir)
            self._send_json(res)
            return

        elif path == "/api/yolo_detect":
            fp = payload.get("filepath", "")
            t = float(payload.get("timestamp", 0.0))
            if not math.isfinite(t) or t < 0:
                self._send_json({"error": "Invalid timestamp"}, status=400)
                return
            if not service.is_registered_filepath(fp):
                self._send_json({"error": "Unknown filepath"}, status=400)
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
