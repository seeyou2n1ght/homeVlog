"""HomeVlog 独立审核工作台 Web 服务端 (Audit Web Server)

基于 Python 标准库 http.server，无需安装额外 Web 框架。
使用方法:
  uv run python scripts/audit_tool/app.py [--port 8765] [--no-browser]
"""

import sys
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
import json
import math
import time
import socket
import secrets
import threading
import logging
import argparse
import webbrowser
from http.cookies import SimpleCookie
from pathlib import Path
from typing import Any
from urllib.parse import urlparse, parse_qs
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler

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
service = None
_service_lock = threading.RLock()


def initialize_service():
    global service
    with _service_lock:
        if service is None:
            service = AuditService()
        return service

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
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_bytes(self, data: bytes, content_type: str, status: int = 200, set_token_cookie: bool = False, no_store: bool = False):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        if no_store:
            self.send_header("Cache-Control", "no-store")
        if set_token_cookie:
            self.send_header(
                "Set-Cookie",
                f"homevlog_audit_token={AUDIT_TOKEN}; HttpOnly; SameSite=Strict; Path=/",
            )
        self.end_headers()
        self.wfile.write(data)

    def _send_file(self, file_path: Path, content_type: str, set_token_cookie: bool = False, no_store: bool = False):
        if not file_path.exists():
            self.send_error(404, "File not found")
            return
        try:
            with open(file_path, "rb") as f:
                content = f.read()
            self._send_bytes(content, content_type, set_token_cookie=set_token_cookie, no_store=no_store)
        except Exception as e:
            self.send_error(500, str(e))

    def _send_video_stream(self, file_path: Path, *, expected_identity=None):
        import os
        try:
            with open(file_path, "rb") as stream:
                stat = os.fstat(stream.fileno())
                if expected_identity and (stat.st_size != expected_identity["size"] or stat.st_mtime_ns != expected_identity["mtime_ns"]):
                    self._send_json({"error": "Delivery changed before playback; refresh comparison"}, status=409)
                    return
                self._stream_open_video(stream, stat.st_size)
        except FileNotFoundError:
            self.send_error(404, "Video file not found")

    def _watch_get(self, path, params):
        try:
            q = {k:v[0] for k,v in params.items()}
            date, camera = q.get("date", ""), int(q.get("cam_index", 0))
            manager = service.watch_exports()
            if path == "/api/watch_preview":
                result = manager.preview(date, camera, float(q.get("budget_s", 1200)))
            elif path == "/api/watch_catalog":
                from datetime import datetime
                datetime.strptime(date, "%Y%m%d")
                if len(date) != 8 or camera < 0:
                    raise ValueError("Invalid date/camera")
                folder = manager.root/"watch_exports"/f"{date}_cam{camera}"
                result = {"exports": [p.parent.name for p in sorted(folder.glob("*/manifest.json")) if not p.parent.name.startswith(".")]}
            elif path == "/api/watch_status":
                result = manager.status(date, camera, q["export_id"])
            elif path == "/api/watch_list":
                result = manager.listing(date, camera, q["export_id"], q.get("kind", "review"), int(q.get("offset",0)), int(q.get("limit",50)))
            elif path in ("/api/watch_playback", "/api/watch_stream"):
                result = service.watch_playback(date, camera, q["export_id"], q.get("kind", "daily"), q.get("item_id"))
                if path == "/api/watch_stream":
                    self._send_video_stream(Path(result["path"]), expected_identity=result["identity"])
                    return
            elif path == "/api/watch_manifest":
                directory, result = manager.package(date, camera, q["export_id"])
            else:
                raise ValueError("Unknown watch API")
            self._send_json(result)
        except (ValueError, KeyError, TypeError, OSError) as error:
            self._send_json({"error": str(error)}, status=409)

    def _watch_post(self, path, payload):
        try:
            manager = service.watch_exports()
            if path == "/api/watch_start":
                result = manager.start(str(payload["date"]), int(payload["cam_index"]),
                                       str(payload["expected_snapshot"]), float(payload.get("budget_s",1200)))
            elif path == "/api/watch_cancel":
                result = {"cancel_requested": manager.cancel(str(payload["expected_snapshot"]))}
            else:
                raise ValueError("Unknown watch API")
            self._send_json(result)
        except (ValueError, KeyError, TypeError, OSError) as error:
            self._send_json({"error": str(error)}, status=409)

    def _stream_open_video(self, f, file_size):
        """支持 HTTP 206 Partial Content 的流式视频分发"""
        range_header = self.headers.get("Range")

        if not range_header:
            self.send_response(200)
            self.send_header("Content-Type", "video/mp4")
            self.send_header("Content-Length", str(file_size))
            self.send_header("Accept-Ranges", "bytes")
            self.end_headers()
            try:
                while chunk := f.read(1024 * 1024):
                    self.wfile.write(chunk)
            except (ConnectionResetError, BrokenPipeError):
                pass
            return

        # Only one byte range is supported. Reject malformed/empty ranges before
        # sending headers, and clamp a satisfiable end to the representation size.
        import re
        match = re.fullmatch(r"bytes=([0-9]*)-([0-9]*)", range_header.strip(), re.IGNORECASE)
        try:
            if match is None or file_size == 0:
                raise ValueError("Invalid byte range")
            first, last = match.groups()
            if not first:
                suffix = int(last)
                if suffix <= 0:
                    raise ValueError("Empty suffix")
                start, end = max(0, file_size - suffix), file_size - 1
            else:
                start = int(first)
                end = min(int(last), file_size - 1) if last else file_size - 1
                if start >= file_size or start > end:
                    raise ValueError("Unsatisfiable range")
        except ValueError:
            self.send_response(416)
            self.send_header("Content-Range", f"bytes */{file_size}")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return

        try:
            chunk_len = end - start + 1
            self.send_response(206)
            self.send_header("Content-Type", "video/mp4")
            self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
            self.send_header("Content-Length", str(chunk_len))
            self.send_header("Accept-Ranges", "bytes")
            self.end_headers()

            f.seek(start)
            remaining = chunk_len
            while remaining > 0:
                read_bytes = min(remaining, 1024 * 1024)
                data = f.read(read_bytes)
                if not data:
                    break
                self.wfile.write(data)
                remaining -= len(data)
        except (ConnectionResetError, BrokenPipeError):
            pass
        except Exception as e:
            logger.debug("Video stream error: %s", e)

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
        initialize_service()
        if path.startswith("/api/watch_"):
            self._watch_get(path, params)
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
        elif path == "/api/review_queue":
            try:
                def parameter(name, default=None):
                    return params.get(name, [default])[0]
                date = parameter("date")
                if date:
                    from datetime import datetime
                    if len(date) != 8:
                        raise ValueError("Date must be YYYYMMDD")
                    datetime.strptime(date, "%Y%m%d")
                camera = int(parameter("cam_index")) if parameter("cam_index") else None
                if camera is not None and camera < 0:
                    raise ValueError("Camera must be nonnegative")
                self._send_json(service.get_review_queue(
                    category=parameter("category", "gold_candidates"), date=date, cam_index=camera,
                    state=parameter("state"), scenario=parameter("scenario"), search=parameter("search", ""),
                    hide_reviewed=parameter("hide_reviewed", "0") == "1", limit=int(parameter("limit", "50")),
                    offset=int(parameter("offset", "0")), seed=int(parameter("seed", "0")),
                    reference_id=int(parameter("reference_id")) if parameter("reference_id") else None))
            except ValueError as exc:
                self._send_json({"error": str(exc)}, status=400)
            return
        elif path in ("/api/delivery_comparison", "/api/delivery_stream"):
            try:
                segment_id = int(params.get("segment_id", [""])[0])
                comparison = service.get_delivery_comparison(segment_id)
                if path == "/api/delivery_comparison":
                    self._send_json(comparison)
                else:
                    version_id = params.get("version_id", [""])[0]
                    version = next((item for item in comparison["versions"] if item["version_id"] == version_id), None)
                    if not version:
                        self._send_json({"error": "Delivery changed or unavailable; refresh comparison"}, status=409)
                    else:
                        self._send_video_stream(Path(version["filepath"]), expected_identity=version["media_identity"])
            except ValueError as exc:
                self._send_json({"error": str(exc)}, status=400)
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
                self._send_file(out, "image/jpeg", no_store=True)
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
                self._send_file(out, "image/webp", no_store=True)
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
        elif path == "/api/review_context":
            try:
                segment_id = int(params.get("segment_id", [0])[0])
                self._send_json(service.get_review_context(segment_id))
            except ValueError as exc:
                self._send_json({"error": str(exc)}, status=400)
            return
        elif path == "/api/rerender_status":
            dt = params.get("date", [""])[0]
            cam = int(params.get("cam_index", [0])[0])
            st = rerender_mgr.get_status(dt, cam)
            self._send_json(st)
            return
        elif path == "/api/rerender_plan":
            from scripts.audit_tool.rerender import inspect_review_render
            dt = params.get("date", [""])[0]
            cam = int(params.get("cam_index", [0])[0])
            if len(dt) != 8 or not dt.isdigit() or cam < 0:
                self.send_error(400, "Invalid date or camera")
                return
            try:
                self._send_json(inspect_review_render(service.db, dt, cam))
            except Exception as exc:
                self._send_json({"status": "UNAVAILABLE", "error": str(exc)}, status=409)
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
        elif path == "/api/video_stream":
            fp = params.get("filepath", [""])[0]
            if not fp:
                self.send_error(400, "Missing filepath")
                return
            p = Path(fp).resolve()
            from src.core.config import OUTPUT_DIR
            is_in_output = False
            try:
                is_in_output = p.is_relative_to(OUTPUT_DIR.resolve())
            except Exception:
                is_in_output = False

            if not (is_in_output or service.is_registered_filepath(str(p))):
                self.send_error(403, "Access to specified video path is forbidden")
                return
            self._send_video_stream(p)
            return
        self.send_error(404, "Endpoint not found")

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        post_data = self.rfile.read(min(length, 64 * 1024)) if length > 0 else b"{}"
        if not self._trusted_request(require_token=True):
            self._send_json({"error": "Untrusted request origin"}, status=403)
            return
        parsed = urlparse(self.path)
        path = parsed.path
        if length > 64 * 1024:
            self._send_json({"error": "Request body too large"}, status=413)
            return
        if not self.headers.get("Content-Type", "").lower().startswith("application/json"):
            self._send_json({"error": "Content-Type must be application/json"}, status=415)
            return

        try:
            payload = json.loads(post_data.decode("utf-8"))
        except Exception:
            self._send_json({"error": "Invalid JSON"}, status=400)
            return

        initialize_service()
        if path.startswith("/api/watch_"):
            self._watch_post(path, payload)
            return
        if path == "/api/index_visual_source":
            try:
                segment_id = int(payload.get("segment_id",0))
                segment = service.db.get_segment_by_id(segment_id)
                if not segment:
                    raise ValueError("Unknown segment")
                from src.core.visual_index import index_source
                self._send_json({"success":True,**index_source(service.db,segment["filepath"])})
            except (ValueError,TypeError) as exc:
                self._send_json({"error":str(exc)},status=400)
            except Exception as exc:
                logger.exception("Visual source indexing failed")
                self._send_json({"error":str(exc)},status=503)
            return
        elif path == "/api/review":
            seg_id = payload.get("segment_id")
            label = payload.get("manual_label")
            notes = payload.get("notes", "")
            scenario = payload.get("scenario", "")
            if "review_basis" in payload:
                from scripts.audit_tool.service import REVIEW_BASES
                basis = payload["review_basis"]
                if not isinstance(basis, str) or basis not in REVIEW_BASES or "scenario" not in payload or "notes" not in payload:
                    self._send_json({"error": "review_basis requires a valid value and explicit scenario/notes"}, status=400)
                    return
            if not seg_id or not label:
                self._send_json({"error": "Missing segment_id or manual_label"}, status=400)
                return
            ok = service.submit_review(
                int(seg_id),
                str(label),
                notes=str(notes),
                scenario=str(scenario),
                preserve_attribution="notes" not in payload and "scenario" not in payload,
                review_basis=payload.get("review_basis"),
            )
            self._send_json({"success": ok})
            return

        elif path == "/api/batch_review":
            if "review_basis" in payload:
                self._send_json({"error": "Save review_basis individually; batch decisions preserve each existing basis"}, status=400)
                return
            seg_ids = payload.get("segment_ids", [])
            label = payload.get("manual_label")
            notes = payload.get("notes", "")
            scenario = payload.get("scenario", "")
            if not seg_ids or not label or not isinstance(seg_ids, list):
                self._send_json({"error": "Missing segment_ids list or manual_label"}, status=400)
                return
            count = service.submit_batch_review(
                [int(sid) for sid in seg_ids],
                str(label),
                notes=str(notes),
                scenario=str(scenario),
                preserve_attribution="notes" not in payload and "scenario" not in payload,
            )
            self._send_json({"success": count > 0, "count": count})
            return

        elif path == "/api/clear_review":
            seg_id = payload.get("segment_id")
            if not seg_id:
                self._send_json({"error": "Missing segment_id"}, status=400)
                return
            ok = service.clear_review(int(seg_id))
            self._send_json({"success": ok})
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
            snapshot = payload.get("expected_snapshot")
            if snapshot is not None and (not isinstance(snapshot, str) or len(snapshot) != 64
                                         or any(char not in "0123456789abcdef" for char in snapshot)):
                self._send_json({"error": "Invalid review snapshot"}, status=400)
                return
            res = rerender_mgr.start_rerender(service.db, dt, cam, output_version=version, expected_snapshot=snapshot)
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

        elif path == "/api/exclude_file":
            fp = payload.get("filepath", "")
            reason = payload.get("reason", "MANUAL_EXCLUDED")
            if not fp or not service.is_registered_filepath(fp):
                self._send_json({"error": "Invalid or unregistered filepath"}, status=400)
                return
            try:
                ok = service.exclude_file(fp, reason=reason)
            except RuntimeError as exc:
                self._send_json({"error": str(exc)}, status=409)
                return
            self._send_json({"success": ok})
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
    parser.add_argument("--db", help="Explicit isolated database; default uses the normal workbench database")
    parser.add_argument("--watch-output", help="Parent media and independent watch_exports output directory")
    parser.add_argument("--watch-parent-root", help="Explicit frozen parent directory; exports still use normal output root")
    args = parser.parse_args()
    global service
    from src.database import VlogDatabase
    service = AuditService(VlogDatabase(args.db)) if args.db else AuditService()
    if args.watch_output:
        service.watch_output_root = Path(args.watch_output)
    if args.watch_parent_root:
        service.watch_parent_root = Path(args.watch_parent_root)


    port = find_free_port(args.port)
    server_addr = ("127.0.0.1", port)
    httpd = ThreadingHTTPServer(server_addr, AuditHandler)
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
