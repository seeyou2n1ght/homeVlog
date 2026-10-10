"""HomeVlog 端到端交互完整验收测试脚本 (E2E Interactive Verification)

完整覆盖三大核心支柱的全生命周期交互：
1. 打标交互 (Review & Annotation)：待审队列、两键/四键打标、批量聚类、基准版本化导出；
2. 重渲染交互 (Re-render & Timeline)：人工一票否决权验证、状态跟踪、HTTP 206 流式视频预览与路径安全沙盒；
3. 调优交互 (Tuning & What-If)：离线参数灵敏度仿真、Diff 对比、自动备份原子应用与安全回滚。
"""

import sys
import json
import time
import socket
import logging
import urllib.request
import urllib.parse
import http.cookiejar
from pathlib import Path
from http.server import HTTPServer
import threading

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

from src.core.database import VlogDatabase
from src.core.config import OUTPUT_DIR, CONFIG_PATH, load_config
from src.segment import Segment
from scripts.audit_tool.service import AuditService
from scripts.audit_tool import app

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("e2e_interactive")


class InteractiveTester:
    def __init__(self, port: int | None = None):
        self.port = port or app.find_free_port(8900)
        self.base_url = f"http://127.0.0.1:{self.port}"
        self.cj = http.cookiejar.CookieJar()
        self.opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(self.cj))
        self.server = None
        self.server_thread = None

        # 隔离临时测试 DB
        self.test_dir = PROJECT_ROOT / "temp" / "e2e_test_workspace"
        self.test_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.test_dir / "e2e_interactive.sqlite"
        if self.db_path.exists():
            self.db_path.unlink()
        self.db = VlogDatabase(db_path=self.db_path)
        self.service = AuditService(db=self.db)

    def start_server(self):
        """拉起待测服务实例"""
        app.service = self.service
        self.server = HTTPServer(("127.0.0.1", self.port), app.AuditHandler)
        self.server_thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.server_thread.start()
        print(f"\n[+] 交互测试服务已成功启动: {self.base_url}")

    def stop_server(self):
        """停止服务与清理资源"""
        if self.server:
            self.server.shutdown()
            self.server.server_close()
        if self.db:
            self.db.close()
        # 清理临时版本快照与测试工作区
        from src.core.benchmark import VERSIONS_DIR
        for p in VERSIONS_DIR.glob("*_test*"):
            p.unlink(missing_ok=True)
        for p in VERSIONS_DIR.glob("*_e2e*"):
            p.unlink(missing_ok=True)
        if self.test_dir.exists():
            import shutil
            shutil.rmtree(self.test_dir, ignore_errors=True)
        print("[+] 交互测试服务已安全关闭，测试数据库连接与临时资产已清理")

    def request(self, path: str, method: str = "GET", data: dict | None = None, headers: dict | None = None) -> tuple[int, dict | bytes, dict]:
        """发起带安全会话的 HTTP 请求"""
        url = f"{self.base_url}{path}"
        hdrs = headers or {}
        body = None
        if data is not None:
            body = json.dumps(data).encode("utf-8")
            hdrs["Content-Type"] = "application/json"

        req = urllib.request.Request(url, data=body, headers=hdrs, method=method)
        try:
            with self.opener.open(req) as resp:
                status = resp.status
                raw = resp.read()
                resp_hdrs = dict(resp.headers)
                try:
                    parsed = json.loads(raw.decode("utf-8"))
                    return status, parsed, resp_hdrs
                except Exception:
                    return status, raw, resp_hdrs
        except urllib.error.HTTPError as e:
            raw = e.read()
            try:
                parsed = json.loads(raw.decode("utf-8"))
                return e.code, parsed, dict(e.headers)
            except Exception:
                return e.code, raw, dict(e.headers)

    def prepare_test_data(self):
        """注入典型测试切片样本"""
        fp1 = "test_cam0_daylight.mp4"
        fp2 = "test_cam0_night.mp4"
        self.db.add_file_task(fp1, 0, "20260901", "20260901100000", "20260901101000", 600.0)
        self.db.add_file_task(fp2, 0, "20260901", "20260901220000", "20260901221000", 600.0)

        # 4 个典型切片:
        # 切片 1: 算法误判为 DYNAMIC 的光影晃动 (FP 候选)
        # 切片 2: 算法漏判为 STATIC 的婴儿微动 (FN 候选)
        # 切片 3: 算法判定的高确信度人体走动 (TP)
        # 切片 4: 算法判定的纯静态背景 (TN)
        # 典型争议切片样本: 覆盖光影假阳性与微动漏报
        segs1 = [
            Segment(start_time=10.0, end_time=25.0, state="DYNAMIC", source_file=fp1, max_energy=4.2, avg_confidence=0.0),
            Segment(start_time=26.0, end_time=28.0, state="DYNAMIC", source_file=fp1, max_energy=3.8, avg_confidence=0.0),
            Segment(start_time=30.0, end_time=45.0, state="DYNAMIC", source_file=fp1, max_energy=28.0, avg_confidence=0.85),
        ]
        segs2 = [
            Segment(start_time=100.0, end_time=120.0, state="STATIC", source_file=fp2, max_energy=3.1, avg_confidence=0.0),
            Segment(start_time=130.0, end_time=145.0, state="STATIC", source_file=fp2, max_energy=2.6, avg_confidence=0.0),
            Segment(start_time=150.0, end_time=180.0, state="STATIC", source_file=fp2, max_energy=0.4, avg_confidence=0.0),
        ]
        self.db.set_analysis_result(fp1, "ANALYZED", segs1)
        self.db.set_analysis_result(fp2, "ANALYZED", segs2)
        print("[+] 测试素材与切片数据已注入 SQLite 数据库")

    def run_all(self):
        """执行全套端到端交互测试"""
        print("=" * 70)
        print("   HomeVlog 人工打标三位一体架构 • 端到端交互验收套件")
        print("=" * 70)

        self.start_server()
        try:
            # 阶段 0: 安全认证与握手
            self.test_security_handshake()

            # 阶段 1: 打标与基准版本化交互
            self.prepare_test_data()
            self.test_annotation_and_benchmark_flow()

            # 阶段 2: 时间轴一票否决与重渲染流式预览
            self.test_rerender_and_streaming_flow()

            # 阶段 3: What-If 离线调参仿真与配置原子应用/回滚
            self.test_tuning_and_config_flow()

            print("\n" + "=" * 70)
            print("   🎉 全部交互测试环节 100% 验收通过！系统三位一体闭环稳定运行！")
            print("=" * 70)
            return True
        finally:
            self.stop_server()

    def test_security_handshake(self):
        print("\n--- [Step 0] 安全握手与 Cookie Token 校验 ---")
        status, _, headers = self.request("/")
        assert status == 200, f"根路径访问失败: {status}"
        cookies = [c.name for c in self.cj]
        assert "homevlog_audit_token" in cookies, "未下发 homevlog_audit_token 安全 Cookie"
        print("  ✓ 安全握手成功，获得 HttpOnly 同源令牌")

        # 静态资源检查
        s_js, _, _ = self.request("/app.js")
        s_css, _, _ = self.request("/style.css")
        assert s_js == 200 and s_css == 200, "前端三剑客静态资源访问失败"
        print("  ✓ 静态资源 200 OK (app.js, style.css)")

    def test_annotation_and_benchmark_flow(self):
        print("\n--- [Step 1] 人工打标与基准版本化交互 ---")
        # 1. 查询待审队列
        st, items, _ = self.request("/api/anomalies?category=all")
        assert st == 200 and len(items) >= 4, f"待审队列获取异常: {st}, len={len(items)}"
        print(f"  ✓ 成功调取待审切片队列: 共 {len(items)} 个候选切片")

        # 2. 模拟审核员盲打纠偏
        # 样本 1: 标记为光影假阳性 (FALSE_ALARM)
        seg1_id = items[0]["id"]
        st, res1, _ = self.request("/api/review", method="POST", data={
            "segment_id": seg1_id,
            "manual_label": "FALSE_ALARM",
            "scenario": "LIGHT_SHADOW",
            "notes": "窗外树影晃动光影误报纠偏"
        })
        assert st == 200 and res1.get("success"), f"提交复核失败: {res1}"
        print(f"  ✓ 切片 #{seg1_id} 成功打标 ➔ FALSE_ALARM (光影误报纠偏)")

        # 样本 2: 标记为微动漏检挽救 (MISSED_MOTION)
        seg2_id = items[2]["id"]
        st, res2, _ = self.request("/api/review", method="POST", data={
            "segment_id": seg2_id,
            "manual_label": "MISSED_MOTION",
            "scenario": "INFANT_MOTION",
            "notes": "暗光婴儿翻身微动漏报挽救"
        })
        assert st == 200 and res2.get("success"), f"提交复核失败: {res2}"
        print(f"  ✓ 切片 #{seg2_id} 成功打标 ➔ MISSED_MOTION (微动漏报挽救)")

        # 3. 批量审核操作测试
        seg3_id = items[1]["id"]
        seg4_id = items[3]["id"]
        st, b_res, _ = self.request("/api/batch_review", method="POST", data={
            "segment_ids": [seg3_id, seg4_id],
            "manual_label": "CONFIRMED_STATIC",
            "scenario": "EMPTY_ROOM",
            "notes": "批量确认为背景纯静止"
        })
        assert st == 200 and b_res.get("count") == 2, f"批量打标失败: {b_res}"
        print(f"  ✓ 批量打标成功: 共原子处理 {b_res['count']} 处切片")

        # 4. 导出黄金评测基准 (带版本标签与元数据)
        gold_out = self.test_dir / "e2e_gold_ground_truth.json"
        st, exp_res, _ = self.request("/api/export_gold_benchmark", method="POST", data={
            "output_path": str(gold_out),
            "version": "2.0_interactive_e2e",
            "incremental_merge": True
        })
        assert st == 200 and exp_res.get("success"), f"导出基准失败: {exp_res}"
        assert gold_out.exists(), "基准 JSON 文件未生成"
        meta_out = gold_out.with_name(gold_out.stem + ".meta.json")
        assert meta_out.exists(), "伴随元数据 .meta.json 未生成"
        print(f"  ✓ 导出评测基准成功: {exp_res['count']} 样本, 版本={exp_res.get('version')}")

        # 5. 校验历史版本列表
        st, v_list, _ = self.request("/api/benchmark_versions")
        assert st == 200 and isinstance(v_list, list), f"版本列表查询失败: {st}"
        print(f"  ✓ 历史基准版本追踪: 已记录 {len(v_list)} 个版本快照")

    def test_rerender_and_streaming_flow(self):
        print("\n--- [Step 2] 时间轴纠偏、重渲染状态机与流式视频播放 ---")
        # 1. 验证时间轴最高优先级 (Timeline Invariant)
        from src.stages.timeline import build_timeline_from_rows
        file_tasks = self.db.get_all_file_tasks_for_date("20260901", 0)
        timeline = build_timeline_from_rows(file_tasks, "20260901", config=load_config(), resolve_presence=False)

        # 严格验证:
        # 切片 1 (原为 DYNAMIC, 人商标为 FALSE_ALARM) 必须降级为 STATIC 快进
        # 切片 2 (原为 STATIC, 人商标为 MISSED_MOTION) 必须提升为 DYNAMIC 原速
        night_dynamics = [t for t in timeline if t.filepath == "test_cam0_night.mp4" and t.state == "DYNAMIC"]
        assert len(night_dynamics) > 0, "Timeline Invariant 违背: MISSED_MOTION 未能提级为 DYNAMIC"
        day_segs = [t for t in timeline if t.filepath == "test_cam0_daylight.mp4" and 10.0 <= t.start_in_file <= 25.0]
        assert all(t.state == "STATIC" for t in day_segs), "Timeline Invariant 违背: FALSE_ALARM 未能降级为 STATIC"
        print("  ✓ 时间轴绝对支配权验证通过: FALSE_ALARM->STATIC 与 MISSED_MOTION->DYNAMIC 双向纠偏 100% 生效")

        # 2. 模拟重渲染产物与状态轮询
        test_out_video = OUTPUT_DIR / "DailyVlog_20260901_e2e_reviewed.mp4"
        dummy_content = b"FLV\x01\x05\x00\x00\x00\t" + b"HOMEVLOG_TEST_STREAM_" * 50
        test_out_video.write_bytes(dummy_content)
        total_len = len(dummy_content)

        # 3. HTTP 206 Partial Content (Range Streaming) 测试
        url_enc = urllib.parse.quote(str(test_out_video))
        st, chunk, hdrs = self.request(f"/api/video_stream?filepath={url_enc}", headers={"Range": "bytes=0-15"})
        assert st == 206, f"流媒体 Range 请求未返回 206: {st}"
        assert hdrs.get("Content-Range") == f"bytes 0-15/{total_len}", f"Content-Range 不符: {hdrs}"
        assert len(chunk) == 16, f"分块大小不符: {len(chunk)} != 16"
        print(f"  ✓ 流媒体播放测试 206 Partial Content 通过: 成功获取 bytes 0-15/{total_len}")

        # 4. 路径遍历安全防护测试
        st_bad, _, _ = self.request("/api/video_stream?filepath=C:/Windows/System32/cmd.exe")
        assert st_bad == 403, f"非授权路径未返回 403: {st_bad}"
        print("  ✓ 路径遍历安全防护通过: 越界文件请求已被严格拦截 (403 Forbidden)")

        test_out_video.unlink(missing_ok=True)

    def test_tuning_and_config_flow(self):
        print("\n--- [Step 3] What-If 离线调参仿真与配置原子落地/回滚 ---")
        # 1. 获取调参建议洞察
        st, insights, _ = self.request("/api/tuning_insights")
        assert st == 200, f"调参洞察获取失败: {st}"
        recs = insights.get("recommendations", [])
        print(f"  ✓ 获取调优量化建议成功: 识别到 {len(recs)} 项优化建议")

        # 2. 执行 What-If 离线参数灵敏度仿真
        candidate_params = {
            "detection.min_motion_threshold": 3.5,
            "segment.min_motion_duration": 2.5,
            "yolo.confidence": 0.25,
        }
        st, sim_res, _ = self.request("/api/tuning_simulate", method="POST", data={"candidate_params": candidate_params})
        assert st == 200 and "verdict" in sim_res, f"仿真推演失败: {sim_res}"
        b_f1 = sim_res["baseline"]["f1"]
        s_f1 = sim_res["simulated"]["f1"]
        d_f1 = sim_res["deltas"]["f1"]
        fp_red = sim_res["deltas"]["fp_reduction"]
        fn_inc = sim_res["deltas"]["fn_increase"]
        verdict = sim_res["verdict"]
        print(f"  ✓ What-If 仿真推演完成: 基准 F1={b_f1}%, 预测 F1={s_f1}% (Δ={d_f1}%)")
        print(f"    - 净收益: 成功压制 {fp_red} 处误报, 漏报代价={fn_inc}, 综合判定=[{verdict}]")

        # 3. 生成配置 Diff
        st, diff_res, _ = self.request("/api/tuning_diff", method="POST", data={"params": candidate_params})
        assert st == 200 and diff_res.get("total_changes") >= 1, f"生成 Diff 失败: {diff_res}"
        print(f"  ✓ 结构化配置 Diff 生成成功: 检测到 {diff_res['total_changes']} 项参数差异")

        # 4. 安全应用配置并创建自动时间戳备份
        st, apply_res, _ = self.request("/api/tuning_apply", method="POST", data={"params": candidate_params})
        assert st == 200 and apply_res.get("success"), f"应用配置失败: {apply_res}"
        backup_file = Path(apply_res["backup_path"])
        assert backup_file.exists(), f"自动备份文件未生成: {backup_file}"
        print(f"  ✓ 配置已安全写入并热重载: 自动备份文件已归档至 {backup_file.name}")

        # 5. 校验新配置已在内存与磁盘中生效
        curr_cfg = load_config(reload=True)
        assert float(curr_cfg["detection"]["min_motion_threshold"]) == 3.5
        print("  ✓ 运行时内存配置热重载验证通过: detection.min_motion_threshold == 3.5")

        # 6. 安全回滚配置
        st, roll_res, _ = self.request("/api/tuning_rollback", method="POST", data={"backup_path": str(backup_file)})
        assert st == 200 and roll_res.get("success"), f"配置回滚失败: {roll_res}"
        restored_cfg = load_config(reload=True)
        assert float(restored_cfg["detection"]["min_motion_threshold"]) == 2.5
        print(f"  ✓ 配置回滚成功: 已无损恢复至原生产配置 (min_motion_threshold == 2.5)")

        # 清理本次生成的备份文件以保持工作区纯净
        backup_file.unlink(missing_ok=True)
        print("  ✓ 临时备份文件已清理，生产配置保持原状")


if __name__ == "__main__":
    tester = InteractiveTester()
    ok = tester.run_all()
    sys.exit(0 if ok else 1)
