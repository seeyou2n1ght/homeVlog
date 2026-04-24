import sqlite3
import os
from pathlib import Path
from datetime import datetime, timedelta
import json
import re

# 简单的 HTML 模板，内嵌 Tailwind CSS 和 ApexCharts
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="zh-CN" class="dark">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>HomeVlog V2.4 业务洞察大屏</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://cdn.jsdelivr.net/npm/apexcharts"></script>
    <script>
        tailwind.config = {
            darkMode: 'class',
            theme: {
                extend: {
                    colors: {
                        gray: {
                            900: '#121212',
                            800: '#1e1e1e',
                            700: '#2d2d2d',
                        }
                    }
                }
            }
        }
    </script>
</head>
<body class="bg-gray-900 text-gray-100 min-h-screen font-sans p-6">
    <div class="max-w-7xl mx-auto">
        <header class="mb-8">
            <h1 class="text-3xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-blue-400 to-purple-500">
                HomeVlog 智能洞察大屏
            </h1>
            <p class="text-gray-400 mt-2">数据更新时间: {{ generate_time }}</p>
        </header>

        <!-- KPI 卡片区 -->
        <div class="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
            <div class="bg-gray-800 rounded-xl p-6 border border-gray-700 shadow-lg">
                <h3 class="text-gray-400 text-sm font-medium mb-2">处理总文件数</h3>
                <p class="text-3xl font-bold">{{ total_files }} <span class="text-lg text-gray-500 font-normal">个</span></p>
            </div>
            <div class="bg-gray-800 rounded-xl p-6 border border-gray-700 shadow-lg">
                <h3 class="text-gray-400 text-sm font-medium mb-2">空间释放 (ROI)</h3>
                <p class="text-3xl font-bold text-green-400">{{ space_saved_gb }} <span class="text-lg text-gray-500 font-normal">GB</span></p>
                <p class="text-xs text-gray-500 mt-1">浓缩率 {{ compress_ratio }}%</p>
            </div>
            <div class="bg-gray-800 rounded-xl p-6 border border-gray-700 shadow-lg">
                <h3 class="text-gray-400 text-sm font-medium mb-2">平均吞吐倍速</h3>
                <p class="text-3xl font-bold text-blue-400">{{ avg_speed_x }}<span class="text-lg text-gray-500 font-normal">x</span></p>
            </div>
            <div class="bg-gray-800 rounded-xl p-6 border border-gray-700 shadow-lg">
                <h3 class="text-gray-400 text-sm font-medium mb-2">事件捕获总数</h3>
                <p class="text-3xl font-bold text-purple-400">{{ total_events }} <span class="text-lg text-gray-500 font-normal">次</span></p>
            </div>
        </div>

        <div class="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
            <!-- 业务：高质量陪伴浓度 -->
            <div class="bg-gray-800 rounded-xl p-6 border border-gray-700 shadow-lg">
                <h3 class="text-lg font-medium mb-4">高质量陪伴浓度</h3>
                <div id="chart-companion" class="flex justify-center"></div>
            </div>
            
            <!-- 业务：宝宝活跃雷达图 -->
            <div class="bg-gray-800 rounded-xl p-6 border border-gray-700 shadow-lg">
                <h3 class="text-lg font-medium mb-4">宝宝 24H 活跃雷达图</h3>
                <div id="chart-radar" class="flex justify-center"></div>
            </div>
        </div>

        <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <!-- 性能：算力剪刀差 -->
            <div class="bg-gray-800 rounded-xl p-6 border border-gray-700 shadow-lg">
                <h3 class="text-lg font-medium mb-4">算力水位剪刀差 (FPS)</h3>
                <div id="chart-fps"></div>
            </div>
            
            <!-- 性能：阶段耗时拆解 -->
            <div class="bg-gray-800 rounded-xl p-6 border border-gray-700 shadow-lg">
                <h3 class="text-lg font-medium mb-4">管线耗时拆解 (Phase Breakdown)</h3>
                <div id="chart-phase"></div>
            </div>
        </div>
    </div>

    <script>
        // 渲染陪伴浓度
        new ApexCharts(document.querySelector("#chart-companion"), {
            series: {{ companion_series }},
            labels: ['高质量亲密陪伴', '同框各自独立'],
            chart: { type: 'donut', height: 320, background: 'transparent' },
            theme: { mode: 'dark' },
            colors: ['#a855f7', '#4b5563'],
            stroke: { show: false },
            dataLabels: { enabled: false },
            plotOptions: { pie: { donut: { size: '75%' } } }
        }).render();

        // 渲染宝宝雷达图
        new ApexCharts(document.querySelector("#chart-radar"), {
            series: [{ name: '活跃时长(秒)', data: {{ radar_series }} }],
            chart: { type: 'radar', height: 320, background: 'transparent', toolbar: {show: false} },
            theme: { mode: 'dark' },
            labels: ['0点','','','3点','','','6点','','','9点','','','12点','','','15点','','','18点','','','21点','',''],
            stroke: { width: 2, colors: ['#3b82f6'] },
            fill: { opacity: 0.2, colors: ['#3b82f6'] },
            markers: { size: 0 }
        }).render();

        // 渲染 FPS 剪刀差
        new ApexCharts(document.querySelector("#chart-fps"), {
            series: [
                { name: 'Decode FPS (QSV)', data: {{ decode_fps_series }} },
                { name: 'Infer FPS (TRT)', data: {{ infer_fps_series }} }
            ],
            chart: { type: 'area', height: 320, background: 'transparent', toolbar: {show: false} },
            theme: { mode: 'dark' },
            colors: ['#3b82f6', '#10b981'],
            stroke: { curve: 'smooth', width: 2 },
            fill: { type: 'gradient', gradient: { shadeIntensity: 1, opacityFrom: 0.4, opacityTo: 0.05, stops: [0, 100] } },
            dataLabels: { enabled: false },
            xaxis: { labels: { show: false }, tooltip: { enabled: false } },
            yaxis: { title: { text: 'Frames Per Second' } }
        }).render();

        // 渲染耗时拆解
        new ApexCharts(document.querySelector("#chart-phase"), {
            series: [
                { name: '队列等待 (Wait)', data: {{ phase_wait_series }} },
                { name: '张量推理 (Infer)', data: {{ phase_infer_series }} },
                { name: '物理切割 (I/O)', data: {{ phase_io_series }} }
            ],
            chart: { type: 'bar', height: 320, stacked: true, stackType: '100%', background: 'transparent', toolbar: {show: false} },
            theme: { mode: 'dark' },
            colors: ['#ef4444', '#f59e0b', '#3b82f6'],
            plotOptions: { bar: { horizontal: true } },
            xaxis: { labels: { show: false } },
            yaxis: { labels: { show: false } }
        }).render();
    </script>
</body>
</html>
"""

def parse_time_from_filename(filename: str) -> datetime:
    """提取形如 00_20260411121621_20260411122153.mp4 的时间"""
    m = re.search(r'_(\d{14})_', filename)
    if m:
        return datetime.strptime(m.group(1), "%Y%m%d%H%M%S")
    return datetime.now()

def generate_dashboard(db_path: str, output_path: str):
    if not os.path.exists(db_path):
        print(f"Error: Database {db_path} not found.")
        return
        
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    
    # 1. KPI 统计
    cur.execute("SELECT COUNT(1), SUM(source_size_bytes), SUM(output_size_bytes) FROM processed_files WHERE status='completed'")
    total_files, source_size, out_size = cur.fetchone()
    total_files = total_files or 0
    source_size = source_size or 0
    out_size = out_size or 0
    
    space_saved_gb = max(0, source_size - out_size) / (1024**3)
    compress_ratio = ((source_size - out_size) / source_size * 100) if source_size > 0 else 0
    
    cur.execute("SELECT AVG(decode_fps), COUNT(1) FROM performance_metrics")
    avg_fps, total_events_raw = cur.fetchone()
    avg_speed_x = (avg_fps / 20.0) if avg_fps else 0 # 假设原视频是20fps
    
    cur.execute("SELECT COUNT(1) FROM analytics_events")
    total_events = cur.fetchone()[0] or 0

    # 2. 高质量陪伴浓度
    cur.execute("SELECT SUM(duration_ms) FROM analytics_events WHERE avg_distance >= 0 AND avg_distance < 0.2")
    close_duration = cur.fetchone()[0] or 0
    cur.execute("SELECT SUM(duration_ms) FROM analytics_events WHERE avg_distance >= 0.2")
    far_duration = cur.fetchone()[0] or 0
    companion_series = [close_duration, far_duration] if (close_duration + far_duration) > 0 else [1, 1] # Fallback
    
    # 3. 宝宝雷达图
    radar_data = [0] * 24
    cur.execute("SELECT f.file_name, a.start_pts, a.duration_ms FROM analytics_events a JOIN processed_files f ON a.file_id = f.id WHERE a.classes_detected LIKE '%baby%'")
    for fname, start_pts, dur_ms in cur.fetchall():
        base_time = parse_time_from_filename(fname)
        event_time = base_time + timedelta(seconds=start_pts)
        radar_data[event_time.hour] += (dur_ms / 1000.0)
        
    # 4. 性能双折线
    cur.execute("SELECT decode_fps, infer_fps FROM performance_metrics ORDER BY id DESC LIMIT 50")
    rows = cur.fetchall()
    decode_fps_series = [r[0] for r in rows][::-1]
    infer_fps_series = [r[1] for r in rows][::-1]
    
    # 5. 阶段耗时
    cur.execute("SELECT phase_wait_ms, phase_infer_ms, phase_io_cut_ms FROM performance_metrics ORDER BY id DESC LIMIT 20")
    rows = cur.fetchall()
    phase_wait_series = [r[0] for r in rows][::-1]
    phase_infer_series = [r[1] for r in rows][::-1]
    phase_io_series = [r[2] for r in rows][::-1]

    # 组装 HTML
    html = HTML_TEMPLATE.replace("{{ generate_time }}", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    html = html.replace("{{ total_files }}", str(total_files))
    html = html.replace("{{ space_saved_gb }}", f"{space_saved_gb:.2f}")
    html = html.replace("{{ compress_ratio }}", f"{compress_ratio:.1f}")
    html = html.replace("{{ avg_speed_x }}", f"{avg_speed_x:.1f}")
    html = html.replace("{{ total_events }}", str(total_events))
    html = html.replace("{{ companion_series }}", json.dumps(companion_series))
    html = html.replace("{{ radar_series }}", json.dumps([round(x,1) for x in radar_data]))
    html = html.replace("{{ decode_fps_series }}", json.dumps(decode_fps_series))
    html = html.replace("{{ infer_fps_series }}", json.dumps(infer_fps_series))
    html = html.replace("{{ phase_wait_series }}", json.dumps(phase_wait_series))
    html = html.replace("{{ phase_infer_series }}", json.dumps(phase_infer_series))
    html = html.replace("{{ phase_io_series }}", json.dumps(phase_io_series))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"✅ Dashboard generated at: {output_path}")

if __name__ == "__main__":
    db = str(Path(__file__).parent.parent.parent.parent / "data" / "homevlog.db")
    out = str(Path(__file__).parent.parent.parent.parent / "outputs" / f"dashboard_{datetime.now().strftime('%Y%m%d')}.html")
    generate_dashboard(db, out)
