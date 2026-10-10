"""测试解码流完整性与截断判定策略 (Stream Completion Policy) 纯函数边界"""

import pytest
from src.stages.detector import evaluate_stream_completion, DecodeStreamDecision


def test_evaluate_stream_completion_normal():
    # 60s 文件，5 fps，预期 300 帧。实际解码 300 帧，正常退出
    decision = evaluate_stream_completion(
        total_frames=300,
        file_duration=60.0,
        effective_fps=5.0,
        returncode=0,
    )
    assert decision.complete is True
    assert decision.truncated is False
    assert decision.corrupted_tail is False
    assert decision.reason == "NORMAL_COMPLETION"


def test_evaluate_stream_completion_truncated_clean_eof():
    # 600s 文件，5 fps，预期 3000 帧 (85% 门限为 2550 帧)。
    # 实际仅录制 58s (290 帧) 干净提前 EOF (returncode=0)
    decision = evaluate_stream_completion(
        total_frames=290,
        file_duration=600.0,
        effective_fps=5.0,
        returncode=0,
    )
    assert decision.complete is True
    assert decision.truncated is True
    assert decision.corrupted_tail is False
    assert decision.reason == "TRUNCATED_EOF"


def test_evaluate_stream_completion_corrupted_tail_recovery():
    # 录制到一半硬件断电导致尾部残卷，抛出解码异常但保留了 500 个有效帧
    decision = evaluate_stream_completion(
        total_frames=500,
        file_duration=600.0,
        effective_fps=5.0,
        returncode=-1,
        has_tail_error=True,
    )
    assert decision.complete is True
    assert decision.truncated is True
    assert decision.corrupted_tail is True
    assert decision.reason == "CORRUPTED_TAIL"


def test_evaluate_stream_completion_insufficient_frames():
    # 彻底损坏原片，仅读出 0 帧或 1 帧
    for frames in (0, 1):
        decision = evaluate_stream_completion(
            total_frames=frames,
            file_duration=60.0,
            effective_fps=5.0,
            returncode=0,
        )
        assert decision.complete is False
        assert decision.reason == "INSUFFICIENT_FRAMES"


def test_evaluate_stream_completion_interrupted():
    # 用户按 Ctrl+C 或管线提前中断
    decision = evaluate_stream_completion(
        total_frames=500,
        file_duration=60.0,
        effective_fps=5.0,
        is_interrupted=True,
    )
    assert decision.complete is False
    assert decision.reason == "INTERRUPTED"

    decision_aborted = evaluate_stream_completion(
        total_frames=500,
        file_duration=60.0,
        effective_fps=5.0,
        is_aborted=True,
    )
    assert decision_aborted.complete is False
    assert decision_aborted.reason == "INTERRUPTED"


def test_evaluate_stream_completion_process_error():
    # 解码进程非零退出且帧数严重不足，无尾部残卷容错保护
    decision = evaluate_stream_completion(
        total_frames=10,
        file_duration=60.0,
        effective_fps=5.0,
        returncode=1,
        has_tail_error=False,
    )
    assert decision.complete is False
    assert decision.reason == "PROCESS_ERROR"
