import pytest
import av
from av.codec.hwaccel import HWAccel, hwdevices_available

def test_pyav_hwaccel_support():
    # 1. 探查当前可用的硬件加速设备
    devices = hwdevices_available()
    print(f"\n[INFO] Available hardware devices: {devices}")
    assert isinstance(devices, list)

    # 2. 测试 HWAccel 初始化语法
    if "cuda" in devices:
        hw_cuda = HWAccel("cuda")
        assert hw_cuda.device_id is None or isinstance(hw_cuda.device_id, (str, int))
        print("[INFO] Successfully initialized CUDA HWAccel")

    if "qsv" in devices:
        hw_qsv = HWAccel("qsv")
        assert hw_qsv.device_id is None or isinstance(hw_qsv.device_id, (str, int))
        print("[INFO] Successfully initialized QSV HWAccel")

def test_pyav_open_with_hwaccel(tmp_path):
    # 创建一个极小的空文件尝试 av.open，虽然会报错，但主要是为了测试 hwaccel 参数解析是否抛出 "Cannot convert str to av.codec.hwaccel.HWAccel"
    test_file = tmp_path / "dummy.mp4"
    test_file.write_bytes(b"\x00" * 100)
    
    devices = hwdevices_available()
    for hw_name in ["cuda", "qsv"]:
        if hw_name in devices:
            hw = HWAccel(hw_name)
            try:
                # 传入 HWAccel 实例，此时即使因为文件损坏报错，也不应该抛出 "Cannot convert str to av.codec.hwaccel.HWAccel"
                av.open(str(test_file), hwaccel=hw)
            except Exception as e:
                # 过滤掉文件损坏错误，抛出类型转换错误
                err_msg = str(e)
                print(f"[DEBUG] Open with {hw_name} threw: {err_msg}")
                assert "Cannot convert str" not in err_msg
