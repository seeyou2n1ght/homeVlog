import numpy as np
import cv2

class BlockMotionDetector:
    """
    轻量级自适应分块帧差分析器 (V2.4)。
    职责: 通过动态背景均值建模，过滤红外噪声和全局光变，精确捕捉婴儿缓慢微动。
    """
    def __init__(self, block_size: int = 16, history_size: int = 20, threshold_factor: float = 2.5, min_change_blocks: int = 3):
        self.block_size = block_size
        self.history_size = history_size
        self.k = threshold_factor
        self.min_change_blocks = min_change_blocks
        
        self.history_variances = []
        self.baseline_variances = None
        self.background_means = None

    def _rgb2gray(self, frame: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    def analyze(self, bgr_frame: np.ndarray) -> bool:
        """
        分析一帧，返回是否存在有效的像素级运动。
        """
        gray = self._rgb2gray(bgr_frame)
        h, w = gray.shape
        
        # 裁剪边缘以严格适应块大小
        h_crop = h - (h % self.block_size)
        w_crop = w - (w % self.block_size)
        gray = gray[:h_crop, :w_crop]
        
        num_blocks_h = h_crop // self.block_size
        num_blocks_w = w_crop // self.block_size
        total_blocks = num_blocks_h * num_blocks_w
        
        # 内存安全地展开矩阵: [num_blocks_h * num_blocks_w, block_size * block_size]
        blocks = gray.reshape(num_blocks_h, self.block_size, num_blocks_w, self.block_size)
        blocks = blocks.transpose(0, 2, 1, 3)
        blocks = blocks.reshape(-1, self.block_size * self.block_size)
        
        # 使用 float32 避免计算溢出
        blocks_f = blocks.astype(np.float32)
        means = np.mean(blocks_f, axis=1)
        variances = np.var(blocks_f, axis=1)
        
        # 初始建模阶段
        if self.baseline_variances is None:
            self.baseline_variances = variances.copy()
            self.background_means = means.copy()
            self.history_variances.append(variances)
            return False

        # 计算块均值与【滑动背景均值】的绝对变化量，以对抗低帧率极缓运动
        mean_diffs = np.abs(means - self.background_means)
        
        # 动态阈值: max(绝对低频阈值 10.0, k * sqrt(基线方差))
        dynamic_threshold = np.maximum(10.0, self.k * np.sqrt(self.baseline_variances + 1e-5))
        
        changed_blocks = np.sum((mean_diffs > dynamic_threshold))
        
        # 全局光照屏蔽策略: 如果突变面积超过 40%，必定为光变（如开灯/闪烁），直接重置背景
        if changed_blocks > total_blocks * 0.4:
            self.background_means = means.copy()
            return False
            
        motion_detected = bool(changed_blocks >= self.min_change_blocks)
        
        # 背景平滑更新：如果静止，更新较快；如果运动中，更新极慢（允许缓慢的天色变化，但保留目标运动痕迹）
        alpha = 0.05 if motion_detected else 0.2
        self.background_means = (1.0 - alpha) * self.background_means + alpha * means

        # 环境方差更新：仅在没有发生大规模变化时更新，确保基线不受运动目标干扰
        if changed_blocks < total_blocks * 0.1:
            self.history_variances.append(variances)
            if len(self.history_variances) > self.history_size:
                self.history_variances.pop(0)
            
            # 使用中位值来抵抗历史队列中偶尔包含的小噪点
            self.baseline_variances = np.median(np.stack(self.history_variances), axis=0)
            
        return motion_detected
