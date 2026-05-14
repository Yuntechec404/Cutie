#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import cv2
import torch
import numpy as np
import pyrealsense2 as rs
from torchvision.transforms import ToTensor
from ultralytics import SAM

# 引入 Cutie 相關模組
from cutie.inference.inference_core import InferenceCore
from cutie.utils.get_default_model import get_default_model

def get_realsense_intrinsics_matrix(video_stream_profile):
    """將 RealSense 內參轉換為 3x3 Numpy 矩陣"""
    intrinsics = video_stream_profile.get_intrinsics()
    K = np.array([
        [intrinsics.fx, 0, intrinsics.ppx],
        [0, intrinsics.fy, intrinsics.ppy],
        [0, 0, 1]
    ], dtype=np.float32)
    return K

# ==========================================
# 互動式 SAM 2 點擊處理類別
# ==========================================
class InteractiveSAM2:
    def __init__(self, model_path="sam2_t.pt"):
        print(f"Loading SAM 2 model from {model_path}...")
        self.model = SAM(model_path)
        self.points = []
        self.labels = []
        self.image = None
        self.mask = None

    def set_image(self, image):
        self.image = image
        self.points = []
        self.labels = []
        self.mask = np.zeros(image.shape[:2], dtype=np.uint8)

    def click_callback(self, event, x, y, flags, param):
        """處理滑鼠點擊事件"""
        if event == cv2.EVENT_LBUTTONDOWN:     # 左鍵：新增前景點
            self.points.append([x, y])
            self.labels.append(1)
            self.predict()
        elif event == cv2.EVENT_RBUTTONDOWN:   # 右鍵：新增背景點
            self.points.append([x, y])
            self.labels.append(0)
            self.predict()
        elif event == cv2.EVENT_MBUTTONDOWN:   # 中鍵：清除所有點
            self.points = []
            self.labels = []
            self.mask = np.zeros(self.image.shape[:2], dtype=np.uint8)

    def predict(self):
        """執行 SAM 2 推論"""
        if not self.points:
            return
        results = self.model(self.image, points=self.points, labels=self.labels, verbose=False)
        if results[0].masks is not None:
            self.mask = results[0].masks.data[0].cpu().numpy()

    def draw(self, vis_image):
        """將 Mask 與點繪製到影像上"""
        if self.mask is not None and np.any(self.mask):
            vis_image[self.mask > 0.5] = vis_image[self.mask > 0.5] * 0.6 + np.array([0, 255, 0]) * 0.4
        
        for pt, lbl in zip(self.points, self.labels):
            color = (0, 255, 0) if lbl == 1 else (0, 0, 255)
            cv2.circle(vis_image, (pt[0], pt[1]), 5, color, -1)
        return vis_image


@torch.inference_mode()
def main():
    # ==========================================
    # 0. 實驗與功能開關設定 (Ablation Flags)
    # ==========================================
    ENABLE_KINEMATICS_MEMORY = True   
    ENABLE_DEPTH_WARP        = True   
    ENABLE_EXPLICIT_SHIFT    = True   
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 1. 載入 Cutie 模型 (只載入一次網路權重)
    cutie = get_default_model()
    print("Cutie model weights loaded successfully.")

    # 2. 載入 SAM 2 模型
    sam2_interactor = InteractiveSAM2("sam2_t.pt")

    # 3. 初始化 RealSense
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    
    print("Starting RealSense camera...")
    profile = pipeline.start(config)
    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
    align = rs.align(rs.stream.color)
    
    K_matrix = get_realsense_intrinsics_matrix(profile.get_stream(rs.stream.color).as_video_stream_profile())
    K_tensor = torch.from_numpy(K_matrix).unsqueeze(0).to(device)

    to_tensor = ToTensor()
    window_name = 'RealSense Cutie VOS'
    cv2.namedWindow(window_name)

    try:
        while True:
            # 每次重測時，必須產生全新的 InferenceCore 來清空舊物件的記憶！
            processor = InferenceCore(cutie, cfg=cutie.cfg)
            
            # 清空 SAM 2 過去的點擊紀錄
            sam2_interactor.points = []
            sam2_interactor.labels = []
            sam2_interactor.mask = None

            print("\n" + "="*50)
            print("【階段零：待機預覽】")
            print("[SPACE 空白鍵]: 凍結畫面並選擇目標")
            print("[q 鍵]: 離開程式")

            first_color_image = None
            first_depth_image = None
            
            quit_app = False
            restart_test = False

            # ------------------------------------------
            # 階段零：待機預覽
            # ------------------------------------------
            while True:
                frames = pipeline.wait_for_frames()
                aligned_frames = align.process(frames)
                color_frame = aligned_frames.get_color_frame()
                depth_frame = aligned_frames.get_depth_frame()
                
                if not color_frame or not depth_frame:
                    continue

                preview_color = np.asanyarray(color_frame.get_data())
                preview_depth = np.asanyarray(depth_frame.get_data())
                
                cv2.putText(preview_color, "Preview: Press SPACE to freeze | Press 'q' to quit", 
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                cv2.imshow(window_name, preview_color)

                key = cv2.waitKey(1)
                if key == 32:  # SPACE 空白鍵
                    first_color_image = preview_color.copy()
                    first_depth_image = preview_depth.copy()
                    print(">> 畫面已凍結！進入 SAM 2 選擇模式。")
                    break
                elif key == ord('q'): # q 鍵離開
                    quit_app = True
                    break
            
            if quit_app: break

            # ------------------------------------------
            # 階段一：SAM 2 初始化
            # ------------------------------------------
            sam2_interactor.set_image(first_color_image)
            cv2.setMouseCallback(window_name, sam2_interactor.click_callback)

            print("\n" + "="*50)
            print("階段一：SAM 2 初始化】")
            print("滑鼠左/右鍵 : 選擇 前景/背景")
            print("[ENTER]: 確認 Mask 並開始追蹤")
            print("[ESC]: 放棄當前畫面，重新回到待機預覽")
            print("[q 鍵]: 離開程式")

            while True:
                vis_image = first_color_image.copy()
                vis_image = sam2_interactor.draw(vis_image)
                
                cv2.putText(vis_image, "SAM2: Click to mask | ENTER to track | ESC to restart | 'q' to quit", 
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.imshow(window_name, vis_image)
                
                key = cv2.waitKey(15)
                if key == 13:  # ENTER 鍵
                    if len(sam2_interactor.points) == 0:
                        print("尚未選擇任何點！請先點擊目標。")
                    else:
                        break
                elif key == 27:  # ESC 鍵
                    print(">> 取消選擇，回到待機預覽。")
                    restart_test = True
                    break
                elif key == ord('q'): # q 鍵離開
                    quit_app = True
                    break

            # 移除滑鼠監聽器
            cv2.setMouseCallback(window_name, lambda *args: None)
            
            if quit_app: break
            if restart_test: continue # 跳出當前迴圈，回到最外層重新開始

            print("Mask confirmed. Initializing Cutie tracker...")

            # ------------------------------------------
            # 階段二：初始化 Cutie 並進入連續追蹤
            # ------------------------------------------
            initial_mask = torch.from_numpy(sam2_interactor.mask > 0.5).float().to(device)
            image_tensor = to_tensor(cv2.cvtColor(first_color_image, cv2.COLOR_BGR2RGB)).to(device)
            depth_tensor = torch.from_numpy(first_depth_image * depth_scale).unsqueeze(0).unsqueeze(0).to(device).float()
            
            kinematics_dict = {
                'velocity': 0.0,
                'angular_velocity': 0.0,
                'depth_map': depth_tensor,
                'intrinsics': K_tensor,
                'se3_matrix': torch.eye(4, device=device).unsqueeze(0),
                'pixel_shift_uv': torch.tensor([[0.0, 0.0]], device=device),
                'use_depth_warp': ENABLE_DEPTH_WARP,
                'use_explicit_shift': ENABLE_EXPLICIT_SHIFT
            }

            processor.step(image_tensor, mask=initial_mask, objects=[1], 
                           use_kinematics_memory=ENABLE_KINEMATICS_MEMORY, 
                           kinematics_data=kinematics_dict)

            print("\n" + "="*50)
            print("【階段二：Cutie 即時追蹤中】")
            print("[ESC]: 停止追蹤，重置並回到待機預覽")
            print("[q 鍵]: 離開程式")

            while True:
                frames = pipeline.wait_for_frames()
                aligned_frames = align.process(frames)
                color_frame = aligned_frames.get_color_frame()
                depth_frame = aligned_frames.get_depth_frame()

                if not color_frame or not depth_frame:
                    continue

                color_image = np.asanyarray(color_frame.get_data())
                depth_image = np.asanyarray(depth_frame.get_data())

                image_tensor = to_tensor(cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)).to(device)
                depth_tensor = torch.from_numpy(depth_image * depth_scale).unsqueeze(0).unsqueeze(0).to(device).float()

                kinematics_dict['depth_map'] = depth_tensor
                
                output_prob = processor.step(
                    image_tensor, 
                    mask=None, 
                    use_kinematics_memory=ENABLE_KINEMATICS_MEMORY, 
                    kinematics_data=kinematics_dict
                )

                out_mask = processor.output_prob_to_mask(output_prob).cpu().numpy().astype(np.uint8)
                
                vis_image = color_image.copy()
                vis_image[out_mask > 0] = vis_image[out_mask > 0] * 0.5 + np.array([0, 0, 255]) * 0.5
                
                cv2.putText(vis_image, "Tracking: ESC to restart | 'q' to quit", 
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                cv2.imshow(window_name, vis_image)
                
                # Debug 用深度圖
                depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
                cv2.imshow('RealSense Depth', depth_colormap)
                
                key = cv2.waitKey(1)
                if key == 27:  # ESC 鍵
                    print(">> [ESC] 已觸發，重置追蹤器，準備重新測試...")
                    break # 觸發 break，會結束當前迴圈並被最外層的 while True 接住重新開始
                elif key == ord('q'): # q 鍵離開
                    quit_app = True
                    break

            if quit_app: break

    finally:
        print("Stopping camera and cleaning up...")
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
