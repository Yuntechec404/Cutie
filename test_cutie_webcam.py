#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import torch
import numpy as np
import torchvision.transforms.functional as F
from ultralytics import YOLO
import rospy

# 匯入 Cutie 核心組件
import sys
CUTIE_PATH = "/home/user/Cutie" 
if CUTIE_PATH not in sys.path:
    sys.path.append(CUTIE_PATH)

from cutie.inference.inference_core import InferenceCore
from cutie.utils.get_default_model import get_default_model

def cv2_to_tensor(img_bgr, device):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    tensor = F.to_tensor(img_rgb).to(device)
    return tensor

def overlay_mask(img, mask, color=(0, 255, 0), alpha=0.5):
    vis = img.copy()
    img_color = np.zeros_like(vis)
    img_color[:] = color
    mask_bool = mask > 0
    vis[mask_bool] = cv2.addWeighted(vis, 1.0, img_color, alpha, 0)[mask_bool]
    return vis

@torch.inference_mode()
@torch.cuda.amp.autocast()
def main():
    rospy.init_node("yolo_cutie_test", anonymous=True)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # ==========================================
    # 🎯 自訂參數區
    # ==========================================
    TARGET_CLASS_ID = 0
    YOLO_MODEL_PATH = "/home/user/catkin_ws/src/FoundationPose/data/bunch_stem-seg.pt"
    CAMERA_INDEX = 0
    
    # [新增] 腐蝕運算次數：用來縮小 Mask 防止多割
    # 數值越大，Mask 越瘦。若發現還是會割到兩根，就調大此數值。
    ERODE_ITERATIONS = 2
    # ==========================================

    yolo_model = YOLO(YOLO_MODEL_PATH)

    network = get_default_model()
    network.eval().to(device)
    processor = InferenceCore(network, cfg=network.cfg)
    processor.max_internal_size = 640 
    
    cap = cv2.VideoCapture(CAMERA_INDEX)
    win_name = "YOLO+Cutie Hybrid Test (Eroded)"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    is_tracking = False

    # 定義腐蝕用的核心 (5x5 矩形)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    while not rospy.is_shutdown():
        ret, frame = cap.read()
        if not ret: continue
        vis_frame = frame.copy()

        if not is_tracking:
            results = yolo_model.predict(frame, classes=[TARGET_CLASS_ID], imgsz=640, conf=0.1, verbose=False)[0]
            best_mask_uint8 = None
            max_area = 0

            if results.masks is not None:
                masks = results.masks.data.cpu().numpy()
                for m in masks:
                    m_resized = cv2.resize(m, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
                    m_uint8 = (m_resized > 0.5).astype(np.uint8)
                    vis_frame = overlay_mask(vis_frame, m_uint8, color=(0, 255, 0), alpha=0.4)
                    if m_uint8.sum() > max_area:
                        max_area = m_uint8.sum()
                        best_mask_uint8 = m_uint8

            cv2.putText(vis_frame, "MODE: YOLO", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow(win_name, vis_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' ') and best_mask_uint8 is not None:
                init_mask = torch.from_numpy(best_mask_uint8).to(device).long()
                img_tensor = cv2_to_tensor(frame, device)
                processor.clear_memory()
                processor.step(img_tensor, init_mask, objects=[1])
                is_tracking = True

        else:
            img_tensor = cv2_to_tensor(frame, device)
            output_prob = processor.step(img_tensor)
            pred_mask_tensor = processor.output_prob_to_mask(output_prob)
            raw_tracked_mask = (pred_mask_tensor.cpu().numpy() == 1).astype(np.uint8)
            
            # --------------------------------------------------
            # [核心修改] 執行腐蝕運算，強迫 Mask 收縮
            # --------------------------------------------------
            tracked_mask = cv2.erode(raw_tracked_mask, kernel, iterations=ERODE_ITERATIONS)
            # --------------------------------------------------

            if tracked_mask.sum() < 50:
                rospy.logwarn("🚨 Cutie lost target!")
                processor.clear_memory()
                is_tracking = False
                continue

            # 顯示腐蝕後的 Mask (紅色)
            vis_frame = overlay_mask(vis_frame, tracked_mask, color=(0, 0, 255), alpha=0.5)
            
            # 計算中心點（因為 Mask 變瘦了，中心點會精確落在單一根莖上）
            M = cv2.moments(tracked_mask)
            if M["m00"] != 0:
                cX, cY = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
                cv2.circle(vis_frame, (cX, cY), 8, (255, 0, 0), -1)

            cv2.putText(vis_frame, f"MODE: LOCKED (Erode:{ERODE_ITERATIONS})", (20, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.imshow(win_name, vis_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('r') or key == ord('R'):
                processor.clear_memory()
                is_tracking = False

        if key == ord('q') or key == ord('Q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
