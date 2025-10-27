#!/usr/bin/env python
# coding: utf-8

# In[2]:


import tensorflow as tf
from tensorflow.keras.applications.resnet_v2 import preprocess_input
from roboflow import Roboflow
import cv2
import numpy as np
import os
import time
import sys

def resource_path(relative_path):
    """ 取得打包後資源的絕對路徑 """
    try:
        # PyInstaller 建立一個暫存資料夾並將路徑存在 _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)
    
# --- 步驟一：定義偵測和選擇攝影機的函式 ---

def find_available_cameras(max_cameras_to_check=10):
    """
    測試索引 0 到 max_cameras_to_check-1，回傳所有可用的攝影機索引列表。
    """
    available_cameras = []
    print("🔍 正在尋找可用的攝影機...")
    for i in range(max_cameras_to_check):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            print(f"  ✅ 找到攝影機，索引為: {i}")
            available_cameras.append(i)
            cap.release()
        else:
            # 在找不到更多攝影機時可以提早結束，以節省時間
            break
    return available_cameras

def select_camera(camera_indices):
    """
    根據找到的攝影機索引列表，讓使用者選擇。
    """
    if not camera_indices:
        print("❌ 找不到任何攝影機。")
        return None
    
    if len(camera_indices) == 1:
        print(f"✅ 自動選擇唯一的攝影機，索引為: {camera_indices[0]}")
        return camera_indices[0]

    print("\n📷 發現多個攝影機，將逐一顯示預覽畫面2秒鐘...")
    
    for index in camera_indices:
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            continue
        
        print(f"  預覽來自攝影機索引 {index} (預覽視窗可能會彈出在背景)...")
        start_time = time.time()
        window_name = f"Preview from Camera Index {index} (Close in 2s)"
        
        while time.time() - start_time < 2:
            ret, frame = cap.read()
            if not ret:
                break
            cv2.putText(frame, f"Index: {index}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow(window_name, frame)
            if cv2.waitKey(1) != -1: # 等待1毫秒，讓視窗可以更新
                break
        
        cap.release()
        cv2.destroyWindow(window_name)

    # 讓使用者輸入選擇
    while True:
        try:
            choice = input(f"\n👉 請從可用的索引 {camera_indices} 中，輸入您想使用的攝影機編號: ")
            selected_index = int(choice)
            if selected_index in camera_indices:
                print(f"✅ 您已選擇使用攝影機，索引為: {selected_index}")
                return selected_index
            else:
                print(f"❌ 無效的選擇。請輸入 {camera_indices} 中的一個數字。")
        except ValueError:
            print("❌ 輸入無效，請輸入一個數字。")


# --- 步驟二：主程式邏輯 ---

# 1. 參數設定 (與之前相同)
ROBOFLOW_API_KEY = "UCLBeCClmaD7BW6BWuLG"
PROJECT_ID = "potato-detection-3et6q"
MODEL_VERSION = 11
CLASSIFIER_MODEL_PATH = resource_path('best_potato_model.keras') 
CLASSIFIER_IMG_SIZE = (224, 224)
PROCESS_EVERY_N_FRAMES = 30

# 2. 載入模型 (與之前相同)
# ... (此處省略載入模型的程式碼，請確保它們已成功載入)
try:
    rf = Roboflow(api_key=ROBOFLOW_API_KEY)
    project = rf.workspace().project(PROJECT_ID)
    detection_model = project.version(MODEL_VERSION).model
    classifier_model = tf.keras.models.load_model(CLASSIFIER_MODEL_PATH)
    print("✅ 所有模型載入成功！")
    print(f"  > 分類模型路徑: {CLASSIFIER_MODEL_PATH}") # 可以加上這行來偵錯，看看路徑是否正確
except Exception as e:
    print(f"❌ 模型載入失敗: {e}")
    detection_model = None
    classifier_model = None
    
# 3. 執行攝影機選擇並啟動主迴圈
if detection_model and classifier_model:
    available_cams = find_available_cameras()
    selected_cam_index = select_camera(available_cams)

    if selected_cam_index is not None:
        cap = cv2.VideoCapture(selected_cam_index, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        if not cap.isOpened():
            print(f"❌ 錯誤：無法打開選擇的攝影機 {selected_cam_index}。")
        else:
            print("\n📹 攝影機已啟動！按下 'q' 鍵退出實時偵測視窗。")
            
            frame_counter = 0
            latest_boxes_and_labels = []

            while True:
                ret, frame = cap.read()
                # print("讀取畫面中...")
                if not ret: break

                if frame_counter % PROCESS_EVERY_N_FRAMES == 0:
                    latest_boxes_and_labels = []
                    
                    # --- ↓↓↓ 以下是修改後的區塊 ↓↓↓ ---
                    
                    # 1. 獲取原始畫面的尺寸
                    original_h, original_w, _ = frame.shape
                    
                    # 2. 定義偵測用的尺寸
                    detection_size = (320, 320)
                    
                    # 3. 將畫面縮放到偵測用的尺寸
                    resized_for_detection = cv2.resize(frame, detection_size)
                    
                    # 4. 暫存縮小後的圖片並進行偵測
                    temp_frame_path = "temp_frame_for_detection.jpg"
                    cv2.imwrite(temp_frame_path, resized_for_detection)
                    predictions = detection_model.predict(temp_frame_path, confidence=40, overlap=30).json()['predictions']
    
                    # 5. 計算從「偵測尺寸」放大回「原始尺寸」的縮放比例
                    x_scale = original_w / detection_size[0]
                    y_scale = original_h / detection_size[1]
    
                    for pred in predictions:
                        # 獲取的是在 416x416 圖片上的座標
                        center_x, center_y, width, height = int(pred['x']), int(pred['y']), int(pred['width']), int(pred['height'])
                        det_x1, det_y1 = int(center_x - width / 2), int(center_y - height / 2)
                        det_x2, det_y2 = int(center_x + width / 2), int(center_y + height / 2)
    
                        # 6. ！！！關鍵步驟：將偵測到的座標等比例放大回原始尺寸！！！
                        draw_x1 = int(det_x1 * x_scale)
                        draw_y1 = int(det_y1 * y_scale)
                        draw_x2 = int(det_x2 * x_scale)
                        draw_y2 = int(det_y2 * y_scale)
    
                        # 7. 使用「放大後」的座標來從「原始」畫面中裁切馬鈴薯
                        # 加上邊界保護
                        draw_x1, draw_y1 = max(0, draw_x1), max(0, draw_y1)
                        draw_x2, draw_y2 = min(original_w, draw_x2), min(original_h, draw_y2)
                        
                        # 注意！這裡是從 `frame` (原始畫面) 中裁切
                        cropped_potato = frame[draw_y1:draw_y2, draw_x1:draw_x2]
                        
                        if cropped_potato.size == 0: continue
    
                        # (後續的分類預測邏輯不變...)
                        resized_crop = cv2.resize(cropped_potato, CLASSIFIER_IMG_SIZE)
                        rgb_crop = cv2.cvtColor(resized_crop, cv2.COLOR_BGR2RGB)
                        img_array = np.expand_dims(rgb_crop, axis=0)
                        # preprocessed_img = preprocess_input(img_array)
                        prediction_score = classifier_model.predict(img_array, verbose=0)[0][0]
                        
                        if prediction_score > 0.5:
                            label = f"Sprouted: {prediction_score:.2f}"
                            color = (0, 0, 255) # 紅色
                        else:
                            label = f"Not Sprouted: {1-prediction_score:.2f}"
                            color = (0, 255, 0) # 綠色
                        
                        # 8. 保存放大後的座標和標籤，以供繪圖使用
                        latest_boxes_and_labels.append(((draw_x1, draw_y1, draw_x2, draw_y2), label, color))
                        
                for box, label, color in latest_boxes_and_labels:
                    (x1, y1, x2, y2) = box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
                cv2.imshow('Real-time Potato Sprout Detection', frame)
                
                frame_counter += 1

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            cap.release()
            cv2.destroyAllWindows()
            # 在 Jupyter 中，有時需要手動關閉所有視窗
            for i in range(5):
                cv2.waitKey(1)

# In[ ]:




