#!/usr/bin/env python
# coding: utf-8

# In[4]:


import cv2
import numpy as np
import os
import tensorflow as tf # 您的程式碼似乎使用 TensorFlow，所以在此匯入

MODEL_PATH = 'best_potato_model.keras'
model = tf.keras.models.load_model(MODEL_PATH)

# --- 請先在此區域進行設定 ---

# 假設您的模型和圖片尺寸已經定義好了
# 如果沒有，請取消以下註解並設定
# from tensorflow.keras.models import load_model
# model = load_model('your_model.h5') # 載入您訓練好的模型
# IMAGE_SIZE = (224, 224) # 設定模型輸入的圖片尺寸
IMAGE_SIZE = (224, 224)
# 指定包含所有圖片的資料夾路徑
folder_path = '../for_test/N/' # 請將 'for_test' 替換成您的資料夾名稱

# --- 主要處理邏輯 ---

# 檢查指定的路徑是否存在且是一個資料夾
if not os.path.isdir(folder_path):
    print(f"錯誤：找不到資料夾 '{folder_path}'")
else:
    # 取得資料夾中所有檔案的名稱
    all_files = os.listdir(folder_path)
    
    # 篩選出圖片檔案 (可以根據需求增加或修改附檔名)
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
    image_files = [f for f in all_files if os.path.splitext(f)[1].lower() in image_extensions]

    if not image_files:
        print(f"在 '{folder_path}' 資料夾中找不到任何支援的圖片檔案。")
    else:
        print(f"在 '{folder_path}' 中找到 {len(image_files)} 張圖片，開始進行分析...\n")

        # 遍歷每一張圖片
        for image_name in image_files:
            # 組合出完整的圖片檔案路徑
            img_path = os.path.join(folder_path, image_name)
            
            # --- 以下是您原本的處理邏輯，放入迴圈中 ---
            
            # 載入一張圖片
            img = cv2.imread(img_path)

            # 檢查圖片是否成功載入
            if img is None:
                print(f"警告：無法讀取圖片 '{img_path}'，將跳過此檔案。\n")
                continue # 繼續處理下一張圖片
            
            print(f"--- 正在分析: {image_name} ---")

            # 調整圖片大小並轉換顏色空間
            img = cv2.resize(img, IMAGE_SIZE)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # 轉換為 RGB
            img_array = tf.expand_dims(img, 0) # 為模型預測建立一個 batch

            # 進行預測
            prediction = model.predict(img_array)
            score = prediction[0][0]

            # 顯示結果
            if score > 0.5:
                print(f"這顆馬鈴薯已發芽的機率是: {100 * score:.2f}%\n")
            else:
                print(f"這顆馬鈴薯未發芽的機率是: {100 * (1 - score):.2f}%\n")


# In[ ]:




