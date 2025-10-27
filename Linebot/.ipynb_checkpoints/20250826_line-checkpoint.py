#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
os.getcwd()


# In[2]:


from flask import Flask, request
from pyngrok import ngrok
from linebot import LineBotApi, WebhookHandler
from linebot.models import TextSendMessage, MessageEvent, TextMessage
import json
from dotenv import load_dotenv


# #### key

# In[3]:


# %%writefile .env
# # ngrok
# NGROK_AUTHTOKEN = '30fL2Ol11MYKNjH4QwW5m5zarBi_4gGD5uvVGxQC1AAN5mxKC'

# # Line Basic settings -> Channel secret
# LINE_CHANNEL_SECRET = 'c0d5d80e5e72627356acec18aad76a75'
# # Line Messaging API -> Channel access token
# LINE_CHANNEL_ACCESS_TOKEN = 'wsrdGKz4GVNDOmBziacS5WXX4C/f/5Gojxl3X+Merxaovqjqzyz999nWrkOueOrycLgF4qQv2tYoeRk6ZYQdKzR9SH3PijDiRaj5GpyXRrpoq6fXt4kJBhqsyABs/jlfawyDOHWg44/AgCvKZjejKgdB04t89/1O/w1cDnyilFU='


# In[4]:


import os
from dotenv import load_dotenv

load_dotenv()    # 讀取同目錄的 .env

NGROK_AUTHTOKEN = os.getenv("NGROK_AUTHTOKEN")
CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET")
CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")

print(NGROK_AUTHTOKEN)

print('讀取完成!')



import os
import io
from dotenv import load_dotenv
from flask import Flask, request, abort
from pyngrok import ngrok
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, TextSendMessage, ImageMessage

import tensorflow as tf
from tensorflow.keras.applications.resnet_v2 import preprocess_input
# 移除 PIL，改用 OpenCV
import cv2 
import numpy as np

# --- 步驟 1: 載入環境變數 ---
load_dotenv()
ACCESS_TOKEN = os.getenv('LINE_CHANNEL_ACCESS_TOKEN')
CHANNEL_SECRET = os.getenv('LINE_CHANNEL_SECRET')
NGROK_AUTHTOKEN = os.getenv('NGROK_AUTHTOKEN')

# --- 步驟 2: 初始化 Flask、Line Bot 和 ngrok ---
app = Flask(__name__)
line_bot_api = LineBotApi(ACCESS_TOKEN)
handler = WebhookHandler(CHANNEL_SECRET)

ngrok.set_auth_token(NGROK_AUTHTOKEN)
public_url = ngrok.connect(5000).public_url
print(f"公開網址 (Webhook URL): {public_url}/callback")
line_bot_api.set_webhook_endpoint(f"{public_url}/callback")

# --- 步驟 3: 載入並設定 AI 模型 ---
MODEL_PATH = 'best_potato_model.keras'
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"模型 '{MODEL_PATH}' 載入成功。")
except Exception as e:
    print(f"錯誤：無法載入模型 '{MODEL_PATH}'。請確認檔案路徑是否正確。")
    print(e)
    exit()

IMAGE_SIZE = (224, 224)

# --- ★★★ 修改後的預測函式 ★★★ ---
def predict_potato(image_bytes):
    """
    使用 OpenCV 進行圖片預處理，確保與 Jupyter Notebook 流程一致。
    """
    try:
        # 1. 將 bytes 轉換為 numpy array
        np_arr = np.frombuffer(image_bytes, np.uint8)
        
        # 2. 使用 cv2 從 array 解碼圖片，預設為 BGR 格式
        img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        # 3. 將圖片從 BGR 轉換為 RGB (和您筆記本中的做法相同)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        # 4. 調整圖片大小以符合模型輸入
        img_resized = cv2.resize(img_rgb, IMAGE_SIZE)
        
        # 5. 擴展維度以建立一個 batch (1, 224, 224, 3)
        img_array = np.expand_dims(img_resized, axis=0)
        
        # 6. 使用與訓練時相同的 ResNetV2 預處理函式
        preprocessed_img = preprocess_input(img_array)
        
        # 7. 進行預測
        prediction = model.predict(preprocessed_img)
        score = prediction[0][0]
        
        return score
    except Exception as e:
        print(f"模型預測時發生錯誤: {e}")
        return None

# --- 步驟 4: Flask 路由與 Line Bot 訊息處理 (維持不變) ---
@app.route("/callback", methods=['POST'])
def callback():
    signature = request.headers['X-Line-Signature']
    body = request.get_data(as_text=True)
    app.logger.info("Request body: " + body)
    
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)
            
    return 'OK'

@handler.add(MessageEvent, message=ImageMessage)
def handle_image_message(event):
    message_content = line_bot_api.get_message_content(event.message.id)
    image_bytes = message_content.content
    
    score = predict_potato(image_bytes)
    
    if score is not None:
        if score > 0.5:
            reply_text = f"分析結果：\n這顆馬鈴薯已發芽的機率是: {100 * score:.2f}%"
        else:
            reply_text = f"分析結果：\n這顆馬鈴薯未發芽的機率是: {100 * (1 - score):.2f}%"
    else:
        reply_text = "圖片分析失敗，請稍後再試一次。"

    line_bot_api.reply_message(
        event.reply_token,
        TextSendMessage(text=reply_text)
    )

@handler.add(MessageEvent, message=TextMessage)
def handle_text_message(event):
    line_bot_api.reply_message(
        event.reply_token,
        TextSendMessage(text="您好！請直接傳送一張馬鈴薯的圖片給我，我會幫您分析它是否發芽。")
    )

# --- 步驟 5: 啟動伺服器 ---
if __name__ == "__main__":
    app.run(port=5000)


# In[ ]:




