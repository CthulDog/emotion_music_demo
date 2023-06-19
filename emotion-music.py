import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import streamlit as st
from PIL import Image, ImageOps

# 加载模型
emotion_model = tf.keras.models.load_model('D:/demo/emotion_model.h5')
music_model = tf.keras.models.load_model('D:/demo/song-model.h5')

# 读取DEAM数据集中的歌曲信息
deam_data = pd.read_csv('D:/demo/deam_playlist.csv')

# 从图像中识别人脸情绪的函数
def predict_emotion(img):
    # 对图像进行预处理
    gray = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (48, 48))
        img_pixels = roi_gray.astype('float32') / 255.
        img_pixels = np.expand_dims(img_pixels, axis=0)
        img_pixels = np.expand_dims(img_pixels, axis=-1)

        # 预测情绪
        emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
        predictions = emotion_model.predict(img_pixels)[0]
        max_index = np.argmax(predictions)
        return emotions[max_index]

# 修改 recommend_song 函数以返回歌曲 ID 而不是歌曲名称
def recommend_song(emotion, playlist_data):
    filtered_songs = playlist_data[playlist_data['emotion'] == emotion]
    
    if not filtered_songs.empty:
        recommended_song_id = filtered_songs.sample().iloc[0]['song_id']
        return recommended_song_id
    else:
        return None

# 创建一个根据歌曲 ID 加载音频文件的函数
def load_audio_file_by_id(song_id, id_to_name_map):
    audio_file_path = id_to_name_map[id_to_name_map['song_id'] == song_id].iloc[0]['audio_file']
    return audio_file_path

# Streamlit 应用
st.title("基于人脸情绪的音乐推荐")
uploaded_image = st.file_uploader("上传一张包含面部表情的图片", type=["jpg", "jpeg", "png"])
if uploaded_image is not None:
    image = Image.open(uploaded_image)
    
    # 检查图像是否为彩色图像，如果不是，则将其转换为 RGB
    if image.mode != 'RGB':
        image = image.convert('RGB')
        
    st.image(image, caption='上传的图片', use_column_width=True)
    st.write("")
    st.write("正在识别情绪...")

    emotion = predict_emotion(image)
    st.write(f"识别到的情绪是：{emotion}")

    recommended_song_id = recommend_song(emotion, deam_data)  # 添加 playlist_data 参数
    filtered_deam_data = deam_data[deam_data['song_id'] == recommended_song_id]

    if not filtered_deam_data.empty:
        audio_file_path = filtered_deam_data.iloc[0]['audio_file']
        st.write(f"推荐的歌曲ID是：{recommended_song_id}")
        st.write(f"即将播放歌曲：{audio_file_path}")
        st.audio(audio_file_path)
    else:
        st.write("抱歉，我们找不到与您的情绪匹配的歌曲。")