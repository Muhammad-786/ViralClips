import streamlit as st
import tempfile
import ffmpeg
import aiohttp
import asyncio
import os
from transformers import pipeline
import whisper
import cv2

# Initialize AI models
st.title("Viral Clip Extractor")
st.write("Upload a video or provide a OneDrive link, and AI will find the best moments!")

emotion_model = pipeline("sentiment-analysis")
whisper_model = whisper.load_model("base")  # Load Whisper model for transcription

# Input method selection
input_type = st.radio("Choose input type:", ("Upload File", "OneDrive Link"))

video_path = None

# Helper: Asynchronous video downloader
async def download_video(video_url):
    async with aiohttp.ClientSession() as session:
        async with session.get(video_url) as response:
            if response.status == 200:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
                    temp_file.write(await response.read())
                    return temp_file.name
            else:
                raise Exception(f"Failed to download video: {response.status}")

# Video input handling
if input_type == "Upload File":
    uploaded_file = st.file_uploader("Upload a video file (MP4, max 100 MB):", type=["mp4"])
    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
            temp_file.write(uploaded_file.read())
            video_path = temp_file.name

elif input_type == "OneDrive Link":
    video_url = st.text_input("Paste your OneDrive shareable link here:")
    if video_url and st.button("Download Video"):
        st.write("Downloading video...")
        try:
            video_path = asyncio.run(download_video(video_url))
            st.write("Download complete!")
        except Exception as e:
            st.error(f"Failed to download video: {e}")

# Process the video if available
if video_path:
    st.write("Video uploaded successfully. Processing...")

    # Extract transcription and emotions
    def extract_audio_emotion(video_path):
        # Convert video to audio
        audio_path = video_path.replace(".mp4", ".mp3")
        ffmpeg.input(video_path).output(audio_path).run(quiet=True)

        # Transcribe audio using Whisper
        transcription = whisper_model.transcribe(audio_path)["text"]

        # Analyze transcription for sentiment
        emotion_results = emotion_model(transcription)
        return transcription, emotion_results

    # Extract frames for highlights
    def extract_highlight_frames(video_path, n_highlights=3):
        vidcap = cv2.VideoCapture(video_path)
        fps = int(vidcap.get(cv2.CAP_PROP_FPS))
        frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps

        st.write(f"Video duration: {duration:.2f} seconds.")

        highlight_frames = []
        interval = duration / (n_highlights + 1)

        for i in range(1, n_highlights + 1):
            timestamp = i * interval
            vidcap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
            success, image = vidcap.read()
            if success:
                resized_image = cv2.resize(image, (640, 360))  # Resize for memory efficiency
                frame_path = f"highlight_{i}.jpg"
                cv2.imwrite(frame_path, resized_image)
                highlight_frames.append(frame_path)

        return highlight_frames

    # User options for customization
    n_highlights = st.slider("Number of highlights to extract:", min_value=1, max_value=10, value=3)

    # Processing steps with feedback
    with st.spinner("Extracting highlights..."):
        transcript, emotion_results = extract_audio_emotion(video_path)
        highlights = extract_highlight_frames(video_path, n_highlights=n_highlights)

    # Display results
    st.write("**Transcript:**", transcript)
    st.write("**Emotion Analysis:**", emotion_results)

    st.write("**Highlights:**")
    for i, highlight in enumerate(highlights):
        st.image(highlight, caption=f"Highlight {i+1}")

    # Clean up temporary files
    if os.path.exists(video_path):
        os.remove(video_path)
