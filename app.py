import os
import io
import cv2
import shutil
import tempfile
import asyncio
import subprocess
import aiohttp
import streamlit as st

from transformers import pipeline
import whisper

# -------------------------------------------
# 1. Check for FFmpeg and FFprobe dependencies
# -------------------------------------------
import shutil

ffmpeg_path = shutil.which("ffmpeg")
ffprobe_path = shutil.which("ffprobe")

if ffmpeg_path is None or ffprobe_path is None:
    st.error("FFmpeg/FFprobe is not installed or not found in PATH. Please install or configure them properly.")
    st.stop()

# -------------------------------------------
# 2. Caching the models
# -------------------------------------------
@st.cache_resource
def load_models():
    """
    Cache the loading of Whisper and emotion model to avoid re-loading on every rerun.
    """
    # Optional: More detailed emotions model
    emotion_model = pipeline("text-classification", 
                             model="bhadresh-savani/distilbert-base-uncased-emotion", 
                             return_all_scores=True)
    
    # Or revert to simple sentiment: pipeline("sentiment-analysis")
    whisper_model = whisper.load_model("base")
    return emotion_model, whisper_model

emotion_model, whisper_model = load_models()

# -------------------------------------------
# 3. Helper Functions
# -------------------------------------------

def check_video_metadata(file_path):
    """
    Check video metadata using FFprobe with detailed error reporting.
    Returns (is_valid, error_message).
    """
    try:
        probe_command = [
            ffprobe_path,
            '-v', 'quiet',
            '-print_format', 'json',
            '-show_format',
            '-show_streams',
            file_path
        ]
        
        result = subprocess.run(
            probe_command,
            capture_output=True,
            text=True,
            check=False
        )
        
        if result.returncode != 0:
            return False, "The video file appears to be corrupted or incomplete."
            
        if "Invalid data found" in result.stderr or "moov atom not found" in result.stderr:
            return False, "The video file is missing crucial metadata. This often happens with incomplete uploads."
            
        return True, ""
        
    except Exception as e:
        return False, f"Error analyzing video file: {str(e)}"


def repair_video_file(input_path, output_path):
    """
    Attempt to repair video file by re-encoding it.
    Returns (success, error_message).
    """
    try:
        repair_command = [
            ffmpeg_path,
            '-i', input_path,
            '-c:v', 'copy',
            '-c:a', 'copy',
            '-movflags', '+faststart',
            output_path
        ]
        
        result = subprocess.run(
            repair_command,
            capture_output=True,
            text=True,
            check=False
        )
        
        if result.returncode == 0:
            return True, ""
        else:
            return False, "Unable to repair video file."
            
    except Exception as e:
        return False, f"Error during repair attempt: {str(e)}"


def process_uploaded_file(uploaded_file):
    """
    Process and validate uploaded file with repair attempts if needed.
    Returns (valid_path, error_message).
    
    Uses chunk-based copying to avoid reading entire file into memory at once.
    """
    if not uploaded_file:
        return None, "No file uploaded."
        
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
            # More memory-efficient: copy in chunks
            shutil.copyfileobj(uploaded_file, temp_file)
            temp_path = temp_file.name
        
        # Check initial file
        is_valid, error_msg = check_video_metadata(temp_path)
        if is_valid:
            return temp_path, ""
            
        # Attempt repair if initial validation fails
        st.warning("Initial video validation failed. Attempting to repair file...")
        repair_output = temp_path + "_repaired.mp4"
        repair_success, repair_error = repair_video_file(temp_path, repair_output)
        
        # Clean up original temp file
        try:
            os.remove(temp_path)
        except:
            pass
            
        if repair_success:
            # Verify repaired file
            is_valid, error_msg = check_video_metadata(repair_output)
            if is_valid:
                st.success("Video file successfully repaired!")
                return repair_output, ""
                
        # If we get here, both initial and repair attempts failed
        try:
            if os.path.exists(repair_output):
                os.remove(repair_output)
        except:
            pass
            
        return None, """
            The video file appears to be corrupted or incomplete. This can happen when:
            - The upload was interrupted
            - The file wasn't properly finalized when recording
            - The file was corrupted during transfer
            
            Please try:
            1. Re-uploading the file
            2. Re-recording or re-exporting the video
            3. Using a different video file
            """
            
    except Exception as e:
        return None, f"Error processing upload: {str(e)}"


def extract_audio(video_path, output_path):
    """
    Extract audio from video using FFmpeg with proper error handling.
    Returns True if successful, False otherwise.
    """
    command = [
        ffmpeg_path,
        '-i', video_path,
        '-vn',  # Disable video
        '-acodec', 'libmp3lame',
        '-ar', '44100',  # Set audio sample rate
        '-ac', '2',      # Set number of audio channels
        '-ab', '192k',   # Set audio bitrate
        '-y',            # Overwrite output file if exists
        output_path
    ]
    
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True
        )
        return True
    except subprocess.CalledProcessError as e:
        st.error(f"FFmpeg error: {e.stderr}")
        return False


def extract_audio_emotion(video_path):
    """
    Extract audio and analyze emotions from the video.
    Returns (transcript_text, emotion_results).
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
        
    audio_path = video_path.replace(".mp4", ".mp3")
    
    try:
        # Extract audio
        if not extract_audio(video_path, audio_path):
            raise Exception("Failed to extract audio from video.")
        
        if not os.path.exists(audio_path):
            raise FileNotFoundError("Audio extraction failed - no output file created.")
        
        # Transcribe audio using Whisper
        transcription = whisper_model.transcribe(audio_path)
        if not transcription or "text" not in transcription:
            raise ValueError("Transcription failed - no text output generated.")
            
        transcript_text = transcription["text"]
        
        # Analyze transcription for sentiment/emotion
        if not transcript_text.strip():
            raise ValueError("Empty transcription.")
            
        # For the more detailed emotion model, returns a list of scores for each label
        emotion_results = emotion_model(transcript_text)
        
        return transcript_text, emotion_results
        
    except Exception as e:
        st.error(f"Error during audio processing: {str(e)}")
        raise
    finally:
        # Clean up the temporary audio file
        try:
            if os.path.exists(audio_path):
                os.remove(audio_path)
        except Exception as e:
            st.warning(f"Failed to clean up audio file: {str(e)}")


def extract_highlight_frames(video_path, n_highlights=3):
    """
    Extract highlight frames from the video at evenly spaced intervals.
    Returns a list of file paths to the saved highlight frames.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
        
    vidcap = cv2.VideoCapture(video_path)
    if not vidcap.isOpened():
        raise Exception("Failed to open video file.")
        
    try:
        fps = vidcap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            raise ValueError("Invalid FPS value detected.")
            
        frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count <= 0:
            raise ValueError("Invalid frame count detected.")
            
        duration = frame_count / fps

        st.write(f"Video duration: {duration:.2f} seconds.")

        highlight_frames = []
        interval = duration / (n_highlights + 1)

        for i in range(1, n_highlights + 1):
            timestamp = i * interval
            vidcap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
            success, image = vidcap.read()
            
            if not success or image is None:
                st.warning(f"Failed to extract frame {i}.")
                continue
                
            try:
                resized_image = cv2.resize(image, (640, 360))
                frame_path = f"highlight_{i}.jpg"
                cv2.imwrite(frame_path, resized_image)
                highlight_frames.append(frame_path)
            except Exception as e:
                st.warning(f"Failed to process frame {i}: {str(e)}")

        return highlight_frames
        
    finally:
        vidcap.release()

# -------------------------------------------
# 4. Streamlit App Layout and Flow
# -------------------------------------------

st.title("Viral Clip Extractor")
st.write("Upload a video or provide a OneDrive link, and AI will find the best moments!")

# Input method selection
input_type = st.radio("Choose input type:", ("Upload File", "OneDrive Link"))

video_path = None

if input_type == "Upload File":
    uploaded_file = st.file_uploader(
        "Upload a video file (MP4, max 100 MB):", 
        type=["mp4"],
        help="Make sure the video file is properly finalized and not corrupted."
    )
    
    if uploaded_file:
        video_path, error_msg = process_uploaded_file(uploaded_file)
        if error_msg:
            st.error(error_msg)
            video_path = None

elif input_type == "OneDrive Link":
    st.warning("Note: OneDrive downloads may be more prone to corruption. If you encounter issues, try uploading the file directly.")
    video_url = st.text_input("Paste your OneDrive shareable link here:")
    if video_url and st.button("Download Video"):
        st.write("Downloading video...")
        try:
            async def download_video(url):
                async with aiohttp.ClientSession() as session:
                    async with session.get(url) as response:
                        if response.status == 200:
                            data = await response.read()
                            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
                                temp_file.write(data)
                                return temp_file.name
                        else:
                            raise Exception(f"Download failed with status: {response.status}")

            temp_path = asyncio.run(download_video(video_url))
            
            # Re-open the file for processing
            with open(temp_path, 'rb') as f:
                video_path, error_msg = process_uploaded_file(f)
            
            if error_msg:
                st.error(error_msg)
                video_path = None
            else:
                st.success("Download and validation complete!")
                
        except Exception as e:
            st.error(f"Failed to download video: {e}")
            video_path = None

# -------------------------------------------
# 5. Processing the video if available
# -------------------------------------------
if video_path and os.path.exists(video_path):
    st.write("Video uploaded successfully. Processing...")

    # User option for number of highlights
    n_highlights = st.slider("Number of highlights to extract:", min_value=1, max_value=10, value=3)

    # Processing steps with feedback
    with st.spinner("Extracting highlights and analyzing emotion..."):
        try:
            # Extract audio and analyze
            transcript, emotion_results = extract_audio_emotion(video_path)
            
            # Initialize progress bar
            progress_bar = st.progress(0)

            # Extract frames
            highlights = []
            frame_results = extract_highlight_frames(video_path, n_highlights=n_highlights)
            
            # Update progress
            for i, frame_path in enumerate(frame_results):
                highlights.append(frame_path)
                progress_bar.progress(int((i + 1) / n_highlights * 100))

            # Display results
            if transcript:
                st.subheader("Transcript:")
                st.write(transcript)
            
            if emotion_results:
                st.subheader("Emotion Analysis:")
                st.write(emotion_results)

            if highlights:
                st.subheader("Highlights:")
                for i, highlight in enumerate(highlights):
                    if os.path.exists(highlight):
                        st.image(highlight, caption=f"Highlight {i+1}")
                    else:
                        st.warning(f"Highlight {i+1} image not found.")

            st.success("Processing complete!")

        except Exception as e:
            st.error(f"An error occurred during processing: {str(e)}")
            raise

        finally:
            # Clean up temporary files
            try:
                if os.path.exists(video_path):
                    os.remove(video_path)
                    
                # Clean up highlight images
                for i in range(n_highlights):
                    frame_path = f"highlight_{i+1}.jpg"
                    if os.path.exists(frame_path):
                        os.remove(frame_path)
            except Exception as e:
                st.warning(f"Failed to clean up temporary files: {str(e)}")
