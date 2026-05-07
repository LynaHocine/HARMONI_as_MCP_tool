from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av
import numpy as np

class AVRecorder(VideoTransformerBase):
    def __init__(self):
        self.video_frames = []
        self.audio_frames = []

    def transform(self, frame: av.VideoFrame):
        img = frame.to_ndarray(format="bgr24")
        self.video_frames.append(img)
        return frame

    def recv_audio(self, frame: av.AudioFrame):
        audio = frame.to_ndarray()
        self.audio_frames.append(audio)
        return frame