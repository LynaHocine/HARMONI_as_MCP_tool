import os
import sys
import time
import shutil
import tempfile
import asyncio

import streamlit as st
import nest_asyncio
from streamlit_webrtc import webrtc_streamer, WebRtcMode
from aiortc.contrib.media import MediaRecorder
from langchain_core.messages import HumanMessage, AIMessage

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(BASE_DIR)
from agent.agent import MCPAgent

nest_asyncio.apply()

class ChatBotUI:
    """
    Streamlit-based chat interface that supports:
      - Text input via st.chat_input
      - Webcam + microphone recording via streamlit-webrtc / aiortc
      - Sending recorded video (with audio) to an MCP agent for analysis
    """

    def __init__(self):
        # Retrieve or create an event loop for running async code synchronously.
        # Streamlit does not natively run inside an async context, so we manage
        # the loop manually and use nest_asyncio to avoid "loop already running"
        # errors.
        try:
            self.loop = asyncio.get_event_loop()
        except RuntimeError:
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)

        self._init_session()

    def _init_session(self):
        """
        Initialise all Streamlit session-state keys on first run.
        Streamlit reruns the entire script on every interaction, so we guard
        each key with an 'if not in' check to avoid resetting state mid-session.
        """

        # --- MCP Agent ---
        if "agent" not in st.session_state:
            st.session_state.agent = MCPAgent()
            self.loop.run_until_complete(st.session_state.agent.initialize())

        # --- Video recording state machine ---
        # Possible values:
        #   "idle"     → no recording, nothing pending
        #   "stopping" → user clicked Stop, waiting for WebRTC stream to close
        #   "ready"    → stream is closed, file is ready to be processed
        if "recording_state" not in st.session_state:
            st.session_state.recording_state = "idle"

        # Path of the video file currently being (or just finished) recorded.
        if "pending_video_path" not in st.session_state:
            st.session_state.pending_video_path = None

        # Stable temp-file path reused across reruns so MediaRecorder always
        # writes to the same file on disk.
        if "current_tmp_video" not in st.session_state:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tmp.close()
            st.session_state.current_tmp_video = tmp.name

    # ------------------------------------------------------------------
    # Static helper
    # ------------------------------------------------------------------

    @staticmethod
    def wait_for_file_ready(
        path: str,
        timeout: float = 15.0,
        stable_for: float = 0.8,
    ) -> bool:
        """
        Block until the file at *path* is non-empty AND its size has not
        changed for *stable_for* seconds (meaning aiortc finished writing).

        Why a @staticmethod inside the class?
        --------------------------------------
        It belongs to ChatBotUI's video-handling responsibility but needs no
        access to instance state (self).  A @staticmethod keeps the module
        namespace clean while remaining callable as
        ChatBotUI.wait_for_file_ready(...) without instantiation.

        Parameters
        ----------
        path        : filesystem path to the video file to watch
        timeout     : maximum seconds to wait before giving up
        stable_for  : seconds the file size must stay unchanged to be
                      considered fully written by aiortc

        Returns
        -------
        True  – file is ready within the timeout window
        False – timeout reached before the file stabilised
        """
        deadline = time.time() + timeout
        last_size = -1
        stable_since = None

        while time.time() < deadline:
            if os.path.exists(path):
                size = os.path.getsize(path)
                if size > 0:
                    if size == last_size:
                        # Size unchanged – start or continue stability timer
                        if stable_since is None:
                            stable_since = time.time()
                        elif time.time() - stable_since >= stable_for:
                            return True  # stable long enough → ready
                    else:
                        # Still growing → reset stability timer
                        last_size = size
                        stable_since = None
            time.sleep(0.2)  # poll every 200 ms

        return False  # timed out

    # ------------------------------------------------------------------
    # Chat history display
    # ------------------------------------------------------------------

    def display_chat(self):
        """
        Render the full conversation history stored in the agent.
        ToolMessage objects (internal tool results) are intentionally skipped.
        """
        for msg in st.session_state.agent.chat_history:
            if isinstance(msg, HumanMessage):
                st.chat_message("user").write(msg.content)
            elif isinstance(msg, AIMessage):
                st.chat_message("assistant").write(msg.content)

    # ------------------------------------------------------------------
    # Video recording UI  ── sidebar only, NO messages rendered here
    # ------------------------------------------------------------------

    def record_video_ui(self):
        """
        Render the webcam/microphone recorder widget inside the sidebar.

        KEY DESIGN RULE: this method must NEVER call st.chat_message(),
        st.info(), st.error(), st.success(), or st.write() outside the
        `with st.sidebar` block.  Any call to those functions here would
        render the element inside the sidebar instead of the main column.
        All user-visible feedback is the responsibility of handle_input().

        State machine
        -------------
        idle     → user clicks START          → stream opens (WebRTC handles it)
        playing  → user clicks Stop & Send    → recording_state = "stopping"
        stopping → Streamlit reruns, stream   → recording_state = "ready"
                   is no longer playing              + st.rerun()
        ready    → handled in handle_input()  → _send_video_to_agent()
                   (main column)                     → recording_state = "idle"
        """
        with st.sidebar:
            st.subheader("🎥 Webcam & Microphone Recorder")

            temp_video_path = st.session_state.current_tmp_video

            def recorder_factory():
                """
                Called by streamlit-webrtc when a new recording starts.
                Truncate the file first so a second recording never appends
                to leftover bytes from the previous session.
                """
                open(temp_video_path, "wb").close()  # truncate / create
                return MediaRecorder(temp_video_path)

            webrtc_ctx = webrtc_streamer(
                key="webcam-recorder",
                mode=WebRtcMode.SENDRECV,
                # ── Audio enabled ─────────────────────────────────────────
                # Requesting both video and audio from the browser.
                # aiortc muxes them into a single .mp4 file through
                # MediaRecorder so the agent receives a complete AV file.
                media_stream_constraints={"video": True, "audio": True},
                in_recorder_factory=recorder_factory,
                rtc_configuration={
                    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
                },
            )

            # ── Branch A: stream is active ─────────────────────────────────
            if webrtc_ctx.state.playing:
                st.success("🔴 Recording in progress (video + audio)...")

                if st.button("⏹️ Stop & Send to Agent"):
                    # Save the target path and request a stop.
                    # We do NOT call st.rerun() yet: the stream is still open
                    # and aiortc hasn't flushed the file.  On the next Streamlit
                    # cycle, webrtc_ctx.state.playing will be False, which moves
                    # us to Branch B.
                    st.session_state.pending_video_path = temp_video_path
                    st.session_state.recording_state = "stopping"
                    # Trigger one rerun so Streamlit re-evaluates the WebRTC
                    # state quickly instead of waiting for the next user event.
                    st.rerun()

            # ── Branch B: stream just closed after Stop was requested ──────
            elif st.session_state.recording_state == "stopping":
                # The WebRTC stream is no longer playing → aiortc has received
                # the close signal and will finish writing the file shortly.
                # Advance to "ready" and rerun so handle_input() can take over.
                st.session_state.recording_state = "ready"
                st.rerun()

            # ── Branch C: ready – handled in main column ───────────────────
            elif st.session_state.recording_state == "ready":
                # Nothing to render here; handle_input() will call
                # _send_video_to_agent() from the main column context.
                st.info("⏳ Processing video...")

            # ── Branch D: idle ─────────────────────────────────────────────
            else:
                st.info("Click START to open the webcam and microphone.")

    # ------------------------------------------------------------------
    # Agent interaction  ── always called from the MAIN column
    # ------------------------------------------------------------------

    def _send_video_to_agent(self, video_path: str):
        """
        Copy the recorded video to a safe path, display it in the main chat
        area, forward the path to the MCP agent, and render the response.

        This method is ONLY called from handle_input(), which runs outside
        any `with st.sidebar` context, so every st.* call here renders in
        the main column on the right side of the page.

        Parameters
        ----------
        video_path : path to the finalised source video file
        """

        # Final sanity check.
        if not os.path.exists(video_path) or os.path.getsize(video_path) == 0:
            st.error("❌ Source video file is missing or empty. Please try again.")
            return

        # Copy to a controlled directory (short path, no spaces).
        safe_dir = os.path.join(tempfile.gettempdir(), "mcp_videos")
        os.makedirs(safe_dir, exist_ok=True)
        safe_path = os.path.join(safe_dir, f"vid_{int(time.time())}.mp4")

        try:
            shutil.copy2(video_path, safe_path)
        except Exception as exc:
            st.error(f"❌ Failed to copy video to safe path: {exc}")
            return

        # ── "Video sent to agent" – MAIN column ───────────────────────────
        with open(safe_path, "rb") as f:
            st.chat_message("user").video(f.read())
        st.chat_message("user").write("📹 Video sent to agent")

        # ── "Agent is thinking" – MAIN column ─────────────────────────────
        thinking_placeholder = st.empty()
        thinking_placeholder.info("🤖 Agent is thinking...")

        with st.spinner("Processing video..."):
            response = self.loop.run_until_complete(
                st.session_state.agent.run(f"Analyse this video: {safe_path}")
            )

        thinking_placeholder.empty()
        st.chat_message("assistant").write(response)

        # Clean up both copies to avoid filling the disk.
        for path in (video_path, safe_path):
            try:
                os.unlink(path)
            except OSError:
                pass

        # Prepare a fresh temp file for the next recording session.
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tmp.close()
        st.session_state.current_tmp_video = tmp.name

        # Reset state machine.
        st.session_state.recording_state = "idle"
        st.session_state.pending_video_path = None

        # Force a full rerun so the updated chat history is rendered cleanly.
        st.rerun()

    # ------------------------------------------------------------------
    # Main input handler  ── always runs in the MAIN column
    # ------------------------------------------------------------------

    def handle_input(self):
        """
        Render and process all user inputs.

        Layout contract
        ---------------
        record_video_ui() renders ONLY the WebRTC widget inside the sidebar.
        It never creates chat bubbles or status messages.

        Everything visible in the main column (chat bubbles, status messages,
        agent responses) is created HERE, ensuring it always appears on the
        right side of the page regardless of how it was triggered.
        """

        # Step 1 – Render the sidebar recorder (may call st.rerun() internally
        # during the stopping → ready transition; if so, the rest of this
        # method does not execute on the current cycle – that is intentional).
        self.record_video_ui()

        # Step 2 – If a video is ready, process it here in the main column.
        if st.session_state.recording_state == "ready":
            video_path = st.session_state.pending_video_path

            # Wait for aiortc to finish flushing the file.
            with st.spinner("⏳ Waiting for video file to be finalised..."):
                ready = self.wait_for_file_ready(
                    video_path,
                    timeout=15.0,
                    stable_for=0.8,
                )

            if not ready:
                st.error(
                    "❌ Timeout: the video file could not be finalised. "
                    "Please try again."
                )
                st.session_state.recording_state = "idle"
                st.session_state.pending_video_path = None
            else:
                # _send_video_to_agent renders all messages in the main column
                # and calls st.rerun() at the end.
                self._send_video_to_agent(video_path)

        # Step 3 – Text chat input (bottom of main column).
        user_input = st.chat_input("Ask a question...")
        if user_input:
            st.chat_message("user").write(user_input)

            thinking_placeholder = st.empty()
            thinking_placeholder.info("🤖 Agent is thinking...")

            with st.spinner("Thinking..."):
                response = self.loop.run_until_complete(
                    st.session_state.agent.run(user_input)
                )

            thinking_placeholder.empty()
            st.chat_message("assistant").write(response)

    def run(self):
        """Configure the Streamlit page and render all components."""
        st.set_page_config(page_title="MCP Agent – Video Input", layout="wide")
        st.title("MCP Agent")
        self.display_chat()
        self.handle_input()


if __name__ == "__main__":
    ui = ChatBotUI()
    ui.run()