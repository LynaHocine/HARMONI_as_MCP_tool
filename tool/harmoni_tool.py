import threading
import requests

HARMONI_BASE_URL = "http://127.0.0.1:8000"

class HarmoniTool:

    @staticmethod
    def harmoni_tool(video_path: str, query:str) -> str:
        """Sends video to Harmoni to get user context, updates memory in parallel.
        Returns the context as a string for the agent to use"""

        #Call /set_video to get user information 
        with open(video_path, "rb") as f:
            response = requests.post(
                f"{HARMONI_BASE_URL}/set_video",
                files={"video": f}
            )

        if response.status_code != 200:
            return f"Harmoni /set_video failed: {response.status_code} - {response.text}"
        
        data = response.json()
        detected_user = data.get("detected_user", "unknown")
        profile = data.get("profile",[])
        emotion = data.get("emotion", "unknown")
        transcript=data.get("transcript","")

        #Call /answer in to update memory in parallel (we ignore the response)
        def _update_memory():
            requests.post(
                f"{HARMONI_BASE_URL}/answer",
                data={
                    "emotion": emotion,
                    "current_user": detected_user,
                    "question": query,
                }
            )
        threading.Thread(target=_update_memory, daemon=True).start()

        #Format context for the agent to use 
        lines = [
            f"Detected user: {detected_user}",
            f"Emotion: {emotion}",
            f"Transcript: {transcript}",
            f"Profile:",
        ]
        
        if profile:
            for feature in profile: 
                lines.append(f"- {feature.get('name')}: {feature.get('value')}")
        else: 
            lines.append(" No profile data found.")
        
        return "\n".join(lines)