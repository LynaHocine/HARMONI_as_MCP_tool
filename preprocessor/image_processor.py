import requests
import base64
import mimetypes

class ImagePreprocessor:
    """Handles images for multimodal input"""

    @staticmethod
    def image_to_base64(source) -> str:
        """Fetch image from url, local file path or raw bytes and return base64 data uri string"""

        # raw bytes passed as input ( from a video frame)
        if isinstance(source, bytes):
            base64_str = base64.b64encode(source).decode('utf-8')
        
        #url passed as input 
        if source.startswith("http://") or source.startswith("https://"):
            #send get request to retrieve the image
            response = requests.get(source, timeout=10)

            #if the request is successful
            if response.status_code != 200:
                raise Exception(f"Failed to fetch image: {response.status_code}")
            content_type = response.headers.get("Content-Type","")

            if not content_type.startswith("image/"):
                raise Exception(f"URL does not point to an image. Content-type : {content_type}")
            
            #encode image into Base64 string
            base64_str = base64.b64encode(response.content).decode('utf-8')
            #return properly formatted data URI for multimodal LLM usage
            return f"data:{content_type};base64,{base64_str}"
        
        #local file path passed as input
        content_type, _ = mimetypes.guess_type(source)
        if not content_type or not content_type.startswith("image/"):
            raise Exception(f"File is not an image: {source}")
        with open(source, "rb") as f:
            base64_str = base64.b64encode(f.read()).decode('utf-8')
        return f"data:{content_type};base64,{base64_str}"