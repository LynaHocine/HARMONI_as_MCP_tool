import requests
import base64

class ImagePreprocessor:
    """Handles images for multimodal input"""

    @staticmethod
    def image_to_base64(url : str) -> str:
        """Fetch image from url and return base64 data uri string"""

        #send get request to retrieve the image
        response = requests.get(url, timeout=10)

        #if the request is successful
        if response.status_code == 200:
            #verify file type 
            content_type = response.headers.get("Content-Type","")

            if not content_type.startswith("image/"):
                raise Exception(f"URL does not point to an image. Content-type : {content_type}")
            
            #encode image into Base64 string
            base64_str = base64.b64encode(response.content).decode('utf-8')
            #return properly formatted data URI for multimodal LLM usage
            return f"data:{content_type};base64,{base64_str}"
        
        else: 
            raise Exception(f"Failed to fetch image :{response.status_code}")
    