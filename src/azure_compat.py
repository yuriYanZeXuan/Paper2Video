import os
import sys
import json
import base64
import requests
import io
from PIL import Image

# Minimal mock for Camel response
class MockMessage:
    def __init__(self, content):
        self.content = content

class MockResponse:
    def __init__(self, content, usage=None):
        self.msgs = [MockMessage(content)]
        self.msg = self.msgs[0]
        self.info = {"usage": usage or {}}

def encode_pil_image_to_data_url(image: Image.Image) -> str:
    """
    Convert PIL Image to data URL (base64).
    """
    buffered = io.BytesIO()
    # Convert to RGB if needed
    if image.mode in ("RGBA", "P"):
        image = image.convert("RGB")
    image.save(buffered, format="JPEG")
    b64 = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/jpeg;base64,{b64}"

def load_env_api_key() -> str:
    """
    Read API key from environment.
    """
    api_key = os.getenv("RUNWAY_API_KEY") or os.getenv("OPENAI_API_KEY") or os.getenv("AZURE_OPENAI_API_KEY") or ""
    return api_key.strip()

def call_chat_completions(
    model: str,
    messages: list,
    temperature: float = 0.2,
    max_tokens: int = 4096,
    attempts: int = 3,
) -> str:
    """
    Azure/Runway compatible chat completions call.
    """
    # Default base URL for Azure/Runway if not provided
    base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("RUNWAY_API_BASE")
    if not base_url:
         # Fallback or raise error? The user said they have Azure key. 
         # They usually also have an endpoint.
         # If they provided export commands without endpoint, maybe they expect standard OpenAI URL?
         # But user said "Azure gpt 4o". Azure usually requires specific endpoint.
         # I will assume they might set OPENAI_BASE_URL or I'll use a placeholder they can change.
         pass 
    
    if not base_url:
        print("Warning: OPENAI_BASE_URL not set. Using default OpenAI endpoint, but for Azure you likely need a custom endpoint.")
        base_url = "https://api.openai.com/v1"

    # Adjust endpoint construction for Azure vs Standard
    # Standard: /v1/chat/completions
    # Azure: /openai/deployments/{deployment-id}/chat/completions?api-version={api-version}
    # The user's example uses: f"{base_url.rstrip('/')}/chat/completions?api-version={api_version}"
    # This looks like a specific corporate proxy or Runway style. I will stick to the user's reference implementation.
    
    api_version = os.getenv("RUNWAY_API_VERSION") or "2024-12-01-preview"
    api_key = load_env_api_key()
    
    if not api_key:
        raise RuntimeError("missing OPENAI_API_KEY (or AZURE_OPENAI_API_KEY) in environment")

    endpoint = f"{base_url.rstrip('/')}/chat/completions?api-version={api_version}"
    
    headers = {
        "api-key": api_key, # Azure often uses api-key header, or Authorization: Bearer
        "Content-Type": "application/json",
    }
    # Some proxies accept Authorization header too
    if "api.openai.com" in base_url:
         headers["Authorization"] = f"Bearer {api_key}"
         endpoint = f"{base_url.rstrip('/')}/chat/completions" # Standard OpenAI doesn't use api-version query param typically in same way

    body = {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "messages": messages,
    }

    last_err = None
    for i in range(max(1, attempts)):
        try:
            r = requests.post(endpoint, headers=headers, json=body, timeout=120)
            if r.status_code < 200 or r.status_code >= 300:
                # try with Authorization header if api-key failed?
                if r.status_code == 401 and "Authorization" not in headers:
                     headers["Authorization"] = f"Bearer {api_key}"
                     continue
                raise RuntimeError(f"HTTP {r.status_code}: {r.text[:800]}")
            data = r.json()
            choices = data.get("choices") or []
            content = (choices[0].get("message", {}).get("content") if choices else "") or ""
            if not content:
                raise RuntimeError("empty completion content")
            return content
        except Exception as e:
            last_err = e
            print(f"[AzureAgent] Attempt {i+1} failed: {e}")
    
    raise RuntimeError(f"chat completions failed: {last_err}")

class AzureCamelAgent:
    def __init__(self, model_type, model_config_dict=None):
        self.model_name = model_type
        self.model_config = model_config_dict or {}
        self.system_message = "You are a helpful assistant."

    def reset(self):
        pass

    def step(self, input_message):
        """
        Mimics Camel's agent.step() but uses direct HTTP call.
        input_message: Camel BaseMessage
        """
        # Construct messages list
        msgs = []
        if self.system_message:
            msgs.append({"role": "system", "content": self.system_message})
        
        # Parse user message
        # Camel BaseMessage has .content, .role_name, and potentially .image_list
        content = input_message.content
        role = input_message.role_name.lower() if hasattr(input_message, 'role_name') else "user"
        if role == "user":
            user_content = []
            user_content.append({"type": "text", "text": content})
            
            # Handle images
            if hasattr(input_message, 'image_list') and input_message.image_list:
                for img in input_message.image_list:
                    if img:
                        data_url = encode_pil_image_to_data_url(img)
                        user_content.append({"type": "image_url", "image_url": {"url": data_url}})
            
            msgs.append({"role": "user", "content": user_content})
        else:
            msgs.append({"role": role, "content": content})

        # Call API
        # Map model name if necessary. User said "Azure gpt 4o".
        model_id = self.model_name
        if "azure" in model_id:
             # Extract actual model name if encoded like 'azure-gpt-4o' -> 'gpt-4o'
             # Or just pass it through if the user configured the deployment name as the model name.
             pass

        response_text = call_chat_completions(
            model=model_id,
            messages=msgs,
            temperature=self.model_config.get("temperature", 0.2),
            max_tokens=self.model_config.get("max_tokens", 4096)
        )

        return MockResponse(response_text)

