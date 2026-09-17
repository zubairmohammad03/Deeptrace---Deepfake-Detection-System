from google import genai
import time
import os

# Read Gemini API key from environment.
# Supports both backend-friendly and frontend-style variable names.
API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("VITE_GEMINI_API_KEY")

client = genai.Client(api_key=API_KEY)

def generate_response(prompt):
    for attempt in range(5):
        try:
            response = client.models.generate_content(
                model="gemini-flash-latest",
                contents=prompt
            )
            return response.text
        except Exception as e:
            print(f"Attempt {attempt+1} failed:", e)
            time.sleep(5)

    return "API failed after retries"