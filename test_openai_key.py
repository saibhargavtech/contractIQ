"""
Simple OpenAI API Key Test Script
Run this to check if your API key is working
"""

import openai
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get API key from environment variable
API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is not set. Please set it in your .env file or environment.")

def test_openai_key():
    """Test if OpenAI API key is working"""
    try:
        # Set the API key
        openai.api_key = API_KEY
        
        print("Testing OpenAI API Key...")
        print(f"Key: {API_KEY[:20]}...{API_KEY[-10:]}")
        print("-" * 50)
        
        # Create OpenAI client
        from openai import OpenAI
        client = OpenAI(api_key=API_KEY)
        
        # Test with a simple completion
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": "Say 'API key is working' if you can read this."}
            ],
            max_tokens=50
        )
        
        print("SUCCESS! API Key is working")
        print(f"Response: {response.choices[0].message.content}")
        return True
        
    except openai.AuthenticationError as e:
        print("AUTHENTICATION ERROR")
        print(f"Error: {e}")
        print("The API key is invalid or expired")
        return False
        
    except openai.RateLimitError as e:
        print("RATE LIMIT ERROR")
        print(f"Error: {e}")
        print("You've hit the rate limit, try again later")
        return False
        
    except openai.APIConnectionError as e:
        print("CONNECTION ERROR")
        print(f"Error: {e}")
        print("Network/connection issue")
        return False
        
    except Exception as e:
        print("UNKNOWN ERROR")
        print(f"Error: {e}")
        print(f"Error type: {type(e).__name__}")
        return False

if __name__ == "__main__":
    print("OpenAI API Key Test")
    print("=" * 50)
    
    # Test the key
    success = test_openai_key()
    
    print("\n" + "=" * 50)
    if success:
        print("API Key is working! The issue might be in the Streamlit app.")
    else:
        print("API Key is not working. You need a new key.")
    
    print("\nTo run this test:")
    print("python test_openai_key.py")
