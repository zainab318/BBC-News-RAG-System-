"""
Setup script for Google Gemini API key
"""

import os

def setup_gemini_key():
    """Setup Google Gemini API key"""
    print("🔑 Google Gemini API Key Setup")
    print("=" * 35)
    
    # Check if API key already exists
    existing_key = os.getenv('GOOGLE_API_KEY')
    if existing_key:
        print(f"✅ API key already set: {existing_key[:8]}...")
        return existing_key
    
    # Get API key from user
    print("To use the LLM features, you need a Google Gemini API key.")
    print("Get your FREE API key from: https://makersuite.google.com/app/apikey")
    print()
    print("Steps to get your free API key:")
    print("1. Go to https://makersuite.google.com/app/apikey")
    print("2. Sign in with your Google account")
    print("3. Click 'Create API Key'")
    print("4. Copy the generated key")
    print()
    
    api_key = input("Enter your Google Gemini API key: ").strip()
    
    if not api_key:
        print("❌ No API key provided. LLM features will be disabled.")
        return None
    
    # Set environment variable for current session
    os.environ['GOOGLE_API_KEY'] = api_key
    print("✅ API key set for current session")
    
    # Save to .env file for future sessions
    try:
        with open('.env', 'w') as f:
            f.write(f"GOOGLE_API_KEY={api_key}\n")
        print("✅ API key saved to .env file")
    except Exception as e:
        print(f"⚠️  Could not save to .env file: {e}")
    
    return api_key

if __name__ == "__main__":
    setup_gemini_key()
