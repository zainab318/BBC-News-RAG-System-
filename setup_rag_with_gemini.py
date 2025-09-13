"""
Complete setup script for RAG system with FREE Google Gemini
"""

import subprocess
import sys
import os

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def main():
    """Complete setup process with Gemini"""
    print("🚀 Setting up RAG System with FREE Google Gemini")
    print("=" * 55)
    
    # Step 1: Install dependencies
    print("\n📦 Step 1: Installing dependencies...")
    packages = [
        "pandas",
        "langchain", 
        "chromadb",
        "sentence-transformers",
        "langchain-google-genai",
        "google-generativeai",
        "python-dotenv"
    ]
    
    for package in packages:
        if not run_command(f"pip install {package}", f"Installing {package}"):
            print(f"⚠️  Failed to install {package}, continuing...")
    
    # Step 2: Setup Gemini API key
    print("\n🔑 Step 2: Setting up FREE Google Gemini API key...")
    print("Getting your FREE API key from Google...")
    print("1. Go to: https://makersuite.google.com/app/apikey")
    print("2. Sign in with your Google account")
    print("3. Click 'Create API Key'")
    print("4. Copy the generated key")
    print()
    
    try:
        from setup_gemini_key import setup_gemini_key
        api_key = setup_gemini_key()
    except Exception as e:
        print(f"⚠️  API key setup failed: {e}")
        api_key = None
    
    # Step 3: Test the system
    print("\n🧪 Step 3: Testing the system...")
    print("✅ System setup completed! You can now run the main application.")
    
    # Step 4: Instructions
    print("\n📋 Step 4: Next steps...")
    print("=" * 30)
    print("✅ Setup completed!")
    print("\nTo use the system:")
    print("1. Run: python main.py")
    print("\nCommands in the interactive mode:")
    print("- ask <question>     - Ask questions about news")
    print("- search <topic>     - Search for topics")
    print("- stats              - Show database stats")
    print("- help               - Show all commands")
    print("- quit               - Exit")
    
    if api_key:
        print("\n🤖 FREE Gemini LLM features are enabled!")
        print("You'll get AI-generated answers using Google's FREE Gemini model.")
        print("No payment required! 🎉")
    else:
        print("\n⚠️  LLM features are disabled.")
        print("Run 'python setup_gemini_key.py' to enable FREE AI-generated answers.")

if __name__ == "__main__":
    main()
