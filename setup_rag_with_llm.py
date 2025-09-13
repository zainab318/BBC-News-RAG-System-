"""
Complete setup script for RAG system with LLM
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
    """Complete setup process"""
    print("🚀 Setting up RAG System with LLM")
    print("=" * 50)
    
    # Step 1: Install dependencies
    print("\n📦 Step 1: Installing dependencies...")
    packages = [
        "pandas",
        "langchain", 
        "chromadb",
        "sentence-transformers",
        "openai",
        "tiktoken",
        "python-dotenv"
    ]
    
    for package in packages:
        if not run_command(f"pip install {package}", f"Installing {package}"):
            print(f"⚠️  Failed to install {package}, continuing...")
    
    # Step 2: Setup API key
    print("\n🔑 Step 2: Setting up OpenAI API key...")
    try:
        from setup_api_key import setup_api_key
        api_key = setup_api_key()
    except Exception as e:
        print(f"⚠️  API key setup failed: {e}")
        api_key = None
    
    # Step 3: Test the system
    print("\n🧪 Step 3: Testing the system...")
    if run_command("python test_rag.py", "Running system test"):
        print("✅ System test passed!")
    else:
        print("⚠️  System test failed, but you can still try running the main application")
    
    # Step 4: Instructions
    print("\n📋 Step 4: Next steps...")
    print("=" * 30)
    print("✅ Setup completed!")
    print("\nTo use the system:")
    print("1. Run: python main.py")
    print("2. Or run: python test_rag.py")
    print("\nCommands in the interactive mode:")
    print("- ask <question>     - Ask questions about news")
    print("- search <topic>     - Search for topics")
    print("- stats              - Show database stats")
    print("- help               - Show all commands")
    print("- quit               - Exit")
    
    if api_key:
        print("\n🤖 LLM features are enabled!")
        print("You'll get AI-generated answers using OpenAI's GPT model.")
    else:
        print("\n⚠️  LLM features are disabled.")
        print("Run 'python setup_api_key.py' to enable AI-generated answers.")

if __name__ == "__main__":
    main()
