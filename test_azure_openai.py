"""
Simple Azure OpenAI API Key Test Script
Tests if Azure OpenAI is configured correctly and working
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get Azure OpenAI configuration
api_key = os.getenv("OPENAI_API_KEY")
azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "")
azure_api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview")
use_azure = bool(azure_endpoint)

print("=" * 60)
print("Azure OpenAI Configuration Test")
print("=" * 60)
print()

# Check configuration
if not api_key:
    print("❌ ERROR: OPENAI_API_KEY not found in .env file")
    exit(1)
else:
    print(f"✅ API Key found: {api_key[:20]}...{api_key[-10:]}")

if not use_azure:
    print("❌ ERROR: AZURE_OPENAI_ENDPOINT not found in .env file")
    print("   Make sure AZURE_OPENAI_ENDPOINT is set in your .env file")
    exit(1)
else:
    print(f"✅ Azure Endpoint found: {azure_endpoint}")

print(f"✅ API Version: {azure_api_version}")
print()

# Extract deployment name and base URL
if "/deployments/" in azure_endpoint:
    parts = azure_endpoint.split("/deployments/")
    azure_base_url = parts[0]
    deployment_part = parts[1].split("/")[0]  # Get just the deployment name, before any /chat/completions
    if "?" in deployment_part:
        deployment_part = deployment_part.split("?")[0]
    azure_deployment = deployment_part
else:
    azure_base_url = azure_endpoint
    azure_deployment = os.getenv("AZURE_DEPLOYMENT_NAME", "gpt-4.1")

# Try base URL without /openai if it ends with /openai
if azure_base_url.endswith("/openai"):
    azure_base_url_alt = azure_base_url[:-7]  # Remove /openai
else:
    azure_base_url_alt = None

print(f"✅ Base URL: {azure_base_url}")
if azure_base_url_alt:
    print(f"   Alternative Base URL (without /openai): {azure_base_url_alt}")
print(f"✅ Deployment Name: {azure_deployment}")
print()

# Test Azure OpenAI connection
print("Testing Azure OpenAI connection...")
print("-" * 60)

try:
    from openai import AzureOpenAI
    
    # Try with the original base URL first
    test_base_urls = [azure_base_url]
    if azure_base_url_alt:
        test_base_urls.append(azure_base_url_alt)
    
    success = False
    last_error = None
    
    for test_url in test_base_urls:
        try:
            print(f"Trying base URL: {test_url}")
            client = AzureOpenAI(
                api_key=api_key,
                api_version=azure_api_version,
                azure_endpoint=test_url
            )
            
            print("✅ AzureOpenAI client created successfully")
            print()
            print("Sending test message...")
            
            response = client.chat.completions.create(
                model=azure_deployment,
                messages=[
                    {"role": "user", "content": "Say 'Azure OpenAI is working!' if you can read this."}
                ],
                max_tokens=50
            )
            
            print("✅ SUCCESS! Azure OpenAI is working!")
            print()
            print(f"Response: {response.choices[0].message.content}")
            print()
            print(f"✅ Working Base URL: {test_url}")
            print()
            print("=" * 60)
            print("✅ All tests passed! Your Azure OpenAI configuration is correct.")
            print("=" * 60)
            success = True
            break
            
        except Exception as e:
            last_error = e
            print(f"❌ Failed with base URL {test_url}: {str(e)}")
            print()
            continue
    
    if not success:
        raise last_error
    
except Exception as e:
    print()
    print("❌ ERROR: Azure OpenAI test failed")
    print(f"Error Type: {type(e).__name__}")
    print(f"Error Message: {str(e)}")
    print()
    print("Troubleshooting:")
    print("1. Check that your API key is correct")
    print("2. Check that your endpoint URL is correct")
    print("3. Check that your deployment name matches the endpoint")
    print("4. Check that your Azure OpenAI resource is active")
    print("5. Check network connectivity")
    print()
    print("=" * 60)
    exit(1)

