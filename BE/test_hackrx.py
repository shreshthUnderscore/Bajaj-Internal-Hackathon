#!/usr/bin/env python3
"""
Test script for the HackRX endpoint
Tests the complete flow: PDF URL -> Document Processing -> Question Answering
"""

import requests
import json
import time
from typing import Dict, Any

def test_hackrx_endpoint():
    """Test the /hackrx/run endpoint with the provided sample"""
    
    # API endpoint
    url = "http://localhost:8000/hackrx/run"
    
    # Headers
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Authorization": "Bearer 52d676a251b57c9a1ac3603f7a67b3c960fba5de52faccd7267abcc49f9fcc50"
    }
    
    # Sample request payload
    payload = {
        "documents": "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D",
        "questions": [
            "What is the grace period for premium payment under the National Parivar Mediclaim Plus Policy?",
            "What is the waiting period for pre-existing diseases (PED) to be covered?",
            "Does this policy cover maternity expenses, and what are the conditions?",
            "What is the waiting period for cataract surgery?",
            "Are the medical expenses for an organ donor covered under this policy?",
            "What is the No Claim Discount (NCD) offered in this policy?",
            "Is there a benefit for preventive health check-ups?",
            "How does the policy define a 'Hospital'?",
            "What is the extent of coverage for AYUSH treatments?",
            "Are there any sub-limits on room rent and ICU charges for Plan A?"
        ]
    }
    
    print("🚀 Testing HackRX Endpoint")
    print("=" * 50)
    print(f"URL: {url}")
    print(f"Document URL: {payload['documents']}")
    print(f"Number of questions: {len(payload['questions'])}")
    print("\n⏳ Sending request...")
    
    start_time = time.time()
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=300)
        
        end_time = time.time()
        response_time = end_time - start_time
        
        print(f"✅ Response received in {response_time:.2f} seconds")
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("\n📊 Response Summary:")
            print(f"Document ID: {result.get('document_id', 'N/A')}")
            print(f"Processing Time: {result.get('processing_time', 'N/A'):.2f}s")
            print(f"Status: {result.get('status', 'N/A')}")
            print(f"Answers Count: {len(result.get('answers', []))}")
            
            print("\n📝 Answers:")
            for i, answer in enumerate(result.get('answers', []), 1):
                print(f"\n{i}. Q: {answer.get('question', '')[:80]}...")
                print(f"   A: {answer.get('answer', '')[:200]}...")
                print(f"   Confidence: {answer.get('confidence', 0):.2f}")
                print(f"   Sources: {len(answer.get('sources', []))}")
            
            return True
        else:
            print(f"❌ Error: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except requests.exceptions.Timeout:
        print("⏰ Request timed out (>300s)")
        return False
    except requests.exceptions.ConnectionError:
        print("🔌 Connection error - is the server running?")
        return False
    except Exception as e:
        print(f"💥 Unexpected error: {str(e)}")
        return False

def test_health_endpoint():
    """Test the health endpoint"""
    try:
        response = requests.get("http://localhost:8000/health", timeout=10)
        if response.status_code == 200:
            print("✅ Health check passed")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check error: {str(e)}")
        return False

if __name__ == "__main__":
    print("🧪 HackRX API Testing Suite")
    print("=" * 50)
    
    # Test health first
    print("\n1. Testing Health Endpoint...")
    health_ok = test_health_endpoint()
    
    if health_ok:
        print("\n2. Testing HackRX Endpoint...")
        success = test_hackrx_endpoint()
        
        if success:
            print("\n🎉 All tests passed!")
        else:
            print("\n❌ HackRX test failed!")
    else:
        print("\n❌ Server is not responding. Please start the server first:")
        print("   cd BE && uvicorn main:app --reload --port 8000")
