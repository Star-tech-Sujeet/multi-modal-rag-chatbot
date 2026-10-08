#!/usr/bin/env python3
"""
Comprehensive test script for the Multi-Modal RAG application.
Tests API endpoints, document processing, and query functionality.
"""

import requests
import json
import time
import sys
import os
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from dotenv import load_dotenv


# Load environment variables from .env and .env.example
load_dotenv()
if not Path('.env').exists():
    load_dotenv('.env.example')

# =============================================================================
# Configuration
# =============================================================================

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000/api/v1")
TEST_FILES = {
    "text": "sample_docs/sample.txt",
    "csv": "sample_docs/sample.csv"
}
TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "30"))  # seconds


# =============================================================================
# Test Result Tracking
# =============================================================================

@dataclass
class TestResult:
    """Represents a single test result."""
    name: str
    passed: bool
    message: str
    duration_ms: Optional[float] = None
    details: Optional[Dict[str, Any]] = None


class TestRunner:
    """Manages test execution and reporting."""
    
    def __init__(self):
        self.results: List[TestResult] = []
    
    def add_result(self, result: TestResult):
        """Add a test result."""
        self.results.append(result)
        status = "✅ PASSED" if result.passed else "❌ FAILED"
        print(f"{status}: {result.name}")
        print(f"   {result.message}")
        if result.duration_ms:
            print(f"   Duration: {result.duration_ms:.2f}ms")
        if result.details and not result.passed:
            print(f"   Details: {json.dumps(result.details, indent=2)}")
        print()
    
    def summary(self):
        """Print test summary."""
        passed = sum(1 for r in self.results if r.passed)
        failed = sum(1 for r in self.results if not r.passed)
        total = len(self.results)
        
        print("=" * 60)
        print("TEST SUMMARY")
        print("=" * 60)
        print(f"Total: {total} | Passed: {passed} | Failed: {failed}")
        
        if failed > 0:
            print("\nFailed Tests:")
            for r in self.results:
                if not r.passed:
                    print(f"  - {r.name}: {r.message}")
        
        return failed == 0


runner = TestRunner()


# =============================================================================
# Test Functions
# =============================================================================

def test_api_root():
    """Test the root endpoint."""
    start = time.time()
    try:
        response = requests.get(f"{API_BASE_URL.replace('/api/v1', '')}/", timeout=TIMEOUT)
        duration = (time.time() - start) * 1000
        
        if response.status_code == 200:
            data = response.json()
            runner.add_result(TestResult(
                name="API Root Endpoint",
                passed=True,
                message=f"API version: {data.get('version', 'unknown')}",
                duration_ms=duration,
                details=data
            ))
            return True
        else:
            runner.add_result(TestResult(
                name="API Root Endpoint",
                passed=False,
                message=f"Unexpected status code: {response.status_code}",
                duration_ms=duration
            ))
            return False
    except Exception as e:
        runner.add_result(TestResult(
            name="API Root Endpoint",
            passed=False,
            message=f"Exception: {str(e)}"
        ))
        return False


def test_api_health():
    """Test the health check endpoint."""
    start = time.time()
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=TIMEOUT)
        duration = (time.time() - start) * 1000
        
        if response.status_code == 200:
            data = response.json()
            status = data.get("status", "unknown")
            docs = data.get("vectorstore", {}).get("total_documents", 0)
            
            runner.add_result(TestResult(
                name="API Health Check",
                passed=status in ["healthy", "degraded"],
                message=f"Status: {status}, Documents: {docs}",
                duration_ms=duration,
                details=data
            ))
            return status == "healthy"
        else:
            runner.add_result(TestResult(
                name="API Health Check",
                passed=False,
                message=f"Status code: {response.status_code}",
                duration_ms=duration
            ))
            return False
    except Exception as e:
        runner.add_result(TestResult(
            name="API Health Check",
            passed=False,
            message=f"Exception: {str(e)}"
        ))
        return False


def test_file_upload(file_type: str = "text") -> Optional[str]:
    """Test file upload functionality."""
    test_file = TEST_FILES.get(file_type)
    
    if not test_file or not Path(test_file).exists():
        runner.add_result(TestResult(
            name=f"File Upload ({file_type})",
            passed=False,
            message=f"Test file not found: {test_file}"
        ))
        return None
    
    start = time.time()
    try:
        with open(test_file, 'rb') as f:
            files = {'file': (Path(test_file).name, f, 'text/plain')}
            response = requests.post(f"{API_BASE_URL}/upload", files=files, timeout=TIMEOUT)
        
        duration = (time.time() - start) * 1000
        
        if response.status_code == 200:
            data = response.json()
            file_id = data.get('file_id')
            chunks = data.get('chunks_created', 'unknown')
            
            runner.add_result(TestResult(
                name=f"File Upload ({file_type})",
                passed=True,
                message=f"File ID: {file_id[:8]}..., Chunks: {chunks}",
                duration_ms=duration,
                details=data
            ))
            return file_id
        else:
            runner.add_result(TestResult(
                name=f"File Upload ({file_type})",
                passed=False,
                message=f"Status: {response.status_code}",
                duration_ms=duration,
                details=response.json() if response.text else None
            ))
            return None
    except Exception as e:
        runner.add_result(TestResult(
            name=f"File Upload ({file_type})",
            passed=False,
            message=f"Exception: {str(e)}"
        ))
        return None


def test_query(file_id: str, question: str = "What is this document about?") -> bool:
    """Test query functionality."""
    if not file_id:
        runner.add_result(TestResult(
            name="Document Query",
            passed=False,
            message="No file ID provided (upload may have failed)"
        ))
        return False
    
    start = time.time()
    try:
        payload = {
            "question": question,
            "file_id": file_id,
            "max_sources": 3
        }
        
        response = requests.post(
            f"{API_BASE_URL}/query",
            headers={"Content-Type": "application/json"},
            data=json.dumps(payload),
            timeout=TIMEOUT
        )
        
        duration = (time.time() - start) * 1000
        
        if response.status_code == 200:
            data = response.json()
            answer_preview = data.get('answer', '')[:100]
            sources_count = len(data.get('sources', []))
            model = data.get('model_used', 'unknown')
            
            runner.add_result(TestResult(
                name="Document Query",
                passed=True,
                message=f"Model: {model}, Sources: {sources_count}",
                duration_ms=duration,
                details={"answer_preview": answer_preview + "..."}
            ))
            return True
        else:
            runner.add_result(TestResult(
                name="Document Query",
                passed=False,
                message=f"Status: {response.status_code}",
                duration_ms=duration,
                details=response.json() if response.text else None
            ))
            return False
    except Exception as e:
        runner.add_result(TestResult(
            name="Document Query",
            passed=False,
            message=f"Exception: {str(e)}"
        ))
        return False


def test_list_files():
    """Test file listing endpoint."""
    start = time.time()
    try:
        response = requests.get(f"{API_BASE_URL}/files", timeout=TIMEOUT)
        duration = (time.time() - start) * 1000
        
        if response.status_code == 200:
            data = response.json()
            file_count = len(data)
            
            runner.add_result(TestResult(
                name="List Files",
                passed=True,
                message=f"Found {file_count} file(s)",
                duration_ms=duration
            ))
            return True
        else:
            runner.add_result(TestResult(
                name="List Files",
                passed=False,
                message=f"Status: {response.status_code}",
                duration_ms=duration
            ))
            return False
    except Exception as e:
        runner.add_result(TestResult(
            name="List Files",
            passed=False,
            message=f"Exception: {str(e)}"
        ))
        return False


def test_invalid_file_type():
    """Test upload rejection for invalid file types."""
    start = time.time()
    try:
        # Try to upload a file with invalid extension
        files = {'file': ('test.xyz', b'test content', 'application/octet-stream')}
        response = requests.post(f"{API_BASE_URL}/upload", files=files, timeout=TIMEOUT)
        duration = (time.time() - start) * 1000
        
        # Should be rejected with 400
        if response.status_code == 400:
            runner.add_result(TestResult(
                name="Invalid File Type Rejection",
                passed=True,
                message="Invalid file type correctly rejected",
                duration_ms=duration
            ))
            return True
        else:
            runner.add_result(TestResult(
                name="Invalid File Type Rejection",
                passed=False,
                message=f"Expected 400, got {response.status_code}",
                duration_ms=duration
            ))
            return False
    except Exception as e:
        runner.add_result(TestResult(
            name="Invalid File Type Rejection",
            passed=False,
            message=f"Exception: {str(e)}"
        ))
        return False


def test_empty_query():
    """Test query validation for empty question."""
    start = time.time()
    try:
        payload = {
            "question": "   ",  # Whitespace only
            "file_id": "test-id"
        }
        
        response = requests.post(
            f"{API_BASE_URL}/query",
            headers={"Content-Type": "application/json"},
            data=json.dumps(payload),
            timeout=TIMEOUT
        )
        duration = (time.time() - start) * 1000
        
        # Should be rejected with 400 or 422
        if response.status_code in [400, 422]:
            runner.add_result(TestResult(
                name="Empty Query Validation",
                passed=True,
                message="Empty query correctly rejected",
                duration_ms=duration
            ))
            return True
        else:
            runner.add_result(TestResult(
                name="Empty Query Validation",
                passed=False,
                message=f"Expected 400/422, got {response.status_code}",
                duration_ms=duration
            ))
            return False
    except Exception as e:
        runner.add_result(TestResult(
            name="Empty Query Validation",
            passed=False,
            message=f"Exception: {str(e)}"
        ))
        return False


# =============================================================================
# Main Test Runner
# =============================================================================

def main():
    """Run all tests."""
    print("=" * 60)
    print("🧪 Multi-Modal RAG Application Test Suite")
    print("=" * 60)
    print(f"API URL: {API_BASE_URL}")
    print(f"Timeout: {TIMEOUT}s")
    print("=" * 60)
    print()
    
    # Check if API is accessible first
    if not test_api_root():
        print("\n❌ API is not accessible. Make sure the backend is running:")
        print("   docker-compose up --build")
        print("   OR")
        print("   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000")
        sys.exit(1)
    
    # Run tests
    test_api_health()
    test_list_files()
    test_invalid_file_type()
    test_empty_query()
    
    # Upload and query test
    file_id = test_file_upload("text")
    if file_id:
        test_query(file_id, "What are the main topics in this document?")
        test_query(file_id, "Summarize the key points about machine learning")
    
    # CSV upload test
    test_file_upload("csv")
    
    # Print summary
    print()
    success = runner.summary()
    
    print("\n" + "=" * 60)
    print("📋 Next Steps")
    print("=" * 60)
    print("1. Open http://localhost:8501 for the Streamlit UI")
    print("2. Open http://localhost:8000/docs for API documentation")
    print("3. Try uploading different file types")
    print("4. Test multi-modal queries with images")
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
