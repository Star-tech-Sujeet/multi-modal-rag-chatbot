"""
Gemini Live Smoke Test Script.
Tests real Google Gemini text generation and embeddings using the official modern google-genai SDK.
Safe execution: never prints, logs, or leaks the API key under any circumstance.
"""

import os
import sys
import re
from typing import Optional
from dotenv import load_dotenv

# 1. Load .env
load_dotenv()


def sanitize_error(msg: str, key: Optional[str] = None) -> str:
    """Sanitize error messages to prevent API key leakage."""
    if not msg:
        return ""
    sanitized = str(msg)
    if key and len(key) >= 5:
        sanitized = sanitized.replace(key, "[REDACTED_API_KEY]")
    # Mask any Google API key pattern
    sanitized = re.sub(r'AIza[0-9A-Za-z_-]{30,}', '[REDACTED_API_KEY]', sanitized)
    return sanitized


def main():
    print("=" * 60)
    print("Aura AI - Gemini Provider Live Smoke Test")
    print("=" * 60)

    # 2. Read GEMINI_API_KEY without printing it
    gemini_api_key = os.getenv("GEMINI_API_KEY", "").strip()
    is_placeholder = (
        not gemini_api_key
        or gemini_api_key.startswith("your")
        or "placeholder" in gemini_api_key.lower()
        or gemini_api_key == "your_gemini_api_key_here"
    )

    if is_placeholder:
        print("Gemini API key: MISSING")
        print("Gemini generation: FAIL (GEMINI_API_KEY is not configured or is a placeholder in .env)")
        print("Gemini embedding: FAIL (GEMINI_API_KEY is not configured or is a placeholder in .env)")
        print("=" * 60)
        print("RESULT: Gemini smoke test could not run because GEMINI_API_KEY is missing.")
        print("=" * 60)
        sys.exit(1)

    print("Gemini API key: LOADED")

    # 3. Model Configuration (matching Aura AI defaults in app/config.py)
    gemini_model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip()
    gemini_embedding_model = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001").strip()

    print(f"Configured Generation Model: {gemini_model}")
    print(f"Configured Embedding Model: {gemini_embedding_model}")

    # 4. Initialize official google-genai SDK
    try:
        from google import genai
        from google.genai import types
        client = genai.Client(api_key=gemini_api_key)
    except Exception as e:
        err_sanitized = sanitize_error(str(e), gemini_api_key)
        print(f"Failed to initialize google-genai Client: {type(e).__name__}: {err_sanitized}")
        print("Gemini generation: FAIL")
        print("Gemini embedding: FAIL")
        sys.exit(1)

    # 5. Test Real Gemini Text Generation
    gen_success = False
    try:
        test_prompt = "Reply with exactly the single word 'OK'."
        config = types.GenerateContentConfig(
            temperature=0.0,
            max_output_tokens=1000
        )
        response = client.models.generate_content(
            model=gemini_model,
            contents=test_prompt,
            config=config
        )
        resp_text = (response.text or "").strip()
        if resp_text:
            print("Gemini generation: PASS")
            gen_success = True
        else:
            print("Gemini generation: FAIL (Empty response text)")
    except Exception as e:
        err_sanitized = sanitize_error(str(e), gemini_api_key)
        print("Gemini generation: FAIL")
        print(f"  Error Type: {type(e).__name__}")
        print(f"  Error Detail: {err_sanitized}")

    # 6. Test Real Gemini Embedding
    emb_success = False
    try:
        test_text = "Aura AI multi-modal retrieval augmented generation test query."
        config = types.EmbedContentConfig(task_type="RETRIEVAL_QUERY")
        emb_response = client.models.embed_content(
            model=gemini_embedding_model,
            contents=test_text,
            config=config
        )

        if emb_response and emb_response.embeddings and len(emb_response.embeddings) > 0:
            first_emb = emb_response.embeddings[0]
            vector_values = first_emb.values
            # Verify numeric float values
            if vector_values and len(vector_values) > 0 and all(isinstance(v, (float, int)) for v in vector_values[:10]):
                dim = len(vector_values)
                print("Gemini embedding: PASS")
                print(f"  Embedding dimension: {dim}")
                emb_success = True
            else:
                print("Gemini embedding: FAIL (Invalid or non-numeric embedding values returned)")
        else:
            print("Gemini embedding: FAIL (No embeddings in response)")
    except Exception as e:
        err_sanitized = sanitize_error(str(e), gemini_api_key)
        print("Gemini embedding: FAIL")
        print(f"  Error Type: {type(e).__name__}")
        print(f"  Error Detail: {err_sanitized}")

    print("=" * 60)
    if gen_success and emb_success:
        print("RESULT: ALL REAL GEMINI API CHECKS PASSED SUCCESSFULLY")
        print("=" * 60)
        sys.exit(0)
    else:
        print("RESULT: ONE OR MORE REAL GEMINI API CHECKS FAILED")
        print("=" * 60)
        sys.exit(1)


if __name__ == "__main__":
    main()
