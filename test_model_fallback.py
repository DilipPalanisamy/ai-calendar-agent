from main import get_gemini_model_candidates


def test_prefers_supported_gemini_models():
    candidates = get_gemini_model_candidates("gemini-1.5-flash-latest")

    assert candidates[0] in ["gemini-3.5-flash-lite", "gemini-3.1-flash-lite", "gemini-flash-lite-latest", "gemini-3.5-flash"]
    assert "gemini-1.5-flash-latest" not in candidates
    assert "gemini-2.5-flash" not in candidates


if __name__ == "__main__":
    test_prefers_supported_gemini_models()
    print("model fallback test passed")

