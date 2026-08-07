from main import get_gemini_model_candidates


def test_prefers_supported_gemini_models():
    candidates = get_gemini_model_candidates("gemini-1.5-flash-latest")

    assert candidates[0] == "gemini-3.5-flash"
    assert "gemini-1.5-flash-latest" not in candidates
