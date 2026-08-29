from starlette.testclient import TestClient
from main import app

client = TestClient(app)

def test_endpoints():
    print("Testing GET / (UI)...", flush=True)
    res_root = client.get("/")
    assert res_root.status_code == 200, f"Expected 200, got {res_root.status_code}"
    assert "AI Calendar Agent" in res_root.text, "Root page missing title"
    print("[PASS] GET / (200 OK)", flush=True)

    print("Testing GET /api/me (Unauthenticated)...", flush=True)
    res_me = client.get("/api/me")
    assert res_me.status_code == 200, f"Expected 200, got {res_me.status_code}"
    data = res_me.json()
    assert data.get("authenticated") is False, f"Expected authenticated=False, got {data}"
    print("[PASS] GET /api/me (200 OK, authenticated=False)", flush=True)

    print("Testing POST /api/chat (Unauthenticated)...", flush=True)
    res_chat = client.post("/api/chat", json={"message": "hello"})
    assert res_chat.status_code == 401, f"Expected 401 Unauthorized, got {res_chat.status_code}"
    print("[PASS] POST /api/chat (401 Unauthorized)", flush=True)

    print("Testing GET /logout...", flush=True)
    res_logout = client.get("/logout", follow_redirects=False)
    assert res_logout.status_code == 303, f"Expected 303 See Other, got {res_logout.status_code}"
    print("[PASS] GET /logout (303 Redirect to /)", flush=True)

    print("\n==========================================", flush=True)
    print("ALL FASTAPI ENDPOINT TESTS PASSED 100%!", flush=True)
    print("==========================================", flush=True)

if __name__ == "__main__":
    test_endpoints()
