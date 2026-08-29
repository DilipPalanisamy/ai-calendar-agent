from starlette.testclient import TestClient
from main import (
    app,
    create_chat_session,
    save_chat_message,
    get_user_chat_sessions,
    get_chat_session_details,
    delete_chat_session,
    delete_all_user_sessions,
)

client = TestClient(app)

def test_endpoints():
    print("Testing GET / (Unauthenticated Gated Redirect)...", flush=True)
    res_root = client.get("/", follow_redirects=False)
    assert res_root.status_code == 307, f"Expected 307 Redirect, got {res_root.status_code}"
    assert res_root.headers.get("location") == "/login", f"Expected location /login, got {res_root.headers.get('location')}"
    print("[PASS] GET / (307 Redirect to /login)", flush=True)

    print("Testing GET /login (Login Page UI)...", flush=True)
    res_login = client.get("/login")
    assert res_login.status_code == 200, f"Expected 200 OK, got {res_login.status_code}"
    assert "Sign in with Google" in res_login.text, "Login page missing Google Sign-in button"
    print("[PASS] GET /login (200 OK, Login Page rendered)", flush=True)

    print("Testing GET /api/me (Unauthenticated)...", flush=True)
    res_me = client.get("/api/me")
    assert res_me.status_code == 200, f"Expected 200, got {res_me.status_code}"
    data = res_me.json()
    assert data.get("authenticated") is False, f"Expected authenticated=False, got {data}"
    print("[PASS] GET /api/me (200 OK, authenticated=False)", flush=True)

    print("Testing POST /api/chat (Unauthenticated Guard)...", flush=True)
    res_chat = client.post("/api/chat", json={"message": "hello"})
    assert res_chat.status_code == 401, f"Expected 401 Unauthorized, got {res_chat.status_code}"
    print("[PASS] POST /api/chat (401 Unauthorized)", flush=True)

    print("Testing GET /api/history (Unauthenticated Guard)...", flush=True)
    res_hist = client.get("/api/history")
    assert res_hist.status_code == 401, f"Expected 401 Unauthorized, got {res_hist.status_code}"
    print("[PASS] GET /api/history (401 Unauthorized)", flush=True)

    print("Testing DELETE /api/history (Unauthenticated Guard)...", flush=True)
    res_del = client.delete("/api/history")
    assert res_del.status_code == 401, f"Expected 401 Unauthorized, got {res_del.status_code}"
    print("[PASS] DELETE /api/history (401 Unauthorized)", flush=True)

    print("Testing GET /logout (Redirect to /login)...", flush=True)
    res_logout = client.get("/logout", follow_redirects=False)
    assert res_logout.status_code == 303, f"Expected 303 See Other, got {res_logout.status_code}"
    assert res_logout.headers.get("location") == "/login", f"Expected location /login, got {res_logout.headers.get('location')}"
    print("[PASS] GET /logout (303 Redirect to /login)", flush=True)

    # SQLite Database Unit Tests
    print("Testing SQLite Chat History Database CRUD...", flush=True)
    test_user = "test_user@example.com"
    delete_all_user_sessions(test_user)

    # Create session
    session_id = create_chat_session(test_user, "Test Meeting Chat")
    assert session_id is not None, "Failed creating session"

    # Save messages
    save_chat_message(session_id, "user", "Schedule lunch tomorrow at 1 PM")
    save_chat_message(session_id, "assistant", "I have scheduled your lunch tomorrow at 1:00 PM.")

    # Retrieve sessions
    sessions = get_user_chat_sessions(test_user)
    assert len(sessions) == 1, f"Expected 1 session, got {len(sessions)}"
    assert sessions[0]["title"] == "Test Meeting Chat"

    # Retrieve session details with messages
    details = get_chat_session_details(session_id, test_user)
    assert details is not None
    assert len(details["messages"]) == 2
    assert details["messages"][0]["sender"] == "user"
    assert details["messages"][1]["sender"] == "assistant"

    # Delete session
    del_ok = delete_chat_session(session_id, test_user)
    assert del_ok is True
    assert get_chat_session_details(session_id, test_user) is None
    print("[PASS] SQLite Chat History CRUD tests completed successfully!", flush=True)

    print("\n==========================================", flush=True)
    print("ALL FASTAPI & SQLITE ENDPOINT TESTS PASSED 100%!", flush=True)
    print("==========================================", flush=True)

if __name__ == "__main__":
    test_endpoints()
