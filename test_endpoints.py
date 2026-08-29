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

    print("Testing GET /api/accounts (Unauthenticated Guard)...", flush=True)
    res_acc = client.get("/api/accounts")
    assert res_acc.status_code == 401, f"Expected 401 Unauthorized, got {res_acc.status_code}"
    print("[PASS] GET /api/accounts (401 Unauthorized)", flush=True)

    print("Testing POST /api/accounts/switch (Unauthenticated Guard)...", flush=True)
    res_sw = client.post("/api/accounts/switch", json={"email": "test@gmail.com"})
    assert res_sw.status_code == 404 or res_sw.status_code == 401, f"Expected 404/401, got {res_sw.status_code}"
    print("[PASS] POST /api/accounts/switch (Guarded)", flush=True)

    print("Testing GET /api/history (Unauthenticated Guard)...", flush=True)
    res_hist = client.get("/api/history")
    assert res_hist.status_code == 401, f"Expected 401 Unauthorized, got {res_hist.status_code}"
    print("[PASS] GET /api/history (401 Unauthorized)", flush=True)

    print("Testing DELETE /api/history (Unauthenticated Guard)...", flush=True)
    res_del = client.delete("/api/history")
    assert res_del.status_code == 401, f"Expected 401 Unauthorized, got {res_del.status_code}"
    print("[PASS] DELETE /api/history (401 Unauthorized)", flush=True)

    print("Testing POST /api/calendar/delete (Unauthenticated Guard)...", flush=True)
    res_cal_del = client.post("/api/calendar/delete", json={"event_id": "test_123"})
    assert res_cal_del.status_code == 401, f"Expected 401 Unauthorized, got {res_cal_del.status_code}"
    print("[PASS] POST /api/calendar/delete (401 Unauthorized)", flush=True)

    print("Testing GET /logout (Redirect to /login)...", flush=True)
    res_logout = client.get("/logout", follow_redirects=False)
    assert res_logout.status_code == 303, f"Expected 303 See Other, got {res_logout.status_code}"
    assert res_logout.headers.get("location") == "/login", f"Expected location /login, got {res_logout.headers.get('location')}"
    print("[PASS] GET /logout (303 Redirect to /login)", flush=True)

    # SQLite Database Multi-User Isolation Tests
    print("Testing SQLite Chat History Multi-User Isolation...", flush=True)
    user_a = "user_a@gmail.com"
    user_b = "user_b@gmail.com"
    delete_all_user_sessions(user_a)
    delete_all_user_sessions(user_b)

    # Create sessions for User A and User B
    s_a = create_chat_session(user_a, "User A Private Calendar Plan")
    save_chat_message(s_a, "user", "Schedule sync with team tomorrow", user_a)
    save_chat_message(s_a, "assistant", "Event scheduled for User A.", user_a)

    s_b = create_chat_session(user_b, "User B Flight Details")
    save_chat_message(s_b, "user", "Check flights on Friday", user_b)
    save_chat_message(s_b, "assistant", "Flight check completed for User B.", user_b)

    # Verify User A only sees User A's session
    sessions_a = get_user_chat_sessions(user_a)
    assert len(sessions_a) == 1
    assert sessions_a[0]["id"] == s_a
    assert sessions_a[0]["title"] == "User A Private Calendar Plan"

    # Verify User B cannot access User A's session details
    assert get_chat_session_details(s_a, user_b) is None, "Security violation: User B accessed User A's session!"

    # Verify User B only sees User B's session
    sessions_b = get_user_chat_sessions(user_b)
    assert len(sessions_b) == 1
    assert sessions_b[0]["id"] == s_b
    assert sessions_b[0]["title"] == "User B Flight Details"

    # Deleting User A's session must NOT affect User B's history
    delete_chat_session(s_a, user_a)
    assert len(get_user_chat_sessions(user_a)) == 0
    assert len(get_user_chat_sessions(user_b)) == 1, "Isolation violation: Deleting User A cleared User B's data!"

    delete_all_user_sessions(user_b)
    assert len(get_user_chat_sessions(user_b)) == 0
    print("[PASS] Multi-User Chat History Isolation verified 100%!", flush=True)

    print("\n==========================================", flush=True)
    print("ALL FASTAPI & MULTI-ACCOUNT TESTS PASSED 100%!", flush=True)
    print("==========================================", flush=True)

if __name__ == "__main__":
    test_endpoints()
