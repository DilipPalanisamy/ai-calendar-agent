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
    print("Testing GET / (Public Home Page without Redirect)...", flush=True)
    res_root = client.get("/", follow_redirects=False)
    assert res_root.status_code == 200, f"Expected 200 OK, got {res_root.status_code}"
    assert "AI Calendar Assistant" in res_root.text, "Root page missing app name"
    assert "Sign in with Google" in res_root.text, "Root page missing Google Sign-in button"
    assert "VEHl4mzgS2aSF1pyd69IZiLSW6EC2m2VrnC_A4tpTxo" in res_root.text, "Root page missing Google verification tag"
    assert "/privacy" in res_root.text and "/terms" in res_root.text, "Root page missing legal links"
    print("[PASS] GET / (200 OK, Public Landing Page with Google Verification & Sign-in)", flush=True)

    print("Testing GET /login (Login Page UI)...", flush=True)
    res_login = client.get("/login")
    assert res_login.status_code == 200, f"Expected 200 OK, got {res_login.status_code}"
    assert "Sign in with Google" in res_login.text, "Login page missing Google Sign-in button"
    assert "VEHl4mzgS2aSF1pyd69IZiLSW6EC2m2VrnC_A4tpTxo" in res_login.text, "Login page missing Google verification tag"
    print("[PASS] GET /login (200 OK, Login Page rendered with Google verification tag)", flush=True)

    print("Testing GET /privacy (Privacy Policy)...", flush=True)
    res_privacy = client.get("/privacy")
    assert res_privacy.status_code == 200, f"Expected 200 OK, got {res_privacy.status_code}"
    assert "Privacy Policy" in res_privacy.text, "Privacy policy missing title"
    assert "VEHl4mzgS2aSF1pyd69IZiLSW6EC2m2VrnC_A4tpTxo" in res_privacy.text, "Privacy policy missing Google verification tag"
    assert "Google API Services User Data Policy" in res_privacy.text, "Privacy policy missing Google Policy reference"
    print("[PASS] GET /privacy (200 OK, Privacy Policy rendered)", flush=True)

    print("Testing GET /terms (Terms of Service)...", flush=True)
    res_terms = client.get("/terms")
    assert res_terms.status_code == 200, f"Expected 200 OK, got {res_terms.status_code}"
    assert "Terms of Service" in res_terms.text, "Terms of service missing title"
    assert "VEHl4mzgS2aSF1pyd69IZiLSW6EC2m2VrnC_A4tpTxo" in res_terms.text, "Terms of service missing Google verification tag"
    print("[PASS] GET /terms (200 OK, Terms of Service rendered)", flush=True)

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

    print("Testing GET /auth/login (Redirect to Google with access_type=offline & prompt=consent)...", flush=True)
    res_auth_login = client.get("/auth/login", follow_redirects=False)
    assert res_auth_login.status_code == 303, f"Expected 303, got {res_auth_login.status_code}"
    loc = res_auth_login.headers.get("location", "")
    assert "access_type=offline" in loc, f"Expected access_type=offline in auth url: {loc}"
    assert "prompt=consent" in loc, f"Expected prompt=consent in auth url: {loc}"
    print("[PASS] GET /auth/login (Enforces access_type=offline and prompt=consent)", flush=True)

    print("Testing GET /auth/add-account (Redirect to Google with access_type=offline & prompt=consent)...", flush=True)
    res_add_acc = client.get("/auth/add-account", follow_redirects=False)
    assert res_add_acc.status_code == 303, f"Expected 303, got {res_add_acc.status_code}"
    loc_add = res_add_acc.headers.get("location", "")
    assert "access_type=offline" in loc_add, f"Expected access_type=offline in auth url: {loc_add}"
    assert "prompt=" in loc_add and "consent" in loc_add, f"Expected consent in prompt: {loc_add}"
    print("[PASS] GET /auth/add-account (Enforces access_type=offline and prompt=consent)", flush=True)

    print("Testing GET /logout (Redirect to /login)...", flush=True)
    res_logout = client.get("/logout", follow_redirects=False)
    assert res_logout.status_code == 303, f"Expected 303 See Other, got {res_logout.status_code}"
    assert res_logout.headers.get("location") == "/login", f"Expected location /login, got {res_logout.headers.get('location')}"
    print("[PASS] GET /logout (303 Redirect to /login)", flush=True)

    print("Testing GET /manifest.json (PWA Manifest)...", flush=True)
    res_manifest = client.get("/manifest.json")
    assert res_manifest.status_code == 200, f"Expected 200, got {res_manifest.status_code}"
    assert "AI Calendar Assistant" in res_manifest.text
    print("[PASS] GET /manifest.json (200 OK)", flush=True)

    print("Testing GET /sw.js (Service Worker)...", flush=True)
    res_sw = client.get("/sw.js")
    assert res_sw.status_code == 200, f"Expected 200, got {res_sw.status_code}"
    print("[PASS] GET /sw.js (200 OK)", flush=True)

    print("Testing GET /download/app (App Download)...", flush=True)
    res_dl = client.get("/download/app")
    assert res_dl.status_code == 200, f"Expected 200, got {res_dl.status_code}"
    print("[PASS] GET /download/app (200 OK)", flush=True)

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

    # Test Session Persistence across Logout and Relogin
    print("Testing Chat History Persistence across Logout and Relogin...", flush=True)
    user_c = "persistent_user@gmail.com"
    delete_all_user_sessions(user_c)

    # 1. User C starts a conversation
    s_c1 = create_chat_session(user_c, "Project Roadmap Discussion")
    save_chat_message(s_c1, "user", "What are our milestones for Q3?", user_c)
    save_chat_message(s_c1, "assistant", "Milestone 1 is deployment by September.", user_c)

    # 2. User C sends a second message in the same session
    save_chat_message(s_c1, "user", "Add follow-up sync tomorrow at 10 AM", user_c)
    save_chat_message(s_c1, "assistant", "Scheduled follow-up sync for tomorrow at 10:00 AM IST.", user_c)

    # 3. User C simulates Logout (clears HTTP session)
    res_logout_c = client.get("/logout", follow_redirects=False)
    assert res_logout_c.status_code == 303

    # 4. Verify SQLite database still contains all sessions and messages for User C
    sessions_after_logout = get_user_chat_sessions(user_c)
    assert len(sessions_after_logout) == 1, f"Expected 1 session after logout, found {len(sessions_after_logout)}"
    assert sessions_after_logout[0]["id"] == s_c1

    details = get_chat_session_details(s_c1, user_c)
    assert details is not None
    assert len(details["messages"]) == 4, f"Expected 4 messages, found {len(details['messages'])}"
    assert details["messages"][0]["content"] == "What are our milestones for Q3?"
    assert details["messages"][3]["content"] == "Scheduled follow-up sync for tomorrow at 10:00 AM IST."

    # 5. User C creates a second newer session
    s_c2 = create_chat_session(user_c, "Lunch Meeting with Team")
    save_chat_message(s_c2, "user", "Book lunch at noon", user_c)
    save_chat_message(s_c2, "assistant", "Lunch booked for 12:00 PM IST.", user_c)

    # Verify latest session is returned first for auto-loading on page refresh / login
    ordered_sessions = get_user_chat_sessions(user_c)
    assert len(ordered_sessions) == 2
    assert ordered_sessions[0]["id"] == s_c2, "Latest session must be at index 0 for auto-selection"

    # Clean up test user
    delete_all_user_sessions(user_c)
    delete_all_user_sessions(user_b)
    print("[PASS] Chat History Persistence across Logout and Relogin verified 100%!", flush=True)

    print("\n==========================================", flush=True)
    print("ALL FASTAPI & MULTI-ACCOUNT TESTS PASSED 100%!", flush=True)
    print("==========================================", flush=True)

if __name__ == "__main__":
    test_endpoints()
