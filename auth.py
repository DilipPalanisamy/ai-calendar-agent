import os
from google_auth_oauthlib.flow import InstalledAppFlow

SCOPES = [
    'https://www.googleapis.com/auth/calendar',
    'https://www.googleapis.com/auth/gmail.readonly'
]

if not os.path.exists('credentials.json'):
    print("❌ ERROR: credentials.json is missing! Please place it in this folder.")
else:
    flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
    # prompt='consent' and access_type='offline' ensure a persistent refresh_token is issued
    creds = flow.run_local_server(port=0, open_browser=False, access_type='offline', prompt='consent')
    
    with open('token.json', 'w') as token:
        token.write(creds.to_json())
    print("✅ Successfully generated token.json!")