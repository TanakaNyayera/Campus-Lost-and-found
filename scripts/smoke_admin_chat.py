import os, sys
WEB_APP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src', 'web_app')
sys.path.insert(0, os.path.abspath(WEB_APP_DIR))

from app import app

with app.test_client() as c:
    endpoints = [
        ('GET', '/admin/chat/rooms', None),
        ('GET', '/admin/chat/messages?room_id=1', None),
        ('POST', '/admin/chat/clear', {'room_id': 1}),
        ('POST', '/admin/chat/delete', {'room_id': 1}),
    ]

    for method, url, payload in endpoints:
        if method == 'GET':
            res = c.get(url)
        else:
            res = c.post(url, json=payload)
        print(f"{method} {url} -> {res.status_code} {res.get_data(as_text=True)[:200]}")
