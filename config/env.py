import os
from dotenv import load_dotenv

load_dotenv()

KAKAO_CLIENT_ID = os.getenv("KAKAO_CLIENT_ID")
KAKAO_REDIRECT_URI = os.getenv("KAKAO_REDIRECT_URI")

if not KAKAO_CLIENT_ID:
    raise RuntimeError("KAKAO_CLIENT_ID is not set")
if not KAKAO_REDIRECT_URI:
    raise RuntimeError("KAKAO_REDIRECT_URI is not set")
