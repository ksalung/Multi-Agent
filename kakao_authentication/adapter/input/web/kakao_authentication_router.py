import json
import os
import uuid
from fastapi import APIRouter, Response, Request, Cookie
from fastapi.responses import RedirectResponse

from account.application.usecase.account_usecase import AccountUseCase
from account.infrastructure.repository.account_repository_impl import AccountRepositoryImpl
from config.redis_config import get_redis
from kakao_authentication.application.usecase.kakao_oauth_usecase import KakaoOAuthUseCase
from kakao_authentication.infrastructure.client.kakao_oauth_client import KakaoOAuthClient

kakao_authentication_router = APIRouter()

# Infrastructure
kakao_client = KakaoOAuthClient()
kakao_usecase = KakaoOAuthUseCase(kakao_client)

account_repository = AccountRepositoryImpl()
account_usecase = AccountUseCase(account_repository)

redis_client = get_redis()

CORS_ALLOWED_FRONTEND_URL = os.getenv("CORS_ALLOWED_FRONTEND_URL")


@kakao_authentication_router.get("/login")
async def redirect_to_kakao():
    login_url = kakao_usecase.get_authorization_url()
    print("[DEBUG] Redirecting to Kakao:", login_url)
    return RedirectResponse(login_url)

@kakao_authentication_router.get("/redirection")
async def kakao_redirect(code: str):
    print("[DEBUG] Kakao redirect called")
    print("[DEBUG] code:", code)

    # 카카오 사용자 조회 (VO 포함)
    result = kakao_usecase.get_kakao_user(code)

    kakao_user = result["user"]          # KakaoUser
    access_token = result["access_token"]  # str

    print("[DEBUG] Kakao User ID:", kakao_user.user_id.value)
    print("[DEBUG] Kakao Nickname:", kakao_user.nickname.value)

    # Account 생성 또는 조회
    account = account_usecase.create_or_get_account(
        kakao_user.email.value,
        kakao_user.nickname.value
    )

    # Session ID 생성
    session_id = str(uuid.uuid4())
    print("[DEBUG] Generated session_id:", session_id)

    # Redis 저장
    redis_client.set(
        f"session:{session_id}",
        json.dumps({
            "user_id": account.id,
            "access_token": access_token
        }),
        ex=6 * 60 * 60  # 6시간
    )

    # HTTP Only 쿠키 발급
    response = RedirectResponse(CORS_ALLOWED_FRONTEND_URL)

    response.set_cookie(
        key="session_id",
        value=session_id,
        httponly=True,
        secure=False,
        samesite="lax",
        max_age=6 * 60 * 60
    )

    print("[DEBUG] Cookie issued successfully")
    return response
