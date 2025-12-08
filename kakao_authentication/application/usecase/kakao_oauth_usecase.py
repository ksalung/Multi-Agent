from config.env import KAKAO_CLIENT_ID, KAKAO_REDIRECT_URI
from kakao_authentication.domain.kakao_user import KakaoUser
from kakao_authentication.domain.port.kakao_oauth_port import KakaoOAuthPort
from kakao_authentication.domain.value_objects.kakao_authorization_url import KakaoAuthorizationUrl
from social_oauth.adapter.input.web.response.access_token import AccessToken


class KakaoOAuthUseCase:

    def __init__(self, kakao_oauth_port: KakaoOAuthPort):
        self.kakao_oauth_port = kakao_oauth_port

    def get_authorization_url(self) -> str:
        auth_url = KakaoAuthorizationUrl(
            client_id=KAKAO_CLIENT_ID,
            redirect_uri=KAKAO_REDIRECT_URI
        )

        return str(auth_url)

    def get_kakao_user(self, code: str) -> dict:
        access_token: AccessToken = self.kakao_oauth_port.get_access_token(code)
        kakao_user: KakaoUser = self.kakao_oauth_port.get_user_info(access_token)

        return {
            "user": kakao_user,
            "access_token": access_token.value
        }
