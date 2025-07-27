"""
도메인 예외 기본 클래스.
"""


class DomainException(Exception):
    """
    도메인 예외 기본 클래스.
    
    도메인 로직에서 발생하는 모든 예외의 기본 클래스입니다.
    """
    
    def __init__(self, message: str, error_code: str = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
    
    def __str__(self) -> str:
        if self.error_code:
            return f"[{self.error_code}] {self.message}"
        return self.message