"""
필드 검증자 테스트.
"""

from pathlib import Path

import pytest

from src.common.validation.field_validators import (
    validate_api_key,
    validate_dimension,
    validate_email,
    validate_file_path,
    validate_host,
    validate_port,
    validate_positive_integer,
    validate_positive_number,
    validate_temperature,
    validate_timeout,
    validate_url,
)


class TestValidatePositiveInteger:
    """양의 정수 검증 테스트."""

    def test_valid_positive_integer(self):
        """유효한 양의 정수 테스트."""
        result = validate_positive_integer(10, "테스트 필드")
        assert result == 10

    def test_valid_positive_integer_with_max(self):
        """최대값이 있는 양의 정수 테스트."""
        result = validate_positive_integer(5, "테스트 필드", 10)
        assert result == 5

    def test_invalid_zero(self):
        """0 값 테스트."""
        with pytest.raises(ValueError, match="테스트 필드은\\(는\\) 양수여야 합니다"):
            validate_positive_integer(0, "테스트 필드")

    def test_invalid_negative(self):
        """음수 값 테스트."""
        with pytest.raises(ValueError, match="테스트 필드은\\(는\\) 양수여야 합니다"):
            validate_positive_integer(-5, "테스트 필드")

    def test_invalid_type(self):
        """잘못된 타입 테스트."""
        with pytest.raises(ValueError, match="테스트 필드은\\(는\\) 정수여야 합니다"):
            validate_positive_integer("10", "테스트 필드")

    def test_exceeds_max_value(self):
        """최대값 초과 테스트."""
        with pytest.raises(ValueError, match="테스트 필드은\\(는\\) 10 이하여야 합니다"):
            validate_positive_integer(15, "테스트 필드", 10)


class TestValidatePositiveNumber:
    """양의 실수 검증 테스트."""

    def test_valid_positive_float(self):
        """유효한 양의 실수 테스트."""
        result = validate_positive_number(10.5, "테스트 필드")
        assert result == 10.5

    def test_valid_positive_int_as_float(self):
        """정수를 실수로 변환 테스트."""
        result = validate_positive_number(10, "테스트 필드")
        assert result == 10.0
        assert isinstance(result, float)

    def test_invalid_zero_float(self):
        """0.0 값 테스트."""
        with pytest.raises(ValueError, match="테스트 필드은\\(는\\) 0보다 커야 합니다"):
            validate_positive_number(0.0, "테스트 필드")

    def test_invalid_negative_float(self):
        """음의 실수 테스트."""
        with pytest.raises(ValueError, match="테스트 필드은\\(는\\) 0보다 커야 합니다"):
            validate_positive_number(-5.5, "테스트 필드")


class TestValidatePort:
    """포트 번호 검증 테스트."""

    def test_valid_port(self):
        """유효한 포트 번호 테스트."""
        result = validate_port(8080)
        assert result == 8080

    def test_valid_port_boundary(self):
        """경계값 포트 번호 테스트."""
        assert validate_port(1) == 1
        assert validate_port(65535) == 65535

    def test_invalid_port_zero(self):
        """0 포트 테스트."""
        with pytest.raises(ValueError, match="포트 번호은\\(는\\) 양수여야 합니다"):
            validate_port(0)

    def test_invalid_port_too_high(self):
        """포트 번호 초과 테스트."""
        with pytest.raises(ValueError, match="포트 번호은\\(는\\) 65535 이하여야 합니다"):
            validate_port(65536)


class TestValidateTimeout:
    """타임아웃 검증 테스트."""

    def test_valid_timeout(self):
        """유효한 타임아웃 테스트."""
        result = validate_timeout(30.0)
        assert result == 30.0

    def test_valid_timeout_with_custom_max(self):
        """커스텀 최대값 타임아웃 테스트."""
        result = validate_timeout(10.0, 20.0)
        assert result == 10.0

    def test_invalid_timeout_zero(self):
        """0 타임아웃 테스트."""
        with pytest.raises(ValueError, match="타임아웃은\\(는\\) 0보다 커야 합니다"):
            validate_timeout(0.0)

    def test_invalid_timeout_exceeds_max(self):
        """최대 타임아웃 초과 테스트."""
        with pytest.raises(ValueError, match="타임아웃은\\(는\\) 3600.0 이하여야 합니다"):
            validate_timeout(4000.0)


class TestValidateDimension:
    """차원 수 검증 테스트."""

    def test_valid_dimension(self):
        """유효한 차원 테스트."""
        result = validate_dimension(384)
        assert result == 384

    def test_valid_dimension_boundary(self):
        """경계값 차원 테스트."""
        assert validate_dimension(1) == 1
        assert validate_dimension(4096) == 4096

    def test_invalid_dimension_zero(self):
        """0 차원 테스트."""
        with pytest.raises(ValueError, match="차원은\\(는\\) 양수여야 합니다"):
            validate_dimension(0)

    def test_invalid_dimension_too_high(self):
        """차원 수 초과 테스트."""
        with pytest.raises(ValueError, match="차원은\\(는\\) 4096 이하여야 합니다"):
            validate_dimension(5000)


class TestValidateApiKey:
    """API 키 검증 테스트."""

    def test_valid_api_key(self):
        """유효한 API 키 테스트."""
        api_key = "sk-1234567890abcdef"
        result = validate_api_key(api_key, "OpenAI")
        assert result == api_key

    def test_valid_api_key_with_prefix(self):
        """접두사가 있는 API 키 테스트."""
        api_key = "sk-1234567890abcdef"
        result = validate_api_key(api_key, "OpenAI", "sk-")
        assert result == api_key

    def test_none_api_key(self):
        """None API 키 테스트."""
        result = validate_api_key(None, "OpenAI")
        assert result is None

    def test_empty_api_key(self):
        """빈 API 키 테스트."""
        with pytest.raises(ValueError, match="OpenAI API 키는 비어 있지 않은 문자열이어야 합니다"):
            validate_api_key("", "OpenAI")

    def test_wrong_prefix(self):
        """잘못된 접두사 테스트."""
        with pytest.raises(ValueError, match="OpenAI API 키는 'sk-'로 시작해야 합니다"):
            validate_api_key("pk-1234567890abcdef", "OpenAI", "sk-")

    def test_too_short_api_key(self):
        """너무 짧은 API 키 테스트."""
        with pytest.raises(ValueError, match="OpenAI API 키가 너무 짧습니다"):
            validate_api_key("sk-123", "OpenAI")


class TestValidateUrl:
    """URL 검증 테스트."""

    def test_valid_http_url(self):
        """유효한 HTTP URL 테스트."""
        url = "http://example.com"
        result = validate_url(url)
        assert result == url

    def test_valid_https_url(self):
        """유효한 HTTPS URL 테스트."""
        url = "https://api.openai.com/v1/embeddings"
        result = validate_url(url)
        assert result == url

    def test_empty_url(self):
        """빈 URL 테스트."""
        with pytest.raises(ValueError, match="URL은\\(는\\) 비어 있지 않은 문자열이어야 합니다"):
            validate_url("")

    def test_invalid_url_no_scheme(self):
        """스키마 없는 URL 테스트."""
        with pytest.raises(ValueError, match="올바르지 않은 URL 형식입니다"):
            validate_url("example.com")

    def test_invalid_url_no_netloc(self):
        """네트워크 위치 없는 URL 테스트."""
        with pytest.raises(ValueError, match="올바르지 않은 URL 형식입니다"):
            validate_url("http://")


class TestValidateFilePath:
    """파일 경로 검증 테스트."""

    def test_valid_file_path(self):
        """유효한 파일 경로 테스트."""
        path = "data/test.db"
        result = validate_file_path(path)
        assert result == path

    def test_valid_absolute_path(self):
        """유효한 절대 경로 테스트."""
        path = "/tmp/test.db"
        result = validate_file_path(path)
        assert result == path

    def test_empty_file_path(self):
        """빈 파일 경로 테스트."""
        with pytest.raises(ValueError, match="파일 경로는 비어 있지 않은 문자열이어야 합니다"):
            validate_file_path("")

    def test_parent_directory_reference(self):
        """상위 디렉토리 참조 테스트."""
        with pytest.raises(
            ValueError, match="파일 경로에 상위 디렉토리 참조\\(\\.\\.)는 허용되지 않습니다"
        ):
            validate_file_path("../test.db")

    def test_file_must_exist_but_doesnt(self):
        """존재해야 하지만 존재하지 않는 파일 테스트."""
        with pytest.raises(ValueError, match="파일 경로에 지정된 파일이 존재하지 않습니다"):
            validate_file_path("/nonexistent/file.db", must_exist=True)


class TestValidateTemperature:
    """온도 값 검증 테스트."""

    def test_valid_temperature(self):
        """유효한 온도 테스트."""
        result = validate_temperature(0.7)
        assert result == 0.7

    def test_valid_temperature_boundary(self):
        """경계값 온도 테스트."""
        assert validate_temperature(0.0) == 0.0
        assert validate_temperature(2.0) == 2.0

    def test_invalid_temperature_too_low(self):
        """너무 낮은 온도 테스트."""
        with pytest.raises(ValueError, match="온도는 0.0에서 2.0 사이여야 합니다"):
            validate_temperature(-0.1)

    def test_invalid_temperature_too_high(self):
        """너무 높은 온도 테스트."""
        with pytest.raises(ValueError, match="온도는 0.0에서 2.0 사이여야 합니다"):
            validate_temperature(2.1)

    def test_invalid_temperature_type(self):
        """잘못된 온도 타입 테스트."""
        with pytest.raises(ValueError, match="온도는 숫자여야 합니다"):
            validate_temperature("0.7")


class TestValidateEmail:
    """이메일 주소 검증 테스트."""

    def test_valid_email(self):
        """유효한 이메일 테스트."""
        email = "user@example.com"
        result = validate_email(email)
        assert result == email

    def test_valid_complex_email(self):
        """복잡한 이메일 테스트."""
        email = "user.name+tag@example-domain.co.uk"
        result = validate_email(email)
        assert result == email

    def test_empty_email(self):
        """빈 이메일 테스트."""
        with pytest.raises(ValueError, match="이메일 주소는 비어 있지 않은 문자열이어야 합니다"):
            validate_email("")

    def test_invalid_email_no_at(self):
        """@ 없는 이메일 테스트."""
        with pytest.raises(ValueError, match="올바르지 않은 이메일 주소 형식입니다"):
            validate_email("userexample.com")

    def test_invalid_email_no_domain(self):
        """도메인 없는 이메일 테스트."""
        with pytest.raises(ValueError, match="올바르지 않은 이메일 주소 형식입니다"):
            validate_email("user@")


class TestValidateHost:
    """호스트명 검증 테스트."""

    def test_valid_domain_host(self):
        """유효한 도메인 호스트 테스트."""
        host = "example.com"
        result = validate_host(host)
        assert result == host

    def test_valid_ip_host(self):
        """유효한 IP 호스트 테스트."""
        host = "127.0.0.1"
        result = validate_host(host)
        assert result == host

    def test_valid_subdomain_host(self):
        """유효한 서브도메인 호스트 테스트."""
        host = "api.example.com"
        result = validate_host(host)
        assert result == host

    def test_empty_host(self):
        """빈 호스트 테스트."""
        with pytest.raises(ValueError, match="호스트명은 비어 있지 않은 문자열이어야 합니다"):
            validate_host("")

    def test_invalid_host_format(self):
        """잘못된 호스트 형식 테스트."""
        with pytest.raises(ValueError, match="올바르지 않은 호스트명 형식입니다"):
            validate_host("invalid_host!")
