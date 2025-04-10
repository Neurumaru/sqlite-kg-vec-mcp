#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
기본 MCP 서버 모듈

MCP 프로토콜의 기본 동작을 검증하기 위한 최소 기능의 FastMCP 서버 구현입니다.
간단한 에코 도구를 제공합니다.
"""

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("BasicServer")

# 에코 도구
@mcp.tool()
def echo(message: str) -> str:
    """
    에코 도구: 입력 메시지를 그대로 반환
    
    Args:
        message: 메시지 문자열
        
    Returns:
        입력 메시지 그대로 반환
    """
    return message

# 핑 도구
@mcp.tool()
def ping() -> str:
    """
    핑 도구: "pong" 반환
    
    Returns:
        "pong" 문자열
    """
    return "pong"

# 숫자 더하기 도구
@mcp.tool()
def add_numbers(a: float, b: float) -> float:
    """
    숫자 더하기 도구: 두 숫자의 합을 반환
    
    Args:
        a: 첫 번째 숫자
        b: 두 번째 숫자
        
    Returns:
        두 숫자의 합
    """
    return a + b

if __name__ == "__main__":
    mcp.run()
