from langgraph.types import RetryPolicy
import httpx

from core.config import MAX_ATTEMPTS
from core.circuit_breaker import CircuitBreakerError


class SmartRetryPolicy:
    """智能重试策略"""
    
    @staticmethod
    def create_policy(node_name: str) -> RetryPolicy:
        # 根据节点类型设置不同的重试策略
        if "llm" in node_name:
            return RetryPolicy(
                max_attempts=MAX_ATTEMPTS,
                backoff_factor=2.0,
                max_interval=30.0,
                retry_on=lambda e: SmartRetryPolicy.should_retry_llm(e)
            )
        elif "api" in node_name:
            return RetryPolicy(
                max_attempts=MAX_ATTEMPTS,
                backoff_factor=1.5,
                max_interval=60.0,
                retry_on=lambda e: SmartRetryPolicy.should_retry_api(e)
            )
        else:
            # 默认策略
            return RetryPolicy(max_attempts=MAX_ATTEMPTS)
    
    @staticmethod
    def should_retry_llm(error: Exception) -> bool:
        """LLM调用是否需要重试"""
        # 熔断器断路时不重试，直接走降级
        if isinstance(error, CircuitBreakerError):
            return False

        # 限流错误必须重试
        # 429：太多请求。（网关限流，你发的请求太多了）
        # 502：网关错误。（服务器内部错误，可能是临时的）
        # 503：服务不可用。（服务器过载或维护，暂时无法处理请求）
        # 504：网关超时。（服务器响应超时，可能是临时的）
        if isinstance(error, httpx.HTTPStatusError):
            return error.response.status_code in [429, 502, 503, 504]
        
        # 网络错误重试
        if isinstance(error, (httpx.ConnectError, httpx.TimeoutException)):
            return True
        
        # 参数错误不重试
        if "invalid" in str(error).lower():
            return False
        
        return False
    
    @staticmethod  
    def should_retry_api(error: Exception) -> bool:
        """API调用是否需要重试"""
        # 熔断器断路时不重试，直接走降级
        if isinstance(error, CircuitBreakerError):
            return False

        if isinstance(error, httpx.HTTPStatusError):
            # 5xx都重试，429限流也重试
            return error.response.status_code >= 500 or error.response.status_code == 429
        return isinstance(error, (httpx.ConnectError, httpx.TimeoutException))
