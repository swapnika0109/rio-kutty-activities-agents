"""
Resilience utilities: Circuit Breaker, Retry with Backoff, Rate Limiter.

Usage:
    from src.utils.resilience import circuit_breaker, retry_with_backoff, RateLimiter

    @circuit_breaker(failure_threshold=5, recovery_timeout=60)
    @retry_with_backoff(max_retries=3, base_delay=1)
    async def call_api():
        ...
"""

import asyncio
import functools
import random
import time
from enum import Enum
from typing import Callable, TypeVar, Any, Optional
from collections import defaultdict

from .logger import setup_logger

logger = setup_logger(__name__)

T = TypeVar("T")


# =============================================================================
# Circuit Breaker
# =============================================================================

class CircuitState(Enum):
    CLOSED = "closed"       # Normal operation
    OPEN = "open"           # Failing, reject requests
    HALF_OPEN = "half_open" # Testing recovery


class CircuitBreakerError(Exception):
    """Raised when circuit breaker is open."""
    pass


class CircuitBreaker:
    """
    Circuit breaker implementation.
    
    States:
        CLOSED: Normal operation, requests pass through
        OPEN: Service failing, requests fail immediately
        HALF_OPEN: Testing if service recovered
    
    Args:
        failure_threshold: Number of failures before opening circuit
        recovery_timeout: Seconds to wait before testing recovery
        half_open_max_calls: Max calls to allow in half-open state
    """
    
    # Shared state across instances (per service name)
    _instances: dict[str, "CircuitBreaker"] = {}
    
    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        half_open_max_calls: int = 3
    ):
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls
        
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time: Optional[float] = None
        self._half_open_calls = 0
    
    @classmethod
    def get_or_create(cls, name: str, **kwargs) -> "CircuitBreaker":
        """Get existing circuit breaker or create new one."""
        if name not in cls._instances:
            cls._instances[name] = cls(name, **kwargs)
        return cls._instances[name]
    
    @property
    def state(self) -> CircuitState:
        """Get current state, checking for recovery timeout."""
        if self._state == CircuitState.OPEN:
            if self._last_failure_time and \
               time.time() - self._last_failure_time > self.recovery_timeout:
                self._state = CircuitState.HALF_OPEN
                self._half_open_calls = 0
                logger.info(f"Circuit '{self.name}' transitioned to HALF_OPEN")
        return self._state
    
    def can_execute(self) -> bool:
        """Check if request can proceed."""
        state = self.state
        
        if state == CircuitState.CLOSED:
            return True
        elif state == CircuitState.OPEN:
            return False
        else:  # HALF_OPEN
            return self._half_open_calls < self.half_open_max_calls
    
    def record_success(self) -> None:
        """Record successful call."""
        if self._state == CircuitState.HALF_OPEN:
            self._failure_count = 0
            self._state = CircuitState.CLOSED
            logger.info(f"Circuit '{self.name}' CLOSED after successful call")
        elif self._state == CircuitState.CLOSED:
            # Reset failure count on success
            self._failure_count = 0
    
    def record_failure(self) -> None:
        """Record failed call."""
        self._failure_count += 1
        self._last_failure_time = time.time()
        
        if self._state == CircuitState.HALF_OPEN:
            self._state = CircuitState.OPEN
            logger.warning(f"Circuit '{self.name}' OPEN (failed in half-open)")
        elif self._failure_count >= self.failure_threshold:
            self._state = CircuitState.OPEN
            logger.warning(f"Circuit '{self.name}' OPEN (threshold reached: {self._failure_count})")


def circuit_breaker(
    name: Optional[str] = None,
    failure_threshold: int = 5,
    recovery_timeout: float = 60.0
) -> Callable:
    """
    Decorator to add circuit breaker to a function.
    
    Args:
        name: Circuit breaker name (defaults to function name)
        failure_threshold: Failures before circuit opens
        recovery_timeout: Seconds before attempting recovery
    
    Example:
        @circuit_breaker(failure_threshold=5, recovery_timeout=60)
        async def call_external_api():
            ...
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        cb_name = name or func.__name__
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs) -> T:
            cb = CircuitBreaker.get_or_create(
                cb_name,
                failure_threshold=failure_threshold,
                recovery_timeout=recovery_timeout
            )
            
            if not cb.can_execute():
                raise CircuitBreakerError(
                    f"Circuit '{cb_name}' is OPEN. Wait {recovery_timeout}s."
                )
            
            if cb.state == CircuitState.HALF_OPEN:
                cb._half_open_calls += 1
            
            try:
                result = await func(*args, **kwargs)
                cb.record_success()
                return result
            except Exception as e:
                cb.record_failure()
                raise
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs) -> T:
            cb = CircuitBreaker.get_or_create(
                cb_name,
                failure_threshold=failure_threshold,
                recovery_timeout=recovery_timeout
            )
            
            if not cb.can_execute():
                raise CircuitBreakerError(
                    f"Circuit '{cb_name}' is OPEN. Wait {recovery_timeout}s."
                )
            
            if cb.state == CircuitState.HALF_OPEN:
                cb._half_open_calls += 1
            
            try:
                result = func(*args, **kwargs)
                cb.record_success()
                return result
            except Exception as e:
                cb.record_failure()
                raise
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    
    return decorator


# =============================================================================
# Retry with Exponential Backoff
# =============================================================================

def retry_with_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
    retryable_exceptions: tuple = (Exception,)
) -> Callable:
    """
    Decorator for retry with exponential backoff.
    
    Args:
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay in seconds
        max_delay: Maximum delay cap
        exponential_base: Multiplier for each retry (default 2x)
        jitter: Add randomness to prevent thundering herd
        retryable_exceptions: Tuple of exceptions to retry on
    
    Example:
        @retry_with_backoff(max_retries=5, base_delay=1)
        async def flaky_api_call():
            ...
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs) -> T:
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except retryable_exceptions as e:
                    last_exception = e
                    
                    if attempt == max_retries:
                        logger.error(f"{func.__name__} failed after {max_retries + 1} attempts")
                        raise
                    
                    # Calculate delay with exponential backoff
                    delay = min(base_delay * (exponential_base ** attempt), max_delay)
                    
                    # Add jitter (±50%)
                    if jitter:
                        delay = delay * random.uniform(0.5, 1.5)
                    
                    logger.warning(
                        f"{func.__name__} attempt {attempt + 1} failed: {e}. "
                        f"Retrying in {delay:.2f}s..."
                    )
                    await asyncio.sleep(delay)
            
            raise last_exception
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs) -> T:
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except retryable_exceptions as e:
                    last_exception = e
                    
                    if attempt == max_retries:
                        logger.error(f"{func.__name__} failed after {max_retries + 1} attempts")
                        raise
                    
                    delay = min(base_delay * (exponential_base ** attempt), max_delay)
                    if jitter:
                        delay = delay * random.uniform(0.5, 1.5)
                    
                    logger.warning(
                        f"{func.__name__} attempt {attempt + 1} failed: {e}. "
                        f"Retrying in {delay:.2f}s..."
                    )
                    time.sleep(delay)
            
            raise last_exception
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    
    return decorator


# =============================================================================
# Rate Limiter (Token Bucket)
# =============================================================================

class RateLimiter:
    """
    Token bucket rate limiter.
    
    Args:
        rate: Tokens added per second
        capacity: Maximum tokens (allows burst)
    
    Example:
        limiter = RateLimiter(rate=10, capacity=20)  # 10/sec, burst of 20
        
        async def call_api():
            await limiter.acquire()  # Blocks if limit reached
            ...
    """
    
    def __init__(self, rate: float, capacity: float):
        self.rate = rate          # Tokens per second
        self.capacity = capacity  # Max tokens
        self.tokens = capacity    # Current tokens
        self.last_refill = time.time()
        self._lock = asyncio.Lock()
    
    def _refill(self) -> None:
        """Refill tokens based on elapsed time."""
        now = time.time()
        elapsed = now - self.last_refill
        self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
        self.last_refill = now
    
    async def acquire(self, tokens: int = 1) -> bool:
        """
        Acquire tokens. Blocks until tokens available.
        
        Args:
            tokens: Number of tokens to acquire
        
        Returns:
            True when tokens acquired
        """
        async with self._lock:
            while True:
                self._refill()
                
                if self.tokens >= tokens:
                    self.tokens -= tokens
                    return True
                
                # Calculate wait time for tokens to refill
                tokens_needed = tokens - self.tokens
                wait_time = tokens_needed / self.rate
                
                await asyncio.sleep(wait_time)
    
    def try_acquire(self, tokens: int = 1) -> bool:
        """
        Try to acquire tokens without blocking.
        
        Returns:
            True if tokens acquired, False otherwise
        """
        self._refill()
        
        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        return False


class RateLimitExceeded(Exception):
    """Raised when rate limit is exceeded and not blocking."""
    pass


# =============================================================================
# Combined Decorator (Circuit Breaker + Retry + Rate Limit)
# =============================================================================

def resilient(
    circuit_name: Optional[str] = None,
    failure_threshold: int = 5,
    recovery_timeout: float = 60.0,
    max_retries: int = 3,
    base_delay: float = 1.0
) -> Callable:
    """
    Combined decorator applying circuit breaker and retry with backoff.
    
    Example:
        @resilient(circuit_name="gemini", failure_threshold=5, max_retries=3)
        async def call_gemini(prompt):
            ...
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        # Apply retry first (inner), then circuit breaker (outer)
        wrapped = retry_with_backoff(
            max_retries=max_retries,
            base_delay=base_delay
        )(func)
        
        wrapped = circuit_breaker(
            name=circuit_name,
            failure_threshold=failure_threshold,
            recovery_timeout=recovery_timeout
        )(wrapped)
        
        return wrapped
    
    return decorator
