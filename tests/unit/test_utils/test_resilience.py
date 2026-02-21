"""
Unit tests for resilience utilities (circuit breaker, retry, rate limiter).
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock

from src.utils.resilience import (
    CircuitBreaker,
    CircuitState,
    CircuitBreakerError,
    circuit_breaker,
    retry_with_backoff,
    RateLimiter,
    resilient,
)


class TestCircuitBreaker:
    """Tests for CircuitBreaker class."""
    
    def test_initial_state_is_closed(self):
        cb = CircuitBreaker("test", failure_threshold=3)
        assert cb.state == CircuitState.CLOSED
    
    def test_opens_after_threshold_failures(self):
        cb = CircuitBreaker("test", failure_threshold=3)
        
        for _ in range(3):
            cb.record_failure()
        
        assert cb.state == CircuitState.OPEN
    
    def test_stays_closed_below_threshold(self):
        cb = CircuitBreaker("test", failure_threshold=3)
        
        cb.record_failure()
        cb.record_failure()
        
        assert cb.state == CircuitState.CLOSED
    
    def test_success_resets_failure_count(self):
        cb = CircuitBreaker("test", failure_threshold=3)
        
        cb.record_failure()
        cb.record_failure()
        cb.record_success()
        cb.record_failure()
        
        assert cb.state == CircuitState.CLOSED
        assert cb._failure_count == 1
    
    def test_cannot_execute_when_open(self):
        cb = CircuitBreaker("test", failure_threshold=1, recovery_timeout=60)
        cb.record_failure()
        
        assert cb.state == CircuitState.OPEN
        assert cb.can_execute() is False
    
    def test_get_or_create_returns_same_instance(self):
        cb1 = CircuitBreaker.get_or_create("shared", failure_threshold=5)
        cb2 = CircuitBreaker.get_or_create("shared")
        
        assert cb1 is cb2


class TestCircuitBreakerDecorator:
    """Tests for @circuit_breaker decorator."""
    
    @pytest.mark.asyncio
    async def test_decorator_allows_successful_calls(self):
        @circuit_breaker(name="test_success", failure_threshold=3)
        async def successful_func():
            return "success"
        
        result = await successful_func()
        assert result == "success"
    
    @pytest.mark.asyncio
    async def test_decorator_opens_circuit_on_failures(self):
        call_count = 0
        
        @circuit_breaker(name="test_failure", failure_threshold=2, recovery_timeout=60)
        async def failing_func():
            nonlocal call_count
            call_count += 1
            raise ValueError("fail")
        
        # First two calls should fail normally
        with pytest.raises(ValueError):
            await failing_func()
        with pytest.raises(ValueError):
            await failing_func()
        
        # Circuit should now be open
        with pytest.raises(CircuitBreakerError):
            await failing_func()
        
        # Should have only called the actual function twice
        assert call_count == 2


class TestRetryWithBackoff:
    """Tests for @retry_with_backoff decorator."""
    
    @pytest.mark.asyncio
    async def test_succeeds_without_retry(self):
        call_count = 0
        
        @retry_with_backoff(max_retries=3)
        async def successful_func():
            nonlocal call_count
            call_count += 1
            return "success"
        
        result = await successful_func()
        assert result == "success"
        assert call_count == 1
    
    @pytest.mark.asyncio
    async def test_retries_on_failure(self):
        call_count = 0
        
        @retry_with_backoff(max_retries=3, base_delay=0.01)
        async def flaky_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("transient error")
            return "success"
        
        result = await flaky_func()
        assert result == "success"
        assert call_count == 3
    
    @pytest.mark.asyncio
    async def test_raises_after_max_retries(self):
        call_count = 0
        
        @retry_with_backoff(max_retries=2, base_delay=0.01)
        async def always_fails():
            nonlocal call_count
            call_count += 1
            raise ValueError("permanent error")
        
        with pytest.raises(ValueError, match="permanent error"):
            await always_fails()
        
        # Should have tried 3 times (initial + 2 retries)
        assert call_count == 3


class TestRateLimiter:
    """Tests for RateLimiter class."""
    
    @pytest.mark.asyncio
    async def test_allows_requests_under_limit(self):
        limiter = RateLimiter(rate=10, capacity=10)
        
        for _ in range(5):
            result = await limiter.acquire()
            assert result is True
    
    def test_try_acquire_fails_when_empty(self):
        limiter = RateLimiter(rate=1, capacity=2)
        
        assert limiter.try_acquire() is True
        assert limiter.try_acquire() is True
        assert limiter.try_acquire() is False
    
    @pytest.mark.asyncio
    async def test_refills_over_time(self):
        limiter = RateLimiter(rate=100, capacity=1)  # 100 tokens/sec
        
        assert limiter.try_acquire() is True
        assert limiter.try_acquire() is False
        
        await asyncio.sleep(0.02)  # Wait for refill
        
        assert limiter.try_acquire() is True


class TestResilientDecorator:
    """Tests for combined @resilient decorator."""
    
    @pytest.mark.asyncio
    async def test_combines_circuit_breaker_and_retry(self):
        call_count = 0
        
        @resilient(
            circuit_name="test_resilient",
            failure_threshold=5,
            max_retries=2,
            base_delay=0.01
        )
        async def resilient_func():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ValueError("transient")
            return "success"
        
        result = await resilient_func()
        assert result == "success"
        assert call_count == 2
