"""
Shared pytest fixtures for Rio Kutty Activities Engine tests.

Provides mocked services for:
- AI Service (Gemini + FLUX)
- Firestore Service
- Storage Bucket Service
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import json


# =============================================================================
# AI Service Mocks
# =============================================================================

@pytest.fixture
def mock_ai_service():
    """Mock AIService that returns predictable responses."""
    with patch("src.services.ai_service.AIService") as MockAIService:
        instance = MockAIService.return_value
        
        # Mock generate_content to return valid JSON
        async def mock_generate_content(prompt):
            if "MCQ" in prompt or "multiple-choice" in prompt.lower():
                return json.dumps([
                    {"question": "Test question?", "options": ["A", "B", "C"], "correct": "A"}
                ])
            elif "art" in prompt.lower() or "craft" in prompt.lower():
                return json.dumps({
                    "title": "Test Art Activity",
                    "age_appropriateness": "Suitable for testing",
                    "materials": ["paper", "crayons"],
                    "steps": ["Step 1", "Step 2", "Step 3"],
                    "image_generation_prompt": "A colorful craft"
                })
            elif "moral" in prompt.lower():
                return json.dumps([{
                    "title": "Test Moral Activity",
                    "age_appropriateness": "Suitable for testing",
                    "What it Teaches": "Testing values",
                    "materials": ["paper"],
                    "Instructions": ["Step 1", "Step 2"],
                    "image_generation_prompt": "A moral craft"
                }])
            elif "science" in prompt.lower():
                return json.dumps([{
                    "title": "Test Science Experiment",
                    "age_appropriateness": "Suitable for testing",
                    "What it Teaches": "Testing science",
                    "materials": ["water", "cup"],
                    "Instructions": ["Step 1", "Step 2"],
                    "image_generation_prompt": "A science experiment"
                }])
            return json.dumps({"response": "test"})
        
        instance.generate_content = AsyncMock(side_effect=mock_generate_content)
        
        # Mock generate_image to return fake image bytes
        instance.generate_image = AsyncMock(return_value=b"fake_image_bytes")
        
        yield instance


@pytest.fixture
def mock_ai_service_failure():
    """Mock AIService that simulates failures."""
    with patch("src.services.ai_service.AIService") as MockAIService:
        instance = MockAIService.return_value
        instance.generate_content = AsyncMock(side_effect=Exception("API Error"))
        instance.generate_image = AsyncMock(side_effect=Exception("Image API Error"))
        yield instance


# =============================================================================
# Firestore Service Mocks
# =============================================================================

@pytest.fixture
def mock_firestore_service():
    """Mock FirestoreService for database operations."""
    with patch("src.services.database.firestore_service.FirestoreService") as MockFirestore:
        instance = MockFirestore.return_value
        
        # Mock get_story
        instance.get_story = AsyncMock(return_value={
            "story_id": "test_story_123",
            "story_text": "Once upon a time, there was a brave little rabbit.",
            "language": "en"
        })
        
        # Mock save_activity
        instance.save_activity = AsyncMock(return_value=None)
        
        # Mock check_if_activity_exists
        instance.check_if_activity_exists = AsyncMock(return_value=False)
        
        yield instance


@pytest.fixture
def mock_firestore_with_existing_activities():
    """Mock FirestoreService where some activities already exist."""
    with patch("src.services.database.firestore_service.FirestoreService") as MockFirestore:
        instance = MockFirestore.return_value
        
        instance.get_story = AsyncMock(return_value={
            "story_id": "test_story_123",
            "story_text": "Once upon a time...",
            "language": "en"
        })
        
        # MCQ already exists
        async def check_exists(story_id, activity_type):
            return activity_type == "mcq"
        
        instance.check_if_activity_exists = AsyncMock(side_effect=check_exists)
        instance.save_activity = AsyncMock(return_value=None)
        
        yield instance


# =============================================================================
# Storage Bucket Service Mocks
# =============================================================================

@pytest.fixture
def mock_storage_bucket():
    """Mock StorageBucketService for image uploads."""
    with patch("src.services.database.storage_bucket.StorageBucketService") as MockStorage:
        instance = MockStorage.return_value
        instance.upload_file = AsyncMock(return_value="images/test_uuid.png")
        instance.delete_file = AsyncMock(return_value=None)
        yield instance


# =============================================================================
# Sample Data Fixtures
# =============================================================================

@pytest.fixture
def sample_story():
    """Sample story data for testing."""
    return {
        "story_id": "test_story_123",
        "story_text": """
        Once upon a time, in a lush green forest, there lived a brave little rabbit 
        named Ruby. Ruby loved to explore and make new friends. One day, Ruby found 
        a lost baby bird and helped it find its way home. The bird's mother was so 
        grateful that she taught Ruby how to whistle the most beautiful songs.
        """,
        "language": "en"
    }


@pytest.fixture
def sample_state():
    """Sample workflow state for testing."""
    return {
        "activities": {},
        "images": {},
        "completed": [],
        "errors": {},
        "retry_count": {}
    }


@pytest.fixture
def sample_config():
    """Sample workflow config for testing."""
    return {
        "configurable": {
            "thread_id": "test_story_123",
            "story_id": "test_story_123",
            "story_text": "Once upon a time, there was a brave little rabbit.",
            "age": 5,
            "language": "en"
        }
    }


# =============================================================================
# Pytest Configuration
# =============================================================================

@pytest.fixture(autouse=True)
def reset_circuit_breakers():
    """Reset circuit breaker state between tests."""
    from src.utils.resilience import CircuitBreaker
    CircuitBreaker._instances.clear()
    yield


def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
