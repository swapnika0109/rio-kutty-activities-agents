"""
Unit tests for Art Agent.
"""

import pytest
import json
from unittest.mock import AsyncMock, patch

from src.agents.activities.art_agent import ArtAgent


class TestArtAgent:
    """Tests for ArtAgent class."""
    
    @pytest.fixture
    def agent(self):
        """Create ArtAgent instance with mocked AI service."""
        with patch("src.agents.activities.art_agent.AIService") as MockAI:
            instance = MockAI.return_value
            instance.generate_content = AsyncMock()
            instance.generate_image = AsyncMock(return_value=b"fake_image")
            agent = ArtAgent()
            agent.ai_service = instance
            yield agent
    
    @pytest.fixture
    def sample_state(self):
        return {
            "story_text": "A colorful butterfly visited the garden.",
            "age": 6,
            "language": "English",
            "activities": {},
            "completed": [],
            "errors": {}
        }
    
    @pytest.mark.asyncio
    async def test_generate_returns_art_activity_without_image(self, agent, sample_state):
        """generate() returns activity JSON only — image is deferred to generate_image()
        so we don't burn FLUX credits on activities that fail eval."""
        mock_response = json.dumps({
            "title": "Butterfly Craft",
            "age_appropriateness": "Great for 6-year-olds",
            "materials": ["paper", "paint"],
            "steps": ["Fold paper", "Paint wings", "Add antennae"],
            "image_generation_prompt": "A colorful butterfly craft"
        })
        agent.ai_service.generate_content.return_value = mock_response

        result = await agent.generate(sample_state)

        assert "activities" in result
        assert "art" in result["activities"]
        # image generation must NOT happen in generate()
        agent.ai_service.generate_image.assert_not_called()
        assert "image" not in result["activities"]["art"]
        assert "completed" in result
        assert "art" in result["completed"]

    @pytest.mark.asyncio
    async def test_generate_image_attaches_uploaded_filename(self, agent, sample_state):
        """generate_image() uploads the PNG and stores the GCS filename on the
        activity. Called only after the eval pass succeeds."""
        agent.storage.upload_file = AsyncMock()
        state_with_activity = {
            **sample_state,
            "activities": {
                "art": {
                    "title": "Butterfly Craft",
                    "image_generation_prompt": "A colorful butterfly craft",
                }
            },
        }

        result = await agent.generate_image(state_with_activity)

        agent.ai_service.generate_image.assert_called_once_with("A colorful butterfly craft")
        agent.storage.upload_file.assert_called_once()
        assert result["activities"]["art"]["image"].startswith("images/")
        assert result["activities"]["art"]["image"].endswith(".png")

    @pytest.mark.asyncio
    async def test_generate_image_sets_none_on_image_failure(self, agent, sample_state):
        """If the image model returns no bytes, set image to None — don't error out."""
        agent.storage.upload_file = AsyncMock()
        agent.ai_service.generate_image.return_value = None
        state_with_activity = {
            **sample_state,
            "activities": {"art": {"image_generation_prompt": "A craft"}},
        }

        result = await agent.generate_image(state_with_activity)

        assert result["activities"]["art"]["image"] is None
        agent.storage.upload_file.assert_not_called()

    @pytest.mark.asyncio
    async def test_generate_returns_error_on_ai_failure(self, agent, sample_state):
        """Test that generate() returns error on AI failure."""
        agent.ai_service.generate_content.side_effect = Exception("AI Error")

        result = await agent.generate(sample_state)

        assert "errors" in result
        assert "art" in result["errors"]
