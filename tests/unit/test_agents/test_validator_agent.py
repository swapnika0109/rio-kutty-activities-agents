"""
Unit tests for Validator Agent.
"""

import pytest
from src.agents.validators.validator_agent import ValidatorAgent


class TestValidatorAgent:
    """Tests for ValidatorAgent class."""
    
    @pytest.fixture
    def validator(self):
        return ValidatorAgent()
    
    # MCQ Validation Tests
    
    def test_validate_mcq_success(self, validator):
        """Test MCQ validation passes with valid data."""
        state = {
            "activities": {
                "mcq": [
                    {"question": "Q1?", "options": ["A", "B", "C"], "correct": "A"},
                    {"question": "Q2?", "options": ["A", "B", "C"], "correct": "B"},
                ]
            },
            "completed": [],
            "errors": {}
        }
        
        result = validator.validate_mcq(state)
        
        assert "completed" in result
        assert "mcq" in result["completed"]
        assert "errors" not in result or "mcq" not in result.get("errors", {})
    
    def test_validate_mcq_missing_activity(self, validator):
        """Test MCQ validation fails when activity is missing."""
        state = {
            "activities": {},
            "completed": [],
            "errors": {}
        }
        
        result = validator.validate_mcq(state)
        
        assert "errors" in result
        assert "mcq" in result["errors"]
    
    # Art Validation Tests
    
    def test_validate_art_success(self, validator):
        """Test Art validation passes with valid data."""
        state = {
            "activities": {
                "art": {
                    "title": "Paper Butterfly",
                    "materials": ["paper", "scissors"],
                    "steps": ["Cut", "Fold", "Decorate"]
                }
            },
            "completed": [],
            "errors": {}
        }
        
        result = validator.validate_art(state)
        
        assert "completed" in result
        assert "art" in result["completed"]
    
    def test_validate_art_missing_required_fields(self, validator):
        """Test Art validation fails when required fields missing."""
        state = {
            "activities": {
                "art": {
                    "title": "Incomplete Activity"
                    # Missing materials and steps
                }
            },
            "completed": [],
            "errors": {}
        }
        
        result = validator.validate_art(state)
        
        assert "errors" in result
        assert "art" in result["errors"]
    
    # Science Validation Tests
    
    def test_validate_science_success(self, validator):
        """Test Science validation passes with valid data."""
        state = {
            "activities": {
                "science": [{
                    "title": "Water Experiment",
                    "materials": ["water", "cup"],
                    "Instructions": ["Pour", "Observe"]
                }]
            },
            "completed": [],
            "errors": {}
        }
        
        result = validator.validate_science(state)
        
        assert "completed" in result
        assert "science" in result["completed"]
    
    # Moral Validation Tests
    
    def test_validate_moral_success(self, validator):
        """Test Moral validation passes with valid data."""
        state = {
            "activities": {
                "moral": [{
                    "title": "Sharing Activity",
                    "materials": ["toys"],
                    "Instructions": ["Share", "Take turns"]
                }]
            },
            "completed": [],
            "errors": {}
        }
        
        result = validator.validate_moral(state)
        
        assert "completed" in result
        assert "moral" in result["completed"]
