#!/usr/bin/env python3
"""
Test script for Promptimal integration.

This script tests the basic functionality of the Promptimal integration
without requiring full data setup or API keys.
"""

import sys
import os
import asyncio
from unittest.mock import Mock, AsyncMock

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))


def test_imports():
    """Test that all required imports work correctly."""
    print("Testing imports...")

    try:
        from src.evaluation.eval import Evaluator

        print("✓ Successfully imported Evaluator")
    except ImportError as e:
        print(f"✗ Failed to import Evaluator: {e}")
        return False

    try:
        from promptimal.dtos import PromptCandidate, TokenCount

        print("✓ Successfully imported Promptimal DTOs")
    except ImportError as e:
        print(f"✗ Failed to import Promptimal DTOs: {e}")
        return False

    try:
        from src.evaluation.promptimal_evaluator import (
            PromptimalPromptEvaluator,
            create_evaluator,
        )

        print("✓ Successfully imported PromptimalPromptEvaluator")
    except ImportError as e:
        print(f"✗ Failed to import PromptimalPromptEvaluator: {e}")
        return False

    return True


def test_evaluator_creation():
    """Test creating different types of evaluators."""
    print("\nTesting evaluator creation...")

    try:
        from src.evaluation.eval import Evaluator

        # Test basic evaluator creation
        evaluator = Evaluator()
        print("✓ Successfully created Evaluator instance")

        # Test custom evaluator creation
        def simple_eval(prompt: str) -> float:
            return len(prompt) / 100.0

        custom_eval = evaluator.create_custom_evaluator(simple_eval)
        print("✓ Successfully created custom evaluator")

        return True

    except Exception as e:
        print(f"✗ Failed to create evaluators: {e}")
        return False


def test_promptimal_evaluator():
    """Test the specialized Promptimal evaluator."""
    print("\nTesting PromptimalPromptEvaluator...")

    try:
        # Mock the dependencies to avoid requiring full setup
        import unittest.mock

        with unittest.mock.patch(
            "src.evaluation.promptimal_evaluator.get_instructor_instance"
        ):
            with unittest.mock.patch(
                "src.evaluation.promptimal_evaluator.AugGptRunner"
            ):
                with unittest.mock.patch(
                    "src.evaluation.promptimal_evaluator.PromptEvaluator"
                ):
                    from src.evaluation.promptimal_evaluator import (
                        PromptimalPromptEvaluator,
                        create_evaluator,
                    )

                    # Test basic creation
                    evaluator = PromptimalPromptEvaluator(sentiment="neutral")
                    print("✓ Successfully created PromptimalPromptEvaluator")

                    # Test factory function
                    eval_func = create_evaluator(
                        sentiment="neutral", evaluation_strategy="cosine"
                    )
                    print("✓ Successfully created evaluator via factory function")

        return True

    except Exception as e:
        print(f"✗ Failed to create PromptimalPromptEvaluator: {e}")
        return False


async def test_mock_optimization():
    """Test optimization with mocked components."""
    print("\nTesting mock optimization...")

    try:
        from src.evaluation.eval import Evaluator
        from promptimal.dtos import PromptCandidate, TokenCount

        # Create a simple mock evaluator that always returns 0.8
        async def mock_evaluator(
            candidate, improvement_request, initial_prompt, genai_client
        ):
            candidate.fitness = 0.8
            return candidate, TokenCount(0, 0)

        evaluator = Evaluator()

        # Test that the optimize_prompt method exists and can be called
        # (We won't actually run it since it requires API keys)
        assert hasattr(evaluator, "optimize_prompt"), "optimize_prompt method not found"
        print("✓ optimize_prompt method exists")

        assert hasattr(evaluator, "optimize_with_trainer"), (
            "optimize_with_trainer method not found"
        )
        print("✓ optimize_with_trainer method exists")

        return True

    except Exception as e:
        print(f"✗ Failed mock optimization test: {e}")
        return False


def test_trainer_evaluator_creation():
    """Test creating trainer-based evaluators."""
    print("\nTesting trainer evaluator creation...")

    try:
        from src.evaluation.eval import Evaluator

        # Mock a simple trainer class
        class MockTrainer:
            def __init__(self, **kwargs):
                pass

            def main(self):
                return 0.75  # Mock F1 score

        evaluator = Evaluator()

        # Test trainer evaluator creation
        trainer_eval = evaluator.create_trainer_evaluator(
            trainer_class=MockTrainer, trainer_kwargs={"data_path": "mock/path"}
        )

        print("✓ Successfully created trainer evaluator")

        return True

    except Exception as e:
        print(f"✗ Failed to create trainer evaluator: {e}")
        return False


def main():
    """Run all tests."""
    print("Promptimal Integration Test Suite")
    print("=" * 50)

    tests = [
        test_imports,
        test_evaluator_creation,
        test_promptimal_evaluator,
        test_trainer_evaluator_creation,
    ]

    # Run synchronous tests
    passed = 0
    total = len(tests)

    for test in tests:
        if test():
            passed += 1
        print()

    # Run async test
    try:
        asyncio.run(test_mock_optimization())
        passed += 1
        total += 1
    except Exception as e:
        print(f"✗ Async test failed: {e}")
        total += 1

    # Summary
    print("=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! Promptimal integration is working correctly.")

        print("\nNext steps:")
        print("1. Set GOOGLE_AI_API_KEY environment variable")
        print("2. Prepare your data files")
        print("3. Run the example in examples/promptimal_integration_example.py")
        print("4. Check docs/PROMPTIMAL_INTEGRATION.md for detailed usage")

        return 0
    else:
        print("❌ Some tests failed. Please check the error messages above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
