"""Tier 1 tests for the /setup-cluster skill.

Validates that the agent recommends correct cluster configurations
for CPU (local models) and GPU (foundation/global models).
"""

import pytest


@pytest.mark.tier1
class TestSetupClusterCPU:
    """Test CPU cluster configuration for local models."""

    def test_cpu_config_for_local_models(self, skill_prompt, run_agent):
        """Agent should recommend CPU cluster config for local models on AWS."""
        prompt = skill_prompt("setup-cluster")
        result = run_agent(
            system_prompt=prompt,
            user_prompt=("Set up cluster for local models only on AWS. Confirm all defaults without asking questions."),
        )

        response = result["final_response"]

        # Should mention CPU node type for AWS
        assert "i3.xlarge" in response, f"Expected 'i3.xlarge' for AWS CPU. Got: {response[:500]}"

        # Should mention CPU runtime
        assert "17.3.x-cpu-ml-scala2.13" in response, (
            f"Expected CPU runtime '17.3.x-cpu-ml-scala2.13'. Got: {response[:500]}"
        )

        # Should NOT mention GPU node types
        response_lower = response.lower()
        assert "g5.12xlarge" not in response_lower, "CPU-only config should not mention GPU node type g5.12xlarge"


@pytest.mark.tier1
class TestSetupClusterGPU:
    """Test GPU cluster configuration for foundation models."""

    def test_gpu_config_for_foundation_models(self, skill_prompt, run_agent):
        """Agent should recommend GPU cluster config for foundation models on AWS."""
        prompt = skill_prompt("setup-cluster")
        result = run_agent(
            system_prompt=prompt,
            user_prompt=("Set up cluster for foundation models on AWS. Confirm all defaults without asking questions."),
        )

        response = result["final_response"]

        # Should mention GPU node type for AWS
        assert "g5.12xlarge" in response, f"Expected 'g5.12xlarge' for AWS GPU. Got: {response[:500]}"

        # Should mention GPU runtime
        assert "18.0.x-gpu-ml-scala2.13" in response, (
            f"Expected GPU runtime '18.0.x-gpu-ml-scala2.13'. Got: {response[:500]}"
        )

        # Should mention single-node (0 workers)
        assert "0" in response or "single" in response.lower(), (
            f"Expected mention of 0 workers or single-node. Got: {response[:500]}"
        )
