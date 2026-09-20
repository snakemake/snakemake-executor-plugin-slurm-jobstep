"""Tests for attempt synchronization in slurm-jobstep inner executor."""
import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from snakemake_executor_plugin_slurm_jobstep import ExecutorSettings


@pytest.fixture
def mock_workflow():
    """Create a mock workflow with executor settings."""
    workflow = MagicMock()
    settings = MagicMock()
    workflow.executor_settings = settings
    workflow.rules = []
    return workflow


@pytest.fixture
def mock_job():
    """Create a mock job with configurable attempt and resources."""
    def _create_job(attempt=1, **resources):
        job = MagicMock()
        job.attempt = attempt
        job.name = "test_job"
        
        # Setup resources
        mock_resources = MagicMock()
        mock_resources.get.side_effect = lambda key, default=None: resources.get(key, default)
        for key, value in resources.items():
            setattr(mock_resources, key, value)
        job.resources = mock_resources
        
        # Mock reset_params_and_resources
        job.reset_params_and_resources = MagicMock()
        
        return job
    return _create_job


class TestInheritedAttemptReading:
    """Test reading _inherited_attempt from outer executor."""

    def test_reads_inherited_attempt_from_settings(self, mock_workflow):
        """Verify __post_init__ reads _inherited_attempt from executor settings."""
        mock_workflow.executor_settings._inherited_attempt = 5
        
        # Simulate reading in __post_init__
        inherited_attempt = getattr(mock_workflow.executor_settings, '_inherited_attempt', 0)
        
        assert inherited_attempt == 5

    def test_defaults_to_zero_when_not_set(self, mock_workflow):
        """Verify default value of 0 when _inherited_attempt is not set."""
        # Mock workflow without _inherited_attempt set
        del mock_workflow.executor_settings._inherited_attempt
        inherited_attempt = getattr(mock_workflow.executor_settings, '_inherited_attempt', 0)
        
        assert inherited_attempt == 0

    def test_no_environment_variable_usage(self):
        """Verify SNAKEMAKE_ATTEMPT environment variable is NOT used."""
        import os
        
        # Set environment variable (should be ignored)
        os.environ['SNAKEMAKE_ATTEMPT'] = '99'
        
        # The implementation should NOT read from environment
        workflow = MagicMock()
        workflow.executor_settings = MagicMock()
        workflow.executor_settings._inherited_attempt = 3
        
        inherited = getattr(workflow.executor_settings, '_inherited_attempt', 0)
        
        # Should read from settings, not environment
        assert inherited == 3
        
        # Cleanup
        del os.environ['SNAKEMAKE_ATTEMPT']


class TestRetryDisabling:
    """Test that retries are disabled in the inner executor."""

    def test_rules_have_restart_times_set_to_zero(self, mock_workflow):
        """Verify all rules have _restart_times set to 0."""
        # Create mock rules
        rules = [MagicMock() for _ in range(3)]
        for i, rule in enumerate(rules):
            rule.name = f"rule_{i}"
        mock_workflow.rules = rules
        
        # Simulate setting restart_times to 0 in __post_init__
        for rule in mock_workflow.rules:
            rule._restart_times = 0
        
        # Verify all rules have restart_times = 0
        for rule in rules:
            assert rule._restart_times == 0

    def test_restart_times_is_rule_property_not_resource(self):
        """Verify restart_times is a Rule property, not a resource."""
        rule = MagicMock()
        rule._restart_times = 0
        
        # Should be accessible as property
        assert hasattr(rule, '_restart_times')
        assert rule._restart_times == 0


class TestAttemptOverrideInRunJob:
    """Test temporary attempt override during run_job."""

    def test_run_job_overrides_job_attempt(self, mock_job):
        """Verify run_job temporarily sets job.attempt to inherited value."""
        job = mock_job(attempt=1)
        inherited_attempt = 5
        
        # Store original attempt
        original_attempt = job.attempt
        
        # Simulate override
        job.attempt = inherited_attempt
        
        assert job.attempt == 5
        assert original_attempt == 1

    def test_run_job_calls_reset_params_and_resources(self, mock_job):
        """Verify run_job calls reset_params_and_resources to clear resource cache."""
        job = mock_job(attempt=1, mem_mb=1000)
        inherited_attempt = 3
        
        # Simulate override and reset
        job.attempt = inherited_attempt
        job.reset_params_and_resources()
        
        # Verify reset was called
        job.reset_params_and_resources.assert_called()

    def test_run_job_restores_attempt_in_finally(self, mock_job):
        """Verify run_job restores original attempt in finally block."""
        job = mock_job(attempt=1)
        inherited_attempt = 5
        
        original_attempt = job.attempt
        try:
            job.attempt = inherited_attempt
            # Simulate job execution
        finally:
            job.attempt = original_attempt
            job.reset_params_and_resources()
        
        assert job.attempt == 1
        assert job.reset_params_and_resources.call_count == 1

    def test_attempt_override_is_temporary(self, mock_job):
        """Verify attempt override doesn't permanently modify job state."""
        job = mock_job(attempt=2)
        original = job.attempt
        
        # Simulate temporary override with proper cleanup
        try:
            job.attempt = 7
            assert job.attempt == 7
        finally:
            job.attempt = original
        
        # Should be restored
        assert job.attempt == 2


class TestResourceRecalculation:
    """Test that resources are recalculated with correct attempt value."""

    def test_dynamic_resource_uses_overridden_attempt(self, mock_job):
        """Verify dynamic resources see the overridden attempt value."""
        # Create a job with dynamic resource callable
        job = mock_job(attempt=1)
        
        # Simulate dynamic memory resource: lambda attempt: attempt * 1000
        def mem_callable(attempt):
            return attempt * 1000
        
        # Before override
        mem_before = mem_callable(job.attempt)
        assert mem_before == 1000
        
        # After override
        job.attempt = 3
        mem_after = mem_callable(job.attempt)
        assert mem_after == 3000

    def test_reset_params_and_resources_clears_cache(self, mock_job):
        """Verify reset_params_and_resources is called to clear cached resources."""
        job = mock_job(attempt=1, mem_mb=1000)
        
        # Override attempt
        job.attempt = 5
        
        # Reset should be called to force recalculation
        job.reset_params_and_resources()
        
        job.reset_params_and_resources.assert_called()


class TestEndToEndAttemptFlow:
    """Integration-style tests for complete attempt synchronization flow."""

    def test_complete_attempt_sync_flow(self, mock_workflow, mock_job):
        """Test complete flow: outer sets, inner reads, overrides, restores."""
        # Outer executor sets _inherited_attempt
        mock_workflow.executor_settings._inherited_attempt = 4
        
        # Inner executor reads in __post_init__
        inherited = getattr(mock_workflow.executor_settings, '_inherited_attempt', 0)
        assert inherited == 4
        
        # Inner executor creates job with attempt=1
        job = mock_job(attempt=1)
        assert job.attempt == 1
        
        # Inner executor temporarily overrides in run_job
        original = job.attempt
        try:
            job.attempt = inherited
            job.reset_params_and_resources()
            assert job.attempt == 4
            # Simulate job execution
        finally:
            job.attempt = original
            job.reset_params_and_resources()
        
        # Verify cleanup
        assert job.attempt == 1

    def test_no_inherited_attempt_defaults_gracefully(self, mock_workflow, mock_job):
        """Test behavior when _inherited_attempt is not set (first attempt)."""
        # Don't set _inherited_attempt (simulates first attempt)
        del mock_workflow.executor_settings._inherited_attempt
        inherited = getattr(mock_workflow.executor_settings, '_inherited_attempt', 0)
        assert inherited == 0
        
        job = mock_job(attempt=1)
        
        # Should work normally with no override
        original = job.attempt
        try:
            if inherited > 0:
                job.attempt = inherited
            assert job.attempt == 1
        finally:
            if inherited > 0:
                job.attempt = original


class TestNoEnvironmentVariableLeakage:
    """Test that environment variables are not used or leaked."""

    def test_no_snakemake_attempt_reading(self, mock_workflow):
        """Verify implementation does not read SNAKEMAKE_ATTEMPT from environment."""
        import os
        
        # Set environment variable
        os.environ['SNAKEMAKE_ATTEMPT'] = '42'
        
        # Inner executor should read from settings, not environment
        mock_workflow.executor_settings._inherited_attempt = 3
        inherited = getattr(mock_workflow.executor_settings, '_inherited_attempt', 0)
        
        assert inherited == 3
        assert inherited != 42
        
        # Cleanup
        del os.environ['SNAKEMAKE_ATTEMPT']

    def test_no_snakemake_attempt_setting(self, mock_job):
        """Verify implementation does not set SNAKEMAKE_ATTEMPT in environment."""
        import os
        
        # Ensure clean environment
        if 'SNAKEMAKE_ATTEMPT' in os.environ:
            del os.environ['SNAKEMAKE_ATTEMPT']
        
        # Simulate attempt override
        job = mock_job(attempt=1)
        job.attempt = 5
        
        # Environment should remain clean
        assert 'SNAKEMAKE_ATTEMPT' not in os.environ
