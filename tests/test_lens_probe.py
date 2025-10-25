"""
Contract tests for Lens Probe functionality.

Tests compressibility metrics, reconstruction error analysis, and lens selection
for multiscale transforms in builder mode.
"""

import jax.numpy as jnp
import jax.random as random
import pytest

from computable_flows_shim.multi.transform_op import make_transform


@pytest.mark.dtype_parametrized
class TestLensProbeContract:
    """Contract tests for lens probe compressibility and reconstruction analysis."""

    @pytest.fixture(autouse=True)
    def setup_method(self, float_dtype):
        """Set up test method with dtype fixture."""
        self.float_dtype = float_dtype
        # Set tolerance based on precision
        self.tolerance = 1e-5 if float_dtype == jnp.float32 else 1e-12

    @pytest.fixture
    def sample_data_1d(self):
        """Generate 1D test signal with known compressibility properties."""
        key = random.PRNGKey(42)
        # Create a signal with different frequency components
        x = jnp.linspace(0, 10, 256, dtype=self.float_dtype)
        signal = jnp.sin(x) + 0.5 * jnp.sin(4 * x) + 0.2 * jnp.sin(16 * x)
        return signal

    @pytest.fixture
    def sample_data_2d(self):
        """Generate 2D test image with known compressibility properties."""
        key = random.PRNGKey(42)
        # Create a simple 2D pattern
        x = jnp.linspace(-1, 1, 64, dtype=self.float_dtype)
        y = jnp.linspace(-1, 1, 64, dtype=self.float_dtype)
        X, Y = jnp.meshgrid(x, y)
        image = jnp.exp(-(X**2 + Y**2)) + 0.3 * jnp.sin(10 * X) * jnp.sin(10 * Y)
        return image

    @pytest.fixture
    def candidate_transforms_1d(self):
        """Get candidate 1D transforms for testing."""
        return [
            make_transform("haar", levels=3, ndim=1),
            make_transform("db4", levels=3, ndim=1),
        ]

    @pytest.fixture
    def candidate_transforms_2d(self):
        """Get candidate 2D transforms for testing."""
        return [
            make_transform("haar", levels=2, ndim=2),
            make_transform("db4", levels=2, ndim=2),
        ]

    def test_compressibility_metric_calculation(
        self, sample_data_1d, candidate_transforms_1d
    ):
        """Test that compressibility metrics are calculated correctly."""
        from computable_flows_shim.multi.lens_probe import calculate_compressibility

        transform = candidate_transforms_1d[0]  # haar
        coeffs = transform.forward(sample_data_1d)

        # Compressibility should be a dict with sparsity per band
        compressibility = calculate_compressibility(coeffs)

        assert isinstance(compressibility, dict)
        assert "overall_sparsity" in compressibility
        assert "band_sparsity" in compressibility
        assert "energy_distribution" in compressibility

        # Overall sparsity should be between 0 and 1
        assert 0.0 <= compressibility["overall_sparsity"] <= 1.0

        # Band sparsity should be a list with one entry per decomposition level + approximation
        assert len(compressibility["band_sparsity"]) == transform.levels + 1

        # All band sparsities should be valid
        for sparsity in compressibility["band_sparsity"]:
            assert 0.0 <= sparsity <= 1.0

    def test_reconstruction_error_calculation(
        self, sample_data_1d, candidate_transforms_1d
    ):
        """Test that reconstruction error is calculated correctly."""
        from computable_flows_shim.multi.lens_probe import (
            calculate_reconstruction_error,
        )

        transform = candidate_transforms_1d[0]  # haar
        coeffs = transform.forward(sample_data_1d)
        reconstruction = transform.inverse(coeffs)

        error_metrics = calculate_reconstruction_error(sample_data_1d, reconstruction)

        assert isinstance(error_metrics, dict)
        assert "mse" in error_metrics
        assert "rmse" in error_metrics
        assert "relative_error" in error_metrics
        assert "max_error" in error_metrics

        # Errors should be non-negative
        for key, value in error_metrics.items():
            assert value >= 0.0

        # For perfect reconstruction (haar on dyadic length), relative error should be very small
        # Allow for floating point precision issues with non-power-of-2 signal lengths
        assert error_metrics["relative_error"] < self.tolerance

    def test_compressibility_vs_sparsity_tradeoff(
        self, sample_data_1d, candidate_transforms_1d
    ):
        """Test that compressibility correlates with sparsity thresholding."""
        from computable_flows_shim.multi.lens_probe import calculate_compressibility

        transform = candidate_transforms_1d[0]  # haar
        coeffs = transform.forward(sample_data_1d)

        # Test different sparsity thresholds
        thresholds = [1e-8, 1e-4, 1e-2]

        sparsities = []
        for threshold in thresholds:
            compressibility = calculate_compressibility(coeffs, threshold=threshold)
            sparsities.append(compressibility["overall_sparsity"])

        # Higher thresholds should give lower sparsity (fewer non-zero coefficients)
        assert sparsities[0] >= sparsities[1] >= sparsities[2]

    def test_lens_probe_builder_mode(self, sample_data_1d, candidate_transforms_1d):
        """Test that lens probe runs in builder mode and selects best transform."""
        from computable_flows_shim.multi.lens_probe import run_lens_probe

        # Run lens probe on candidates
        probe_results = run_lens_probe(
            data=sample_data_1d,
            candidates=candidate_transforms_1d,
            target_sparsity=0.8,  # Target 80% sparsity
        )

        assert isinstance(probe_results, dict)
        assert "selected_lens" in probe_results
        assert "candidate_results" in probe_results
        assert "selection_criteria" in probe_results

        # Should have results for each candidate
        assert len(probe_results["candidate_results"]) == len(candidate_transforms_1d)

        # Each candidate result should have compressibility and reconstruction metrics
        for candidate_name, results in probe_results["candidate_results"].items():
            assert "compressibility" in results
            assert "reconstruction_error" in results
            assert "sparsity_at_target" in results

    def test_lens_selection_by_reconstruction_error(
        self, sample_data_1d, candidate_transforms_1d
    ):
        """Test lens selection prioritizes reconstruction quality."""
        from computable_flows_shim.multi.lens_probe import run_lens_probe

        # Run with reconstruction error as primary criterion
        probe_results = run_lens_probe(
            data=sample_data_1d,
            candidates=candidate_transforms_1d,
            selection_rule="min_reconstruction_error",
        )

        selected = probe_results["selected_lens"]
        assert selected in [t.name for t in candidate_transforms_1d]

        # Verify the selected lens actually has the best reconstruction error
        candidate_results = probe_results["candidate_results"]
        selected_error = candidate_results[selected]["reconstruction_error"][
            "relative_error"
        ]

        for name, results in candidate_results.items():
            if name != selected:
                assert (
                    results["reconstruction_error"]["relative_error"] >= selected_error
                )

    def test_lens_probe_2d_support(self, sample_data_2d, candidate_transforms_2d):
        """Test that lens probe works with 2D transforms."""
        from computable_flows_shim.multi.lens_probe import run_lens_probe

        probe_results = run_lens_probe(
            data=sample_data_2d, candidates=candidate_transforms_2d, target_sparsity=0.9
        )

        assert isinstance(probe_results, dict)
        assert "selected_lens" in probe_results
        assert len(probe_results["candidate_results"]) == len(candidate_transforms_2d)

    def test_probe_result_consistency(self, sample_data_1d, candidate_transforms_1d):
        """Test that probe results are consistent across multiple runs."""
        from computable_flows_shim.multi.lens_probe import run_lens_probe

        # Run probe multiple times
        results1 = run_lens_probe(
            data=sample_data_1d, candidates=candidate_transforms_1d
        )
        results2 = run_lens_probe(
            data=sample_data_1d, candidates=candidate_transforms_1d
        )

        # Selected lens should be the same
        assert results1["selected_lens"] == results2["selected_lens"]

        # Results should be numerically close
        for candidate in candidate_transforms_1d:
            name = candidate.name
            comp1 = results1["candidate_results"][name]["compressibility"][
                "overall_sparsity"
            ]
            comp2 = results2["candidate_results"][name]["compressibility"][
                "overall_sparsity"
            ]
            assert abs(comp1 - comp2) < self.tolerance

    def test_edge_case_empty_data(self):
        """Test lens probe handles edge cases gracefully."""
        from computable_flows_shim.multi.lens_probe import run_lens_probe

        # Empty data should raise appropriate error
        with pytest.raises(ValueError):
            run_lens_probe(data=jnp.array([]), candidates=[])

    def test_edge_case_single_candidate(self, sample_data_1d, candidate_transforms_1d):
        """Test lens probe with single candidate."""
        from computable_flows_shim.multi.lens_probe import run_lens_probe

        single_candidate = [candidate_transforms_1d[0]]
        probe_results = run_lens_probe(data=sample_data_1d, candidates=single_candidate)

        assert probe_results["selected_lens"] == single_candidate[0].name
        assert len(probe_results["candidate_results"]) == 1
