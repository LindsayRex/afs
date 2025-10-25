import json
import logging
import os
import tempfile
from unittest.mock import patch

import pytest

from computable_flows_shim.logging import (
    JSONFormatter,
    SDKLogger,
    _configure_from_env,
    configure_logging,
    get_logger,
    log_performance,
)


class TestJSONFormatter:
    """Test JSON formatter functionality."""

    def test_format_basic_log_record(self):
        """Test basic log record formatting."""
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        result = formatter.format(record)
        parsed = json.loads(result)

        assert parsed["level"] == "INFO"
        assert parsed["module"] == "test.logger"
        assert parsed["message"] == "Test message"
        assert "timestamp" in parsed

    def test_format_with_extra_fields(self):
        """Test log record formatting with extra fields."""
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test.logger",
            level=logging.DEBUG,
            pathname="test.py",
            lineno=10,
            msg="Operation completed",
            args=(),
            exc_info=None,
        )
        # Set extra directly on the record
        record.__dict__.update({"duration_ms": 45.67, "success": True})

        result = formatter.format(record)
        parsed = json.loads(result)

        assert parsed["extra"]["duration_ms"] == 45.67
        assert parsed["extra"]["success"] is True


class TestSDKLogger:
    """Test SDK logger configuration."""

    def test_configure_valid_parameters(self):
        """Test configuration with valid parameters."""
        SDKLogger.configure(level="DEBUG", format="json", output="null")

        logger = logging.getLogger("afs")
        assert logger.level == logging.DEBUG
        assert len(logger.handlers) == 1

    def test_configure_invalid_level(self):
        """Test configuration with invalid level raises ValueError."""
        with pytest.raises(ValueError, match="Invalid log level"):
            SDKLogger.configure(level="INVALID")

    def test_configure_invalid_format(self):
        """Test configuration with invalid format raises ValueError."""
        with pytest.raises(ValueError, match="Invalid log format"):
            SDKLogger.configure(format="INVALID")

    def test_configure_invalid_output(self):
        """Test configuration with invalid output raises ValueError."""
        with pytest.raises(ValueError, match="Invalid log output"):
            SDKLogger.configure(output="INVALID")

    def test_configure_file_without_log_file(self):
        """Test file output without log_file creates timestamped file in logs directory."""
        import tempfile

        # Create a temporary logs directory for testing
        with tempfile.TemporaryDirectory() as temp_dir:
            logs_dir = os.path.join(temp_dir, "logs")
            os.makedirs(logs_dir)

            # Change to temp directory so logs/ is relative to it
            original_cwd = os.getcwd()
            os.chdir(temp_dir)

            try:
                SDKLogger.configure(output="file")
                logger = logging.getLogger("afs")
                assert isinstance(logger.handlers[0], logging.FileHandler)

                # Check that a log file was created in logs directory
                log_files = os.listdir(logs_dir)
                assert len(log_files) == 1
                assert log_files[0].startswith("afs_")
                assert log_files[0].endswith(".log")

                # Close the handler to release the file
                logger.handlers[0].close()
            finally:
                os.chdir(original_cwd)

    @patch("sys.stdout")
    def test_configure_stdout_output(self, mock_stdout):
        """Test stdout output configuration."""
        SDKLogger.configure(output="stdout")
        logger = logging.getLogger("afs")
        assert isinstance(logger.handlers[0], logging.StreamHandler)

    def test_configure_file_output(self):
        """Test file output configuration."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp_path = tmp.name

        try:
            SDKLogger.configure(output="file", log_file=tmp_path)
            logger = logging.getLogger("afs")
            assert isinstance(logger.handlers[0], logging.FileHandler)
            # Close the handler to release the file
            logger.handlers[0].close()
        finally:
            os.unlink(tmp_path)

    def test_configure_null_output(self):
        """Test null output configuration."""
        SDKLogger.configure(output="null")
        logger = logging.getLogger("afs")
        assert isinstance(logger.handlers[0], logging.NullHandler)


class TestGetLogger:
    """Test logger factory function."""

    def test_get_logger_creates_correct_name(self):
        """Test get_logger creates logger with correct hierarchical name."""
        logger = get_logger("test.module")
        assert logger.name == "afs.test.module"

    def test_get_logger_returns_logger_instance(self):
        """Test get_logger returns logging.Logger instance."""
        logger = get_logger("test")
        assert isinstance(logger, logging.Logger)


class TestLogPerformance:
    """Test performance logging decorator."""

    def test_decorator_success_case(self):
        """Test decorator logs success with timing."""
        logger = get_logger("test")
        logger.setLevel(logging.DEBUG)

        # Mock the logger to capture calls
        with patch.object(logger, "debug") as mock_debug:

            @log_performance(logger, "test_operation")
            def dummy_function():
                return "result"

            result = dummy_function()

            assert result == "result"
            mock_debug.assert_called_once()
            args, kwargs = mock_debug.call_args
            assert "test_operation completed" in args[0]
            assert "duration_ms" in kwargs["extra"]
            assert kwargs["extra"]["success"] is True

    def test_decorator_failure_case(self):
        """Test decorator logs failure with timing and error."""
        logger = get_logger("test")
        logger.setLevel(logging.DEBUG)

        with patch.object(logger, "error") as mock_error:

            @log_performance(logger, "failing_operation")
            def failing_function():
                raise ValueError("Test error")

            with pytest.raises(ValueError, match="Test error"):
                failing_function()

            mock_error.assert_called_once()
            args, kwargs = mock_error.call_args
            assert "failing_operation failed" in args[0]
            assert "duration_ms" in kwargs["extra"]
            assert kwargs["extra"]["success"] is False
            assert "Test error" in kwargs["extra"]["error"]

    def test_decorator_no_logging_when_disabled(self):
        """Test decorator doesn't log when DEBUG level is disabled."""
        logger = get_logger("test")
        logger.setLevel(logging.WARNING)  # Above DEBUG

        @log_performance(logger, "test_operation")
        def dummy_function():
            return "result"

        # Should not raise any exceptions and should not log
        result = dummy_function()
        assert result == "result"


class TestConfigureLogging:
    """Test public logging configuration API."""

    def test_configure_logging_calls_sdk_logger(self):
        """Test configure_logging calls SDKLogger.configure."""
        with patch(
            "computable_flows_shim.logging.SDKLogger.configure"
        ) as mock_configure:
            configure_logging(level="INFO", format="text", output="stdout")
            mock_configure.assert_called_once_with(
                level="INFO",
                format="text",
                output="stdout",
                log_file=None,
                enable_performance_logging=False,
            )


class TestEnvironmentConfiguration:
    """Test environment variable configuration."""

    @patch.dict(
        os.environ,
        {
            "AFS_LOG_LEVEL": "DEBUG",
            "AFS_LOG_FORMAT": "text",
            "AFS_LOG_OUTPUT": "stdout",
            "AFS_LOG_PERFORMANCE": "true",
        },
    )
    def test_configure_from_env_with_variables(self):
        """Test configuration from environment variables."""
        with patch("computable_flows_shim.logging.configure_logging") as mock_configure:
            _configure_from_env()
            mock_configure.assert_called_once_with(
                level="DEBUG",
                format="text",
                output="stdout",
                log_file=None,
                enable_performance_logging=True,
            )

    @patch.dict(os.environ, {}, clear=True)
    def test_configure_from_env_without_variables(self):
        """Test no configuration when no environment variables."""
        with patch("computable_flows_shim.logging.configure_logging") as mock_configure:
            _configure_from_env()
            mock_configure.assert_not_called()


class TestIntegration:
    """Integration tests for logging system."""

    def test_full_logging_workflow(self):
        """Test complete logging workflow from configuration to output."""
        # Configure logging
        SDKLogger.configure(level="DEBUG", format="json", output="null")

        # Get logger and log messages
        logger = get_logger("integration.test")

        with patch("json.dumps") as mock_json:
            mock_json.return_value = '{"test": "data"}'

            logger.info("Test message", extra={"key": "value"})

            # Verify JSON was called (would be called by formatter)
            # Note: This is a simplified test - in real usage the formatter handles JSON

    def test_logger_hierarchy(self):
        """Test logger hierarchy and level inheritance."""
        # Configure root logger
        SDKLogger.configure(level="INFO")

        # Get child loggers
        parent_logger = get_logger("parent")
        child_logger = get_logger("parent.child")

        # Both should respect the root level
        assert parent_logger.isEnabledFor(logging.INFO)
        assert not parent_logger.isEnabledFor(logging.DEBUG)
        assert child_logger.isEnabledFor(logging.INFO)
        assert not child_logger.isEnabledFor(logging.DEBUG)
