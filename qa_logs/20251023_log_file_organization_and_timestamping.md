# QA Log: 2025-10-23 - Log File Organization and Timestamping

## Summary
Implemented automatic log file organization with timestamped filenames in the `logs/` directory. When `--log-output file` is used without specifying a log file, the system now automatically creates timestamped log files in the `logs/` folder.

## Changes Made

### 1. Logging Configuration (`src/computable_flows_shim/logging.py`)
- **Added datetime import** for timestamp generation
- **Modified file handler logic** to auto-generate timestamped filenames when `log_file` is None
- **Updated validation** to allow file output without explicit log_file specification
- **Default filename format**: `logs/afs_YYYYMMDD_HHMMSS.log`

### 2. CLI Interface (`src/scripts/cfs_cli.py`)
- **Updated --log-file argument help text** to indicate it's optional and show default behavior

### 3. Documentation (`README.md`)
- **Updated environment variable documentation** to show default log file location
- **Updated CLI examples** to demonstrate automatic timestamped file creation

### 4. Test Updates (`tests/test_logging.py`)
- **Modified test_configure_file_without_log_file** to verify automatic file creation instead of expecting an error

## Implementation Details

### Automatic Filename Generation
```python
if not log_file:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"logs/afs_{timestamp}.log"
```

### Validation Logic
- File output no longer requires explicit log_file specification
- Custom log files must be absolute paths or start with "logs/" for organization

## Testing Results

### Unit Tests
- ✅ All 20 logging tests pass
- ✅ All 5 controller tests pass
- ✅ Total: 25/25 tests passing

### Integration Testing
- ✅ CLI accepts `--log-output file` without `--log-file`
- ✅ Timestamped log files created in `logs/` directory
- ✅ Log files contain properly formatted JSON output
- ✅ Multiple invocations create separate timestamped files

### Example Log Files Created
```
logs/
├── afs_20251023_202827.log  # From CLI help command
├── afs_20251023_202901.log  # From test script
└── pytest_debug.log         # Existing file
```

## Benefits

1. **Automatic Organization**: All log files are centralized in the `logs/` directory
2. **Timestamp Tracking**: Each invocation creates uniquely identifiable log files
3. **No Configuration Required**: Users can simply use `--log-output file` for basic logging
4. **Backward Compatibility**: Existing `--log-file` specifications still work
5. **Debugging Support**: Easy to correlate log files with specific execution times

## Usage Examples

### Environment Variables
```bash
export AFS_LOG_OUTPUT=file
# Creates: logs/afs_YYYYMMDD_HHMMSS.log
```

### CLI
```bash
# Automatic timestamped file
python -m computable_flows_shim.cli --log-output file [command]

# Custom file (still supported)
python -m computable_flows_shim.cli --log-output file --log-file logs/my_custom.log [command]
```

### Programmatic
```python
configure_logging(output="file")  # Auto-creates timestamped file
configure_logging(output="file", log_file="logs/custom.log")  # Custom file
```

## Validation
- ✅ Code changes implement required functionality
- ✅ Tests pass and cover new behavior
- ✅ Documentation updated for users
- ✅ Backward compatibility maintained
- ✅ Log files properly organized and timestamped