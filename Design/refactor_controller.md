# Controller Refactoring Plan

## Overview
The `controller.py` file has grown to 775 lines and needs modularization. This plan breaks it down into focused, testable modules while maintaining backward compatibility.

## Current State Analysis
- **File**: `src/computable_flows_shim/controller.py` (775 lines)
- **Dependencies**: 4 direct importers, 8+ internal dependencies
- **Risk Level**: HIGH - Core orchestration component

## Refactoring Strategy
**Approach**: Incremental extraction with backward compatibility maintained throughout.

```json
{
  "refactoring_plan": {
    "version": "1.0",
    "target_file": "src/computable_flows_shim/controller.py",
    "current_lines": 775,
    "target_lines": 200,
    "estimated_duration": "2-3 weeks",
    "risk_level": "HIGH",
    "strategy": "incremental_extraction_with_compatibility"
  },

  "phases": [
    {
      "id": "phase_1",
      "name": "Extract Data Structures",
      "description": "Move simple data classes to separate modules",
      "risk_level": "LOW",
      "estimated_lines_removed": 80,
      "dependencies": [],
      "tasks": [
        {
          "id": "extract_phase_enum",
          "description": "Move Phase enum to controller/types.py",
          "files_created": ["src/computable_flows_shim/controller/types.py"],
          "files_modified": ["src/computable_flows_shim/controller.py"],
          "lines_moved": 10,
          "backward_compatibility": "MAINTAINED - re-export from controller.py",
          "testing": "Update imports in dependent files"
        },
        {
          "id": "extract_controller_config",
          "description": "Move ControllerConfig dataclass to controller/config.py",
          "files_created": ["src/computable_flows_shim/controller/config.py"],
          "files_modified": ["src/computable_flows_shim/controller.py"],
          "lines_moved": 25,
          "backward_compatibility": "MAINTAINED - re-export from controller.py",
          "testing": "Verify config instantiation works"
        },
        {
          "id": "extract_checkpoint_dataclass",
          "description": "Move Checkpoint dataclass to controller/checkpoint.py",
          "files_created": ["src/computable_flows_shim/controller/checkpoint.py"],
          "files_modified": ["src/computable_flows_shim/controller.py"],
          "lines_moved": 20,
          "backward_compatibility": "MAINTAINED - re-export from controller.py",
          "testing": "Test checkpoint creation and rollback functionality"
        }
      ]
    },

    {
      "id": "phase_2",
      "name": "Extract Certificate Logic",
      "description": "Move certificate assessment and validation logic",
      "risk_level": "MEDIUM",
      "estimated_lines_removed": 50,
      "dependencies": ["phase_1"],
      "tasks": [
        {
          "id": "extract_certificate_assessment",
          "description": "Move assess_certificates method to controller/certificates.py",
          "files_created": ["src/computable_flows_shim/controller/certificates.py"],
          "files_modified": ["src/computable_flows_shim/controller.py"],
          "lines_moved": 35,
          "backward_compatibility": "MAINTAINED - method stays on FlightController",
          "testing": "Test certificate assessment with various inputs"
        },
        {
          "id": "extract_certificate_validation",
          "description": "Move certificate validation logic to separate functions",
          "files_created": [],
          "files_modified": ["src/computable_flows_shim/controller/certificates.py"],
          "lines_moved": 15,
          "backward_compatibility": "MAINTAINED - internal logic only",
          "testing": "Test validation with edge cases"
        }
      ]
    },

    {
      "id": "phase_3",
      "name": "Extract Phase Machine",
      "description": "Move RED/AMBER/GREEN phase transition logic",
      "risk_level": "MEDIUM",
      "estimated_lines_removed": 40,
      "dependencies": ["phase_1"],
      "tasks": [
        {
          "id": "extract_phase_machine",
          "description": "Move phase transition logic to controller/phases.py",
          "files_created": ["src/computable_flows_shim/controller/phases.py"],
          "files_modified": ["src/computable_flows_shim/controller.py"],
          "lines_moved": 30,
          "backward_compatibility": "MAINTAINED - method stays on FlightController",
          "testing": "Test phase transitions and telemetry logging"
        },
        {
          "id": "extract_phase_validation",
          "description": "Move phase validation logic",
          "files_created": [],
          "files_modified": ["src/computable_flows_shim/controller/phases.py"],
          "lines_moved": 10,
          "backward_compatibility": "MAINTAINED - internal logic only",
          "testing": "Test invalid phase transitions"
        }
      ]
    },

    {
      "id": "phase_4",
      "name": "Extract Main Loop Components",
      "description": "Break down the massive run_certified_flow method",
      "risk_level": "HIGH",
      "estimated_lines_removed": 200,
      "dependencies": ["phase_1", "phase_2", "phase_3"],
      "tasks": [
        {
          "id": "extract_initialization",
          "description": "Move flow initialization logic to controller/initialization.py",
          "files_created": ["src/computable_flows_shim/controller/initialization.py"],
          "files_modified": ["src/computable_flows_shim/controller.py"],
          "lines_moved": 40,
          "backward_compatibility": "MAINTAINED - internal to run_certified_flow",
          "testing": "Test initialization with various configurations"
        },
        {
          "id": "extract_budget_checks",
          "description": "Move budget limit checking to controller/budget.py",
          "files_created": ["src/computable_flows_shim/controller/budget.py"],
          "files_modified": ["src/computable_flows_shim/controller.py"],
          "lines_moved": 25,
          "backward_compatibility": "MAINTAINED - method stays on FlightController",
          "testing": "Test budget limit enforcement"
        },
        {
          "id": "extract_optimization_loop",
          "description": "Move main optimization loop to controller/optimization.py",
          "files_created": ["src/computable_flows_shim/controller/optimization.py"],
          "files_modified": ["src/computable_flows_shim/controller.py"],
          "lines_moved": 150,
          "backward_compatibility": "MAINTAINED - internal to run_certified_flow",
          "testing": "Test full optimization loop with telemetry"
        },
        {
          "id": "extract_finalization",
          "description": "Move finalization logic to controller/finalization.py",
          "files_created": ["src/computable_flows_shim/controller/finalization.py"],
          "files_modified": ["src/computable_flows_shim/controller.py"],
          "lines_moved": 30,
          "backward_compatibility": "MAINTAINED - internal to run_certified_flow",
          "testing": "Test finalization and cleanup"
        }
      ]
    },

    {
      "id": "phase_5",
      "name": "Extract Telemetry Integration",
      "description": "Move telemetry logging logic to dedicated module",
      "risk_level": "LOW",
      "estimated_lines_removed": 60,
      "dependencies": ["phase_1"],
      "tasks": [
        {
          "id": "extract_telemetry_logging",
          "description": "Move telemetry integration to controller/telemetry.py",
          "files_created": ["src/computable_flows_shim/controller/telemetry.py"],
          "files_modified": ["src/computable_flows_shim/controller.py"],
          "lines_moved": 50,
          "backward_compatibility": "MAINTAINED - telemetry calls stay in place",
          "testing": "Test telemetry event logging"
        },
        {
          "id": "extract_event_constants",
          "description": "Move telemetry event constants to separate module",
          "files_created": ["src/computable_flows_shim/controller/events.py"],
          "files_modified": ["src/computable_flows_shim/controller/telemetry.py"],
          "lines_moved": 10,
          "backward_compatibility": "MAINTAINED - imported where needed",
          "testing": "Verify event constants are accessible"
        }
      ]
    },

    {
      "id": "phase_6",
      "name": "Extract Rollback Logic",
      "description": "Move checkpoint and rollback functionality",
      "risk_level": "MEDIUM",
      "estimated_lines_removed": 40,
      "dependencies": ["phase_1"],
      "tasks": [
        {
          "id": "extract_rollback_logic",
          "description": "Move rollback functionality to controller/rollback.py",
          "files_created": ["src/computable_flows_shim/controller/rollback.py"],
          "files_modified": ["src/computable_flows_shim/controller.py"],
          "lines_moved": 35,
          "backward_compatibility": "MAINTAINED - methods stay on FlightController",
          "testing": "Test rollback to various checkpoints"
        },
        {
          "id": "extract_checkpoint_management",
          "description": "Move checkpoint management logic",
          "files_created": [],
          "files_modified": ["src/computable_flows_shim/controller/rollback.py"],
          "lines_moved": 15,
          "backward_compatibility": "MAINTAINED - internal logic only",
          "testing": "Test checkpoint creation and cleanup"
        }
      ]
    }
  ],

  "integration_testing": {
    "description": "End-to-end testing after each phase",
    "tests_required": [
      "test_controller.py - all existing tests pass",
      "test_runtime.py - integration tests pass",
      "API compatibility - api.py and cfs_cli.py work unchanged",
      "Performance regression - no significant slowdown"
    ],
    "automation": "Run full test suite after each phase completion"
  },

  "rollback_plan": {
    "description": "How to revert if something breaks",
    "strategy": "Git revert individual commits, or restore from backup",
    "backup_files": "Keep copies of original controller.py and __init__.py",
    "testing": "Verify all tests pass after rollback"
  },

  "success_criteria": {
    "lines_reduction": "controller.py reduced from 775 to ~200 lines",
    "modularity": "Each module has single responsibility",
    "test_coverage": "All new modules have >80% test coverage",
    "performance": "No >5% performance regression",
    "compatibility": "All existing imports and APIs work unchanged"
  },

  "file_structure_after": {
    "description": "Target directory structure",
    "structure": {
      "src/computable_flows_shim/controller/": {
        "__init__.py": "Re-exports for backward compatibility",
        "controller.py": "Core FlightController class (~200 lines)",
        "types.py": "Phase enum and type definitions",
        "config.py": "ControllerConfig dataclass",
        "checkpoint.py": "Checkpoint dataclass",
        "certificates.py": "Certificate assessment logic",
        "phases.py": "Phase machine logic",
        "initialization.py": "Flow initialization",
        "budget.py": "Budget checking logic",
        "optimization.py": "Main optimization loop",
        "finalization.py": "Flow finalization",
        "telemetry.py": "Telemetry integration",
        "events.py": "Telemetry event constants",
        "rollback.py": "Checkpoint and rollback logic"
      }
    }
  }
}
```

## Implementation Notes

### Phase Ordering Rationale
1. **Start with data structures** (Phase 1) - Low risk, establishes pattern
2. **Extract certificates** (Phase 2) - Self-contained logic
3. **Extract phases** (Phase 3) - Core state machine logic
4. **Break main loop** (Phase 4) - Most complex, done last
5. **Extract telemetry** (Phase 5) - Cross-cutting concern
6. **Extract rollback** (Phase 6) - Depends on checkpoint dataclass

### Risk Mitigation
- **Backward compatibility maintained** throughout via re-exports
- **Incremental testing** after each task
- **Integration tests** after each phase
- **Easy rollback** if issues arise

### Development Workflow
1. Create new module file
2. Move code with minimal changes
3. Add re-export to controller/__init__.py
4. Update controller.py to import from new module
5. Run tests to verify functionality
6. Commit if tests pass

### Testing Strategy
- **Unit tests** for each extracted module
- **Integration tests** for end-to-end functionality
- **Regression tests** to catch performance issues
- **API compatibility tests** to ensure no breaking changes

This plan ensures the controller remains maintainable while preserving all existing functionality.
