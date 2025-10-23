#!/usr/bin/env python3
"""
CLI - Command Line Interface for Computable Flows Shim. 

Provides commands for running optimizations, managing telemetry, and monitoring flows.
"""
import argparse
import asyncio
import sys
import os
import importlib.util
from pathlib import Path
from typing import Dict, Any, Optional
import jax.numpy as jnp

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from computable_flows_shim.energy.specs import EnergySpec, StateSpec
from computable_flows_shim.energy.compile import compile_energy
from computable_flows_shim.controller import FlightController
from computable_flows_shim.telemetry import TelemetryManager
from computable_flows_shim.telemetry.telemetry_streamer import start_streamer
from computable_flows_shim.api import Op
from computable_flows_shim.tuner.gap_dial import GapDialTuner, create_gap_dial_tuner
from computable_flows_shim import configure_logging, get_logger

def load_spec_from_file(spec_file: str) -> Dict[str, Any]:
    """
    Load a flow specification from a Python file.
    
    The file should define:
    - spec: EnergySpec
    - op_registry: Dict[str, Op]
    - initial_state: Dict[str, jnp.ndarray] 
    - Optional: step_alpha, num_iterations, flow_name
    """
    # If spec_file doesn't exist as given, try looking in flows directory
    if not os.path.exists(spec_file):
        flows_path = os.path.join("src", "telematry_cfs", "flows", spec_file)
        if os.path.exists(flows_path):
            spec_file = flows_path
        elif not spec_file.endswith('.py'):
            flows_path_py = os.path.join("src", "telematry_cfs", "flows", f"{spec_file}.py")
            if os.path.exists(flows_path_py):
                spec_file = flows_path_py
    
    if not os.path.exists(spec_file):
        raise FileNotFoundError(f"Spec file not found: {spec_file}")
    
    # Load the module
    spec_name = Path(spec_file).stem
    spec = importlib.util.spec_from_file_location(spec_name, spec_file)
    if spec is None or spec.loader is None:
        raise ValueError(f"Could not load spec from {spec_file}")
    
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    # Extract required objects
    required_attrs = ['spec', 'op_registry', 'initial_state']
    result = {}
    
    for attr in required_attrs:
        if not hasattr(module, attr):
            raise ValueError(f"Spec file must define '{attr}'")
        result[attr] = getattr(module, attr)
    
    # Optional attributes with defaults
    result['step_alpha'] = getattr(module, 'step_alpha', 0.1)
    result['num_iterations'] = getattr(module, 'num_iterations', 100)
    result['flow_name'] = getattr(module, 'flow_name', spec_name)
    
    return result

def run_flow(spec_file: str, output_dir: Optional[str] = None, telemetry: bool = True, gap_dial: bool = False) -> None:
    """
    Run a certified flow from a spec file.
    """
    logger = get_logger(__name__)
    logger.info("Starting flow execution", extra={
        'spec_file': spec_file,
        'telemetry_enabled': telemetry,
        'gap_dial_enabled': gap_dial
    })
    
    print(f"Loading spec from: {spec_file}")
    
    # Load the specification
    try:
        flow_data = load_spec_from_file(spec_file)
        spec = flow_data['spec']
        op_registry = flow_data['op_registry']
        initial_state = flow_data['initial_state']
        step_alpha = flow_data['step_alpha']
        num_iterations = flow_data['num_iterations']
        flow_name = flow_data['flow_name']
        
        logger.debug("Specification loaded successfully", extra={
            'flow_name': flow_name,
            'num_terms': len(spec.terms),
            'state_shapes': {k: v.shape for k, v in initial_state.items()},
            'step_alpha': step_alpha,
            'num_iterations': num_iterations
        })
        
    except Exception as e:
        logger.error("Failed to load specification", extra={
            'spec_file': spec_file,
            'error': str(e)
        })
        print(f"Error loading spec: {e}")
        return
    
    print(f"Flow: {flow_name}")
    print(f"Spec: {len(spec.terms)} terms")
    print(f"Initial state shape: {initial_state['x'].shape}")
    print(f"Parameters: alpha={step_alpha}, iterations={num_iterations}")
    
    # Set up output directory
    if output_dir is None:
        output_dir = "src/telematry_cfs"
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Set up telemetry if requested
    telemetry_manager = None
    if telemetry:
        telemetry_manager = TelemetryManager(
            base_path=output_dir,
            flow_name=flow_name
        )
        print("Telemetry enabled")
    
    # Set up Gap Dial tuner if requested
    gap_dial_tuner = None
    if gap_dial:
        gap_dial_tuner = create_gap_dial_tuner(target_gap=0.1, adaptation_rate=0.1)
        print("Gap Dial auto-tuner enabled")
    
    try:
        # Compile the energy functional
        print("Compiling energy functional...")
        logger.debug("Starting energy compilation")
        compiled = compile_energy(spec, op_registry)
        logger.info("Energy compilation completed successfully", extra={
            'has_compile_report': compiled.compile_report is not None
        })
        print("Compilation successful")
        
        # Write manifest
        if telemetry_manager:
            telemetry_manager.write_run_manifest(
                schema_version=1,
                residual_details={"compiled": True},
                extra={
                    "spec_file": spec_file,
                    "step_alpha": step_alpha,
                    "num_iterations": num_iterations
                }
            )
        
        # Run the certified flow
        print("Starting certified flow execution...")
        logger.info("Starting certified flow execution", extra={
            'num_iterations': num_iterations,
            'initial_alpha': step_alpha,
            'has_telemetry': telemetry_manager is not None,
            'has_gap_dial': gap_dial_tuner is not None
        })
        
        controller = FlightController()
        final_state = controller.run_certified_flow(
            initial_state=initial_state,
            compiled=compiled,
            num_iterations=num_iterations,
            initial_alpha=step_alpha,
            telemetry_manager=telemetry_manager,
            flow_name=flow_name,
            run_id=telemetry_manager.run_id if telemetry_manager else "cli_run",
            gap_dial_tuner=gap_dial_tuner
        )
        
        final_energy = compiled.f_value(final_state)
        logger.info("Flow execution completed successfully", extra={
            'final_energy': float(final_energy),
            'run_id': telemetry_manager.run_id if telemetry_manager else "cli_run"
        })
        
        print("Flow execution completed successfully")
        print(f"Final energy: {final_energy:.6f}")
        
        # Save final state
        if telemetry_manager:
            telemetry_manager.flush()
            logger.debug("Telemetry data flushed", extra={
                'run_path': telemetry_manager.run_path
            })
            print(f"Results saved to: {telemetry_manager.run_path}")
        
    except Exception as e:
        logger.error("Flow execution failed", extra={
            'error_type': type(e).__name__,
            'error_message': str(e),
            'run_id': telemetry_manager.run_id if telemetry_manager else "unknown"
        })
        print(f"Flow execution failed: {e}")
        if telemetry_manager:
            telemetry_manager.flight_recorder.log_event(
                run_id=telemetry_manager.run_id,
                event="RUN_FAILED",
                payload={"error": str(e)}
            )
            telemetry_manager.flush()
        raise

def cert_flow(spec_file: str) -> bool:
    """
    Check certificates for a flow specification.
    """
    print(f"Loading spec from: {spec_file}")
    
    try:
        flow_data = load_spec_from_file(spec_file)
        spec = flow_data['spec']
        op_registry = flow_data['op_registry']
        initial_state = flow_data['initial_state']
        step_alpha = flow_data['step_alpha']
        flow_name = flow_data['flow_name']
    except Exception as e:
        print(f"Error loading spec: {e}")
        return False
    
    print(f"Flow: {flow_name}")
    
    try:
        # Compile the energy functional
        print("Compiling energy functional...")
        compiled = compile_energy(spec, op_registry)
        print("Compilation successful")
        
        # Check certificates
        import jax
        from computable_flows_shim.fda.certificates import estimate_gamma, estimate_eta_dd
        
        key = jax.random.PRNGKey(42)
        input_shape = initial_state['x'].shape
        
        print("Checking certificates...")
        eta = estimate_eta_dd(compiled.L_apply, input_shape)
        gamma = estimate_gamma(compiled.L_apply, key, input_shape)
        
        print(f"Diagonal dominance (η_dd): {eta:.6f}")
        print(f"Spectral gap (γ): {gamma:.6f}")
        
        # Certificate assessment
        if eta < 1.0 and gamma > 0:
            print("✅ CERTIFICATES PASS - Flow is GREEN")
            return True
        else:
            print("❌ CERTIFICATES FAIL - Flow is RED/AMBER")
            if eta >= 1.0:
                print(f"   Issue: η_dd = {eta:.6f} >= 1.0 (too high)")
            if gamma <= 0:
                print(f"   Issue: γ = {gamma:.6f} <= 0 (no gap)")
            return False
            
    except Exception as e:
        print(f"Certificate check failed: {e}")
        return False

def tune_flow(spec_file: str, output_dir: Optional[str] = None, target_gap: float = 0.1, 
              adaptation_rate: float = 0.1, num_iterations: int = 50) -> None:
    """
    Run Gap Dial parameter tuning on a flow specification.
    """
    print(f"Loading spec from: {spec_file}")
    
    try:
        flow_data = load_spec_from_file(spec_file)
        spec = flow_data['spec']
        op_registry = flow_data['op_registry']
        initial_state = flow_data['initial_state']
        step_alpha = flow_data['step_alpha']
        flow_name = flow_data['flow_name']
    except Exception as e:
        print(f"Error loading spec: {e}")
        return
    
    print(f"Flow: {flow_name}")
    print(f"Gap Dial tuning: target_gap={target_gap}, adaptation_rate={adaptation_rate}")
    
    # Set up output directory
    if output_dir is None:
        output_dir = "src/telematry_cfs"
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Set up telemetry for tuning
    telemetry_manager = TelemetryManager(
        base_path=output_dir,
        flow_name=f"{flow_name}_tune"
    )
    print("Telemetry enabled for tuning")
    
    # Create Gap Dial tuner with specified parameters
    gap_dial_tuner = create_gap_dial_tuner(
        target_gap=target_gap,
        adaptation_rate=adaptation_rate
    )
    print("Gap Dial tuner initialized")
    
    try:
        # Compile the energy functional
        print("Compiling energy functional...")
        compiled = compile_energy(spec, op_registry)
        print("Compilation successful")
        
        # Write manifest
        telemetry_manager.write_run_manifest(
            schema_version=1,
            residual_details={"tuning": True},
            extra={
                "spec_file": spec_file,
                "tuning_mode": "gap_dial",
                "target_gap": target_gap,
                "adaptation_rate": adaptation_rate
            }
        )
        
        # Run tuning with Gap Dial
        print("Starting Gap Dial parameter tuning...")
        controller = FlightController()
        final_state = controller.run_certified_flow(
            initial_state=initial_state,
            compiled=compiled,
            num_iterations=num_iterations,
            initial_alpha=step_alpha,
            telemetry_manager=telemetry_manager,
            flow_name=flow_name,
            run_id=telemetry_manager.run_id,
            gap_dial_tuner=gap_dial_tuner
        )
        
        print("Gap Dial tuning completed successfully")
        print(f"Final energy: {compiled.f_value(final_state):.6f}")
        
        # Log final tuning status
        tuning_status = gap_dial_tuner.get_tuning_status()
        print(f"Final lambda_regularization: {tuning_status['current_lambda']:.6f}")
        print(f"Gap measurements: {len(tuning_status['gap_history'])}")
        if tuning_status['gap_history']:
            print(f"Final gap: {tuning_status['gap_history'][-1]:.6f}")
        
        # Save results
        telemetry_manager.flush()
        print(f"Tuning results saved to: {telemetry_manager.run_path}")
        
    except Exception as e:
        print(f"Tuning failed: {e}")
        telemetry_manager.flight_recorder.log_event(
            run_id=telemetry_manager.run_id,
            event="TUNING_FAILED",
            payload={"error": str(e)}
        )
        telemetry_manager.flush()
        raise

def main():
    parser = argparse.ArgumentParser(description="CFS - Computable Flow Shim CLI")
    
    # Global logging options
    parser.add_argument('--log-level', 
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                       default='WARNING',
                       help='Set logging level (default: WARNING)')
    parser.add_argument('--log-format',
                       choices=['json', 'text'],
                       default='json',
                       help='Set log output format (default: json)')
    parser.add_argument('--log-output',
                       choices=['stderr', 'stdout', 'file'],
                       default='stderr',
                       help='Set log output destination (default: stderr)')
    parser.add_argument('--log-file',
                       help='Log file path (optional when --log-output=file, defaults to logs/afs_YYYYMMDD_HHMMSS.log)')
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # cf-run command
    run_parser = subparsers.add_parser('run', help='Run an optimization flow')
    run_parser.add_argument('spec_file', help='Python file containing the flow specification')
    run_parser.add_argument('--output', '-o', help='Output directory for results')
    run_parser.add_argument('--no-telemetry', action='store_true', help='Disable telemetry logging')
    run_parser.add_argument('--gap-dial', action='store_true', help='Enable Gap Dial auto-tuner for in-run parameter optimization')

    # cf-cert command
    cert_parser = subparsers.add_parser('cert', help='Check flow certificates')
    cert_parser.add_argument('spec_file', help='Python file containing the flow specification')

    # cf-tune command
    tune_parser = subparsers.add_parser('tune', help='Run Gap Dial parameter optimization')
    tune_parser.add_argument('spec_file', help='Python file containing the flow specification')
    tune_parser.add_argument('--output', '-o', help='Output directory for results')
    tune_parser.add_argument('--target-gap', type=float, default=0.1, help='Target spectral gap (default: 0.1)')
    tune_parser.add_argument('--adaptation-rate', type=float, default=0.1, help='Parameter adaptation rate (default: 0.1)')
    tune_parser.add_argument('--iterations', type=int, default=50, help='Number of tuning iterations (default: 50)')

    # Parse arguments
    args = parser.parse_args()

    # Configure logging
    configure_logging(
        level=args.log_level,
        format=args.log_format,
        output=args.log_output,
        log_file=getattr(args, 'log_file', None)
    )

    if not args.command:
        parser.print_help()
        return

    if args.command == 'hud':
        print(f"Starting CFS Telemetry Dashboard on {args.host}:{args.port}")
        print("Open http://localhost:8000/sparsity_hud_demo.html in your browser")
        print("Press Ctrl+C to stop")

        try:
            # Start the telemetry streamer
            asyncio.run(start_streamer(host=args.host, port=args.port))
        except KeyboardInterrupt:
            print("\nTelemetry server stopped")

    if args.command == 'run':
        telemetry_enabled = not args.no_telemetry
        gap_dial_enabled = getattr(args, 'gap_dial', False)
        run_flow(args.spec_file, args.output, telemetry_enabled, gap_dial_enabled)

    elif args.command == 'cert':
        success = cert_flow(args.spec_file)
        sys.exit(0 if success else 1)

    elif args.command == 'tune':
        tune_flow(args.spec_file, args.output, args.target_gap, args.adaptation_rate, args.iterations)

    else:
        parser.print_help()

if __name__ == '__main__':
    main()