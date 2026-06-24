#!/usr/bin/env python3
"""
tsmprof-cu Performance Analysis Tool

This script encapsulates tsmprof-cu, enabling users to perform analysis with various parameters.
Key features include First Starting RPU card ID, MLIR path, output directory, and eng_test paramters.

Examples:
    python profiling_tool.py -m './model.mlir' -o './results' -f 0 -x 1 -y 1 -d './chip_out/'
"""

import argparse
import subprocess
import os
import sys

def run_command(cmd):
    """Execute command and handle output"""
    print(f"Running command: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"Output: {result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error: {e.stderr}")
        print(f"Command failed: {' '.join(cmd)}")
        return False
    except Exception as e:
        print(f"Exception: {str(e)}")
        return False

def run_parse2timeline(csv_files):
    """Run parse2timeline command"""
    for csv_file in csv_files:
        if not os.path.exists(csv_file):
            print(f"CSV file not found: {csv_file}")
            return False

    cmd = ['python3', 'parse2timeline.py'] + csv_files
    print(f"Running timeline parser:{' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Parsing complete:{e.stderr}")
        return False
    except Exception as e:
        print(f"Parsing failed:{str(e)}")
        return False

def run_perf_analyzer(trace_file=None, ct_param_file=None, ne_param_file=None, output_file=None):


    cmd = [sys.executable, "perf_analyzer.py"]

    cmd.append(trace_file)
    cmd.append(ct_param_file)
    cmd.append(ne_param_file)
    cmd.append(output_file)

    print(f"执行命令: {' '.join(cmd)}")

    try:
        # 执行子进程并等待完成
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300  # 5分钟超时
        )
        return True
    except subprocess.TimeoutExpired:
        print("错误: 脚本执行超时")
        return False
    except FileNotFoundError:
        print("错误: perf_analyzer.py文件未找到")
        return False
    except Exception as e:
        print(f"执行出错: {e}")
        return False

def create_parser():
    """Create command-line argument parser"""
    parser = argparse.ArgumentParser(
        description="""
tsmprof-cu Performance Analysis Tool 

This tool executes tsmprof-cu's performance data collection and export operations.
It supports full parameter configuration for model performance analysis.
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s -m './model.mlir' -o './results' -f 0 -x 1 -y 1 -d './chip_out/'
        """
    )

    # Core parameters
    parser.add_argument('-f', '--first_card_id',
                       type=int,
                       default=0,
                       help='first_card_id, first starting RPU card ID')

    parser.add_argument('-m', '--mlir_path',
                       type=str,
                       help='mlir_path, MLIR file path')

    parser.add_argument('-o', '--output',
                       type=str,
                       required=True,
                       help='Output directory(required)')

    # eng_test parameters
    parser.add_argument('-c', '--cycle',
                       type=int,
                       default=1,
                       help='eng_test parameter, number of times to loop call run_kernel (default: 1)')

    parser.add_argument('-x', '--x_len',
                       type=int,
                       default=1,
                       help='eng_test parameter, model layout length in x direction (default: 1)')

    parser.add_argument('-y', '--y_len',
                       type=int,
                       default=1,
                       help='eng_test parameter, model layout length in y direction (default: 1)')

    parser.add_argument('-d', '--dir',
                       type=str,
                       required=True,
                       default='./chip_out/',
                       help='eng_test paramete, directory of codegen case(required) (default: ./chip_out/)')

    # parameter to skip collection step
    parser.add_argument('-s', '--skip_collect_step',
                       type=int,
                       choices=[0,1],
                       default=0,
                       help='Skip data collection step(1=skip, 0=execute)(default:0)')

    return parser

def validate_args(args):
    """Validate input arguments"""
    errors = []

    # Check parameter ranges
    if args.first_card_id < 0:
        errors.append("First Card ID cannot be negative")

    if args.cycle <= 0:
        errors.append("Parameter c must be greater than 0")

    if args.x_len <= 0:
        errors.append("Parameter x must be greater than 0")

    if args.y_len <= 0:
        errors.append("Parameter y must be greater than 0")

    return errors

def main():
    """Main function"""
    parser = create_parser()
    args = parser.parse_args()

    # Argument validation
    errors = validate_args(args)
    if errors:
        print("Argument validation failed:")
        for error in errors:
            print(f"  - {error}")
        sys.exit(1)

    # Ensure output directories exist
    try:
        os.makedirs(args.output, exist_ok=True)
    except Exception as e:
        print(f"Directory creation failed: {str(e)}")
        sys.exit(1)

    print("ProfilingCu Performance Analysis Tool")
    print("=" * 50)
    print(f"First Card ID: {args.first_card_id}")
    print(f"MLIR Path: {args.mlir_path}")
    print(f"Output Directory: {args.output}")
    print(f"eng_test Parameters: c={args.cycle}, x={args.x_len}, y={args.y_len}, d={args.dir}")
    print("=" * 50)

    # Execute data collection (default execution, skipped when --skip_collect_step=1)
    if args.skip_collect_step == 0:
        eng_test_cmd = [
            './eng_test',
            '-c', str(args.cycle),
            '-f', str(args.first_card_id),
            '-x', str(args.x_len),
            '-y', str(args.y_len),
            '-d', str(args.dir)
        ]

        collect_cmd = [
            './tsmprof-cu',
            '--command', ' '.join(eng_test_cmd),
            '--output', str(args.output),
            '--mlir_path', str(args.mlir_path),
            '--start_card_id', str(args.first_card_id)
        ]

        # Execute commands
        print("\nStarting collection task...")
        if not run_command(collect_cmd):
            print("Collection failed, terminating")
            sys.exit(1)

    # Build export command
    export_cmd = [
        './tsmprof-cu',
        '--export', 'on',
        '--output', str(args.output),
        '--mlir_path', str(args.mlir_path),
        '--start_card_id', str(args.first_card_id)
    ]

    print("\nStarting export task...")
    if not run_command(export_cmd):
        print("Export failed")
        sys.exit(1)

    # Post-export: Timeline parsing
    profiler_out_dir = os.path.join(args.output, "profilier_out")
    group_csv_file = os.path.join(profiler_out_dir, "group_time.csv")
    instr_csv_file = os.path.join(profiler_out_dir, "ne_dte_instr_time_trace.csv")
    score0_csv_file = os.path.join(profiler_out_dir, "ct_rdma_instr_time_trace.csv")
    score1_csv_file = os.path.join(profiler_out_dir, "wdma_tdma_instr_time_trace.csv")
    csv_files = [group_csv_file, instr_csv_file, score0_csv_file, score1_csv_file]
    print("Starting timeline parsing...")
    if run_parse2timeline(csv_files):
        json_file = os.path.join(profiler_out_dir, "instr_time_trace.timeline.json")
        print(f"Parsing complete, saved to: {json_file}")
    else:
        print("Timeline parsing failed")

    enable_prof = os.environ.get('ENABLE_PROF_INTIRSIC', '0')
    if enable_prof == '1':
        print("Starting perf param info...")
        ct_param_file = os.path.join(profiler_out_dir, "ct_param_info.csv")
        ne_param_file = os.path.join(profiler_out_dir, "ne_param_info.csv")
        output_file = os.path.join(profiler_out_dir, "instr_time_perf.csv")

        if run_perf_analyzer(instr_csv_file, ct_param_file, ne_param_file, output_file):
            print("Generate success")
        else:
            print("Generate failed")
    print("\nAll tasks completed!")

if __name__ == '__main__':
    main()
