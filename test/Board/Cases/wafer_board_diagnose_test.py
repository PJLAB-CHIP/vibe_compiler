#!/usr/bin/env python3
"""Bounded file-backed register protocol tests; no hardware is opened."""
import argparse
import importlib.util
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock

p = argparse.ArgumentParser()
p.add_argument('--tool', type=Path, required=True)
p.add_argument('--library', type=Path, required=True)
config, remaining = p.parse_known_args()
spec = importlib.util.spec_from_file_location('board_diagnose', config.tool)
tool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tool)


class CaptureTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.resource = self.root / 'resource'
        self.bases = [t * 0x800000 for t in range(16)]
        with self.resource.open('wb') as f:
            f.truncate(16 * 0x800000)
            for t, base in enumerate(self.bases):
                xy = (t // 4 << 8) | t % 4
                f.seek(base + 0x6a0058)
                f.write(struct.pack('<II', xy, xy))
                f.seek(base + 0x590020)
                f.write(struct.pack('<I', 100 + t))
        self.output = self.root / 'capture.bin'
        self.ready = self.root / 'ready'
        self.stop = self.root / 'stop'

    def capture(self, capacity=64, maximum_ms=5, post_ms=1):
        reader = tool.Capture(config.library, self.resource, self.bases)
        try:
            reader.run(self.output, self.ready, self.stop, capacity, maximum_ms, post_ms)
        finally:
            reader.close()
        return tool.summarize(self.output)

    def test_nonzero_counts_wrap_and_deadline(self):
        result = self.capture()
        self.assertEqual(result['reason'], 'deadline')
        self.assertEqual(result['kept'], 64)
        self.assertGreater(result['total'], 64)
        self.assertIsNone(result['trigger_sequence'])
        self.assertEqual(set(result['tiles']), set(range(16)))
        for t, tile in result['tiles'].items():
            self.assertEqual(tile['first_count'], 100 + t)
            self.assertEqual(tile['tdma_count_delta_modulo_32'], 0)

    def test_external_stop(self):
        self.stop.touch()
        result = self.capture(capacity=1024)
        self.assertEqual((result['reason'], result['total']), ('external-stop', 256))

    def test_fatal_trigger_retains_window(self):
        failures = []
        def inject():
            deadline = time.monotonic() + 2
            while not self.ready.exists() and time.monotonic() < deadline:
                time.sleep(.001)
            if not self.ready.exists():
                failures.append('collector never ready')
                return
            with self.resource.open('r+b') as f:
                f.seek(0x6100c0)
                f.write(struct.pack('<I', 0x1000))
        writer = threading.Thread(target=inject)
        writer.start()
        try:
            result = self.capture(maximum_ms=1000)
        finally:
            writer.join(timeout=3)
        self.assertFalse(failures)
        self.assertEqual(result['reason'], 'post-trigger-capacity')
        trigger = result['trigger_sequence']
        self.assertIsNotNone(trigger)
        self.assertTrue(result['first_sequence'] <= trigger < result['total'])
        event = next(x for x in result['trigger_neighborhood'] if x['sequence'] == trigger)
        self.assertEqual(event['tile'], 0)
        self.assertEqual(event['fatal_before'] | event['fatal_after'], 0x1000)
        self.assertFalse(result['fault_instruction_identified'])

    def test_dirty_baseline_rejected(self):
        for value in (0x1000, 0xffffffff):
            with self.subTest(value=value):
                with self.resource.open('r+b') as f:
                    f.seek(0x6100c0)
                    f.write(struct.pack('<I', value))
                with self.assertRaisesRegex(RuntimeError, 'baseline'):
                    self.capture()
                self.assertFalse(self.ready.exists())
                self.assertFalse(self.output.exists())

    def test_wrong_identity_and_truncated_aperture(self):
        with self.resource.open('r+b') as f:
            f.seek(0x6a0058)
            f.write(struct.pack('<I', 123))
        with self.assertRaisesRegex(ValueError, 'identity'):
            tool.Capture(config.library, self.resource, self.bases)
        self.resource.write_bytes(b'1234')
        with self.assertRaises(ValueError):
            tool.Capture(config.library, self.resource, self.bases)

    def test_corrupt_capture_rejected(self):
        self.capture()
        original = self.output.read_bytes()
        for raw in (original[:-1], b'badmagic' + original[8:],
                    original[:64] + struct.pack('<Q', 123456789) + original[72:]):
            self.output.write_bytes(raw)
            with self.assertRaises(ValueError):
                tool.summarize(self.output)

    def test_atu_mapping(self):
        rows = [f'BAR4_NPU_TILE{i}: base[{i * 0x2000000:#x}] '
                f'target[{0x8000000000 + i * 0x80000000:#x}] size[0x2000000]'
                for i in range(4)]
        self.assertEqual(tool.tile_bases('\n'.join(rows)), self.bases)
        for bad in ('\n'.join(rows[:3]), '\n'.join(rows + rows[:1]),
                    '\n'.join(rows).replace('base[0x2000000]', 'base[0x0]')):
            with self.assertRaises(ValueError):
                tool.tile_bases(bad)

    def test_kernel_history_can_be_empty_before_new_fault_window(self):
        firmware = self.root / 'pci'
        firmware.mkdir()
        folder = self.root / 'logs'
        folder.mkdir()
        def journal(*arguments, kernel_only=True):
            if '--show-cursor' in arguments:
                return ('-- No entries --\n' if kernel_only else
                        'current userspace entry\n-- cursor: global-position\n')
            self.assertTrue(kernel_only)
            self.assertEqual(arguments, ('--after-cursor=global-position',))
            return 'NPU LSU TDMA Timeout\n'
        with mock.patch.object(tool, 'journal', side_effect=journal), \
             mock.patch.object(tool, 'Path', return_value=self.root):
            start = tool.log_start('pci')
            logs = tool.collect_logs(folder, start)
        self.assertTrue(tool.ALARM.search(logs))
        self.assertEqual((folder / 'kernel.log').read_text(), logs.rstrip() + '\n')

    def test_run_defaults_off_and_only_launches_original_command(self):
        args = tool.parser().parse_args(['run', '--output-dir', str(self.root / 'run'),
            '--device', '/dev/accel/dev-0', '--pci-bus-id', '0000:00:00.0', '--', 'original-runner'])
        self.assertFalse(args.capture_registers)
        with mock.patch.object(tool, 'boot_id', return_value='boot'), \
             mock.patch.object(tool, 'driver_identity', return_value='driver'), \
             mock.patch.object(tool, 'validate_device'), \
             mock.patch.object(tool, 'log_start'), \
             mock.patch.object(tool, 'collect_logs', return_value=''), \
             mock.patch.object(tool.time, 'sleep'), \
             mock.patch.object(tool.subprocess, 'check_output', return_value='{"state":"idle"}'), \
             mock.patch.object(tool.subprocess, 'Popen') as launch:
            launch.return_value.wait.return_value = 0
            launch.return_value.pid = 123
            self.assertEqual(tool.run_command(args), 0)
            self.assertEqual(launch.call_count, 1)
            self.assertEqual(launch.call_args.args[0], ['original-runner'])
        self.assertFalse((args.output_dir / 'capture.bin').exists())

    def test_runner_timeout_preserves_process_and_collects_logs(self):
        args = tool.parser().parse_args(['run', '--output-dir', str(self.root / 'timeout'),
            '--device', '/dev/accel/dev-0', '--pci-bus-id', '0000:00:00.0', '--', 'runner'])
        with mock.patch.object(tool, 'boot_id', return_value='boot'), \
             mock.patch.object(tool, 'driver_identity', return_value='driver'), \
             mock.patch.object(tool, 'validate_device'), \
             mock.patch.object(tool, 'log_start'), \
             mock.patch.object(tool, 'collect_logs', return_value='NPU LSU TDMA Timeout') as logs, \
             mock.patch.object(tool.time, 'sleep'), \
             mock.patch.object(tool.subprocess, 'check_output', return_value='{"state":"idle"}'), \
             mock.patch.object(tool.subprocess, 'Popen') as launch:
            launch.return_value.pid = 123
            launch.return_value.poll.return_value = None
            launch.return_value.wait.side_effect = subprocess.TimeoutExpired('runner', 120)
            self.assertEqual(tool.run_command(args), 1)
            launch.return_value.kill.assert_not_called()
            launch.return_value.terminate.assert_not_called()
            self.assertTrue(logs.called)
        import json
        report = json.loads((args.output_dir / 'run.json').read_text())
        self.assertFalse(report['passed'])
        self.assertTrue(report['runner_still_alive'])
        self.assertEqual(report['alarms'], ['NPU LSU TDMA Timeout'])

    def test_explicit_capture_requires_ready_before_launch(self):
        args = tool.parser().parse_args(['run', '--capture-registers',
            '--output-dir', str(self.root / 'run'), '--device', '/dev/accel/dev-0',
            '--pci-bus-id', '0000:00:00.0', '--', 'original-runner'])
        with mock.patch.object(tool, 'boot_id', return_value='boot'), \
             mock.patch.object(tool, 'driver_identity', return_value='driver'), \
             mock.patch.object(tool, 'validate_device'), \
             mock.patch.object(tool, 'log_start'), \
             mock.patch.object(tool.subprocess, 'check_output', return_value='{"state":"idle"}'), \
             mock.patch.object(tool.subprocess, 'Popen') as launch:
            launch.return_value.poll.return_value = 1
            self.assertEqual(tool.run_command(args), 1)
            self.assertEqual(launch.call_count, 1)
            self.assertIn('capture', launch.call_args.args[0])


if __name__ == '__main__':
    unittest.main(argv=[sys.argv[0], *remaining])
