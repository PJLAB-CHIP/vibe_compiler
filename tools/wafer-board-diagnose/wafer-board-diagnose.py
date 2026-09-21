#!/usr/bin/env python3
"""Run one board command with logs and optional read-only TX81 register capture."""
from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import struct
import subprocess
import sys
import time

sys.dont_write_bytecode = True
HEADER = struct.Struct('<8s7Q')
RECORD = struct.Struct('<3Q8I')
KEYS = ('sequence', 'begin_ns', 'end_ns', 'tile', 'fatal_before', 'count_before',
        'last_low', 'last_high', 'count_after', 'fatal_after', 'reserved')
REASONS = {1: 'external-stop', 2: 'deadline', 3: 'post-trigger-window',
           4: 'aperture-error', 5: 'post-trigger-capacity'}
REGISTERS = {'stream_fatal': 0x6100c0, 'tdma_count': 0x590020,
             'last_low': 0x59027c, 'last_high': 0x590280}
ALARM = re.compile(r'NPU.*Timeout|fatal_err|LSU.*timeout|\bxid=|AP.*clean.*timeout|'
                   r'failed to.*(close|cleanup)|err_code:-', re.I)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def boot_id():
    return Path('/proc/sys/kernel/random/boot_id').read_text().strip()


def driver_identity():
    return Path('/sys/module/tsingmicro/srcversion').read_text().strip()


def tile_bases(atu):
    entries = {}
    for index, base, target, size in re.findall(
        r'BAR4_NPU_TILE([0-3]): base\[(0x[0-9a-fA-F]+)\] '
        r'target\[(0x[0-9a-fA-F]+)\] size\[(0x[0-9a-fA-F]+)\]', atu
    ):
        i = int(index)
        values = tuple(int(v, 16) for v in (base, target, size))
        if i in entries:
            raise ValueError('duplicate ATU window')
        if values[0] % 4096 or values[1:] != (0x8000000000 + i * 0x80000000, 0x2000000):
            raise ValueError('unsupported TX81 ATU mapping')
        entries[i] = values
    if len(entries) != 4:
        raise ValueError('four ATU windows required')
    ranges = sorted((b, b + s) for b, _, s in entries.values())
    if any(a[1] > b[0] for a, b in zip(ranges, ranges[1:])):
        raise ValueError('overlapping ATU windows')
    return [entries[t // 4][0] + t % 4 * 0x800000 for t in range(16)]


def occupancy(device):
    """Check all host-visible process FDs, including other users and containers."""
    if os.geteuid() != 0:
        raise ValueError('root visibility is required for the system-wide occupancy check')
    identity = Path(device).stat()
    if not stat.S_ISCHR(identity.st_mode):
        raise ValueError('device must be a character device')
    holders, unknown = [], []
    for process in sorted(Path('/proc').glob('[0-9]*'), key=lambda p: int(p.name)):
        try:
            for descriptor in (process / 'fd').iterdir():
                try:
                    info = descriptor.stat()
                    if not stat.S_ISCHR(info.st_mode) or info.st_rdev != identity.st_rdev:
                        continue
                    exe = os.readlink(process / 'exe')
                    uid = int((process / 'status').read_text().split('Uid:', 1)[1].split()[0])
                    cgroup = (process / 'cgroup').read_text().strip()
                    log_service = (exe == '/usr/bin/npu_pull_ep_log' and uid == 0 and
                                   cgroup.endswith('/system.slice/npu_ep_log_svc.service'))
                    holders.append(dict(pid=int(process.name), fd=int(descriptor.name),
                                        exe=exe, uid=uid, cgroup=cgroup,
                                        verified_log_service=log_service))
                except OSError as error:
                    if error.errno not in (errno.ENOENT, errno.ESRCH, errno.EBADF):
                        unknown.append(dict(path=str(descriptor), error=str(error)))
        except OSError as error:
            if error.errno not in (errno.ENOENT, errno.ESRCH):
                unknown.append(dict(path=str(process), error=str(error)))
    fuser = subprocess.run(['fuser', '-v', str(device)], capture_output=True, text=True, timeout=10)
    if fuser.returncode not in (0, 1):
        unknown.append(dict(error='fuser failed', output=fuser.stderr))
    extra = {int(x) for x in fuser.stdout.split()} - {h['pid'] for h in holders}
    state = ('unknown' if unknown or extra else
             'busy' if any(not h['verified_log_service'] for h in holders) else 'idle')
    return dict(device=str(device), boot_id=boot_id(), state=state, holders=holders,
                unknown=unknown, unexplained_fuser_pids=sorted(extra),
                fuser_log=fuser.stdout + fuser.stderr)


class Capture:
    """One read-only mapping; also accepts regular files for protocol tests."""
    def __init__(self, library, resource, bases):
        if sys.byteorder != 'little' or len(bases) != 16:
            raise ValueError('little-endian host and 16 Tile bases required')
        self.lib = ctypes.CDLL(str(library))
        self.lib.fast_open.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_uint64), ctypes.c_char_p]
        self.lib.fast_open.restype = ctypes.c_void_p
        self.lib.fast_capture.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p,
                                         ctypes.c_char_p, ctypes.c_uint, ctypes.c_uint,
                                         ctypes.c_uint, ctypes.c_char_p]
        self.lib.fast_capture.restype = ctypes.c_int
        self.lib.fast_close.argtypes = [ctypes.c_void_p]
        error = ctypes.create_string_buffer(256)
        self.handle = self.lib.fast_open(os.fsencode(resource), (ctypes.c_uint64 * 16)(*bases), error)
        if not self.handle:
            raise ValueError(error.value.decode())

    def run(self, output, ready, stop, capacity=1048576, maximum_ms=30000, post_ms=100):
        if not 32 <= capacity <= 1048576 or not 1 <= maximum_ms <= 30000 or not 0 <= post_ms <= 500:
            raise ValueError('capture bounds: capacity 32..1048576, duration 1..30000 ms, post 0..500 ms')
        error = ctypes.create_string_buffer(256)
        rc = self.lib.fast_capture(self.handle, os.fsencode(output), os.fsencode(ready),
                                   os.fsencode(stop), capacity, maximum_ms, post_ms, error)
        if rc:
            raise RuntimeError(f'capture failed ({rc}): {error.value.decode()}')

    def close(self):
        if self.handle:
            self.lib.fast_close(self.handle)
            self.handle = None


def summarize(path):
    """Validate every record; retain only bounded summary and trigger neighborhood."""
    path = Path(path)
    with path.open('rb') as f:
        raw = f.read(HEADER.size)
        if len(raw) != HEADER.size:
            raise ValueError('truncated header')
        magic, start, end, total, kept, trigger, trigger_ns, reason = HEADER.unpack(raw)
        seen = trigger != (1 << 64) - 1
        if (magic != b'TDMARING' or not 0 < kept <= min(total, 1048576) or
                reason not in REASONS or end < start or
                path.stat().st_size != HEADER.size + kept * RECORD.size):
            raise ValueError('invalid header or file length')
        if seen and not total - kept <= trigger < total:
            raise ValueError('trigger missing from retained records')
        tiles, neighborhood = {}, []
        duration_min, duration_max, gap_min, gap_max = None, 0, None, 0
        previous = start
        for seq in range(total - kept, total):
            x = dict(zip(KEYS, RECORD.unpack(f.read(RECORD.size))))
            if (x['sequence'] != seq or x['tile'] != seq % 16 or x['reserved'] or
                    not previous <= x['begin_ns'] <= x['end_ns'] <= end):
                raise ValueError('invalid ordered record')
            previous = x['end_ns']
            duration = x['end_ns'] - x['begin_ns']
            duration_min = duration if duration_min is None else min(duration_min, duration)
            duration_max = max(duration_max, duration)
            tile = tiles.get(x['tile'])
            if tile is None:
                tile = dict(samples=0, first_count=x['count_before'])
                tiles[x['tile']] = tile
            else:
                gap = x['begin_ns'] - tile['last_begin_ns']
                gap_min = gap if gap_min is None else min(gap_min, gap)
                gap_max = max(gap_max, gap)
            tile.update(samples=tile['samples'] + 1, last_begin_ns=x['begin_ns'],
                        last_count=x['count_after'])
            if seen and trigger - 64 <= seq <= trigger + 64:
                neighborhood.append(x)
            if seen and seq == trigger and (x['end_ns'] != trigger_ns or
                    not (x['fatal_before'] | x['fatal_after']) & 0x1000):
                raise ValueError('invalid trigger record')
        for tile in tiles.values():
            tile['tdma_count_delta_modulo_32'] = (tile['last_count'] - tile['first_count']) & 0xffffffff
        return dict(start_ns=start, end_ns=end, total=total, kept=kept,
                    first_sequence=total - kept, trigger_sequence=trigger if seen else None,
                    trigger_ns=trigger_ns if seen else None, reason=REASONS[reason],
                    record_bytes=RECORD.size, register_loads_per_tile_sample=6,
                    tile_read_ns_min=duration_min, tile_read_ns_max=duration_max,
                    same_tile_gap_ns_min=gap_min, same_tile_gap_ns_max=gap_max,
                    tiles=tiles, trigger_neighborhood=neighborhood,
                    atomic_snapshot=False, fault_instruction_identified=False)


def self_command(*arguments, privileged=False):
    prefix = ['sudo', '-n'] if privileged and os.geteuid() != 0 else []
    return prefix + [sys.executable, '-B', str(Path(__file__).resolve()), *map(str, arguments)]


def validate_device(device, pci):
    if not re.fullmatch(r'/dev/accel/dev-\d+', device):
        raise ValueError('expected /dev/accel/dev-N device')
    if not re.fullmatch(r'[0-9a-fA-F]{4}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-7]', pci):
        raise ValueError('invalid PCI bus ID')
    # This vendor exposes virtual misc nodes without a PCI parent symlink.
    # The caller supplies the qualified pair; wafer-run checks it via runtime.
    ident = Path(device).stat()
    if not stat.S_ISCHR(ident.st_mode):
        raise ValueError('device must be a character device')
    if not (Path('/sys/bus/pci/devices') / pci / 'resource4').is_file():
        raise ValueError('selected PCI function has no BAR4 resource')


def live_capture(args):
    if os.geteuid() != 0 or boot_id() != args.expected_boot or driver_identity() != args.expected_driver:
        raise ValueError('root and matching boot/driver identity required')
    validate_device(args.device, args.pci_bus_id)
    state = occupancy(args.device)
    if state['state'] != 'idle':
        raise ValueError('device occupancy is ' + state['state'])
    atu = (Path('/sys/kernel/debug/accel') / Path(args.device).name / 'atu_entries').read_text()
    resource = Path('/sys/bus/pci/devices') / args.pci_bus_id / 'resource4'
    bases = tile_bases(atu)
    for p in (args.output, args.ready_file, args.stop_file):
        if p.exists():
            raise ValueError('capture destination/control file already exists: ' + str(p))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    metadata = dict(boot_id=boot_id(), driver=driver_identity(), device=args.device,
                    pci_bus_id=args.pci_bus_id, atu=atu, occupancy=state, registers=REGISTERS,
                    trigger_mask=0x1000, capacity=args.capacity, maximum_ms=args.maximum_ms,
                    post_ms=args.post_ms, host_register_capture=True,
                    sha256={str(p.name): hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in (Path(__file__), args.library)})
    write_json(str(args.output) + '.identity.json', metadata)
    reader = Capture(args.library, resource, bases)
    try:
        reader.run(args.output, args.ready_file, args.stop_file,
                   args.capacity, args.maximum_ms, args.post_ms)
    finally:
        reader.close()
    report = summarize(args.output)
    write_json(str(args.output) + '.summary.json', report)
    return report


def journal(*arguments, kernel_only=True):
    prefix = ['sudo', '-n'] if os.geteuid() != 0 else []
    filters = ['-b', '-k'] if kernel_only else ['-b']
    return subprocess.check_output(prefix + ['journalctl', *filters, '--no-pager',
                                   '--output=short-iso', *arguments], text=True, timeout=10)


def log_start(pci):
    # A journal cursor identifies a global position, independent of filters.
    # Old kernel entries may have rotated away while the journal is healthy.
    text = journal('-n', '1', '--show-cursor', kernel_only=False)
    match = re.search(r'^-- cursor: (.+)$', text, re.M)
    if not match:
        raise ValueError('journal cursor unavailable')
    directory = Path('/var/npu_ep_log') / pci
    if not directory.is_dir():
        raise ValueError('firmware log directory unavailable')
    offsets = {}
    for p in directory.rglob('*'):
        if p.is_file():
            s = p.stat()
            offsets[(s.st_dev, s.st_ino)] = s.st_size
    return match[1], directory, offsets


def collect_logs(folder, start):
    cursor, directory, offsets = start
    kernel = journal('--after-cursor=' + cursor)
    (folder / 'kernel.log').write_text(kernel)
    chunks = []
    for p in sorted(directory.rglob('*')):
        if p.is_file():
            s = p.stat()
            offset = offsets.get((s.st_dev, s.st_ino), 0)
            with p.open('rb') as f:
                f.seek(offset if s.st_size >= offset else 0)
                data = f.read()
            if data:
                chunks.append(str(p).encode() + b'\n' + data)
    firmware = b'\n'.join(chunks)
    (folder / 'firmware.log').write_bytes(firmware)
    return kernel + '\n' + firmware.decode(errors='replace')


def run_command(args):
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command:
        raise ValueError('run requires a command after --')
    folder = args.output_dir
    folder.mkdir(parents=True, exist_ok=False)
    report = dict(command=command, host_register_capture=args.capture_registers,
                  device=args.device, pci_bus_id=args.pci_bus_id,
                  boot_id=boot_id(), driver=driver_identity(), problems=[])
    write_json(folder / 'run.json', report)
    observer = None
    observer_log = None
    start = None
    launched = False
    stop = folder / 'stop'
    try:
        if args.expected_boot and report['boot_id'] != args.expected_boot:
            raise ValueError('boot identity changed')
        validate_device(args.device, args.pci_bus_id)
        state = json.loads(subprocess.check_output(
            self_command('occupancy', '--device', args.device, privileged=True), text=True, timeout=15))
        write_json(folder / 'occupancy.json', state)
        if state['state'] != 'idle':
            raise ValueError('device occupancy is ' + state['state'])
        start = log_start(args.pci_bus_id)
        if args.capture_registers:
            ready = folder / 'ready'
            observer_log = (folder / 'observer.log').open('x')
            observer = subprocess.Popen(self_command(
                'capture', '--device', args.device, '--pci-bus-id', args.pci_bus_id,
                '--expected-boot', report['boot_id'], '--expected-driver', report['driver'],
                '--output', folder / 'capture.bin', '--ready-file', ready, '--stop-file', stop,
                privileged=True), stdout=observer_log, stderr=subprocess.STDOUT)
            deadline = time.monotonic() + 10
            while not ready.exists() and observer.poll() is None and time.monotonic() < deadline:
                time.sleep(.01)
            if not ready.exists() or observer.poll() is not None:
                raise RuntimeError('register collector did not become ready; inspect observer.log')
        if boot_id() != report['boot_id']:
            raise ValueError('boot changed before launch')
        with (folder / 'runner.log').open('x') as log:
            child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            launched = True
            report['pid'] = child.pid
            try:
                report['returncode'] = child.wait(timeout=args.timeout_seconds)
            except subprocess.TimeoutExpired:
                report['runner_still_alive'] = child.poll() is None
                raise RuntimeError('runner deadline; preserving vendor cleanup, no signal or reset')
        if observer:
            stop.write_text('runner completed\n')
            report['observer_returncode'] = observer.wait(timeout=5)
            summary = json.loads((folder / 'capture.bin.summary.json').read_text())
            report['capture'] = summary
            if (report['observer_returncode'] or summary['trigger_sequence'] is not None or
                    summary['reason'] != 'external-stop'):
                report['problems'].append('register capture reported fatal or incomplete coverage')
        if report['returncode']:
            report['problems'].append('runner failed')
        print((folder / 'runner.log').read_text(errors='replace'), end='')
    except KeyboardInterrupt:
        report['problems'].append('controller interrupted; no automatic retry or reset')
        raise
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        report['problems'].append(str(error))
    finally:
        report['launched'] = launched
        if observer and observer.poll() is None:
            stop.write_text('controller finished\n')
            try:
                observer.wait(timeout=5)
            except subprocess.TimeoutExpired:
                report['problems'].append('collector did not stop')
        if observer_log:
            observer_log.close()
        if launched:
            try:
                # Preserve failure logs too; vendor log flushing is outside device timing.
                for _ in range(10):
                    time.sleep(.5)
                    combined = collect_logs(folder, start)
                report['alarms'] = [line for line in combined.splitlines() if ALARM.search(line)]
                if report['alarms']:
                    report['problems'].append('driver/firmware alarm')
            except (OSError, ValueError, subprocess.SubprocessError) as error:
                report['problems'].append('log collection failed: ' + str(error))
        report['passed'] = not report['problems']
        write_json(folder / 'run.json', report)
    if report['problems']:
        print(json.dumps(report['problems']), file=sys.stderr)
    return 0 if report['passed'] else 1


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='action', required=True)
    run = sub.add_parser('run', help='execute once; register capture defaults to OFF')
    run.add_argument('--capture-registers', action='store_true', default=False)
    run.add_argument('--output-dir', type=Path, required=True)
    run.add_argument('--device', required=True)
    run.add_argument('--pci-bus-id', required=True)
    run.add_argument('--expected-boot')
    run.add_argument('--timeout-seconds', type=float, default=120)
    run.add_argument('command', nargs=argparse.REMAINDER)
    occ = sub.add_parser('occupancy', help='system-wide process check; requires root')
    occ.add_argument('--device', required=True)
    capture = sub.add_parser('capture', help='explicit live read-only capture; requires root')
    capture.add_argument('--device', required=True)
    capture.add_argument('--pci-bus-id', required=True)
    capture.add_argument('--expected-boot', required=True)
    capture.add_argument('--expected-driver', required=True)
    capture.add_argument('--output', type=Path, required=True)
    capture.add_argument('--ready-file', type=Path, required=True)
    capture.add_argument('--stop-file', type=Path, required=True)
    capture.add_argument('--library', type=Path, default=Path(__file__).resolve().parent.parent /
                         'share/wafer/board-diagnose/libwafer_board_diagnose.so')
    capture.add_argument('--capacity', type=int, default=1048576)
    capture.add_argument('--maximum-ms', type=int, default=30000)
    capture.add_argument('--post-ms', type=int, default=100)
    decode = sub.add_parser('decode', help='validate and summarize a capture without device access')
    decode.add_argument('input', type=Path)
    decode.add_argument('--output', type=Path)
    return p


def main():
    p = parser()
    args = p.parse_args()
    try:
        if args.action == 'run':
            if args.timeout_seconds <= 0:
                raise ValueError('timeout must be positive')
            return run_command(args)
        if args.action == 'occupancy':
            result = occupancy(args.device)
        elif args.action == 'capture':
            result = live_capture(args)
        else:
            result = summarize(args.input)
            if args.output:
                write_json(args.output, result)
        print(json.dumps(result, ensure_ascii=False))
        return 0
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        p.exit(1, f'wafer-board-diagnose: {error}\n')


if __name__ == '__main__':
    raise SystemExit(main())
