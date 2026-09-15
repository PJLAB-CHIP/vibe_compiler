"""Bounded concurrency and publication oracle for the device-link driver."""

import argparse
import importlib.util
import pathlib
import subprocess
import sys
import tempfile
import threading
from unittest import mock

sys.dont_write_bytecode = True
spec = importlib.util.spec_from_file_location("device_link", sys.argv[1])
device_link = importlib.util.module_from_spec(spec)
spec.loader.exec_module(device_link)


def check_case(failures):
    with tempfile.TemporaryDirectory(prefix="wafer-device-link-parallel-") as temp:
        root = pathlib.Path(temp)
        output, llvm_object, crt_object = (
            root / "kernel.so", root / "kernel.o", root / "crt.o"
        )
        output.write_bytes(b"previous-output")
        args = argparse.Namespace(
            output=str(output), object_output=str(llvm_object),
            crt_object_output=str(crt_object), loader_abi="tx8-kcore-loader",
        )
        started = threading.Barrier(2, timeout=5)
        lock = threading.Lock()
        finished = set()
        normalized = set()
        linked = []

        def commands(staged):
            return (
                ["compile-llvm", staged.object_output],
                [["normalize-llvm", staged.object_output]],
                ["compile-crt", staged.crt_object_output],
                [["normalize-crt", staged.crt_object_output]],
                ["link", staged.output],
                ["scan", staged.output],
            )

        def run(command, **kwargs):
            kind, destination = command
            if kind.startswith("compile-"):
                # Both independent commands must actually start concurrently.
                # A serial implementation fails the bounded barrier oracle.
                started.wait()
                pathlib.Path(destination).write_bytes(kind.encode())
                with lock:
                    finished.add(kind)
            elif kind.startswith("normalize-"):
                with lock:
                    assert kind.replace("normalize-", "compile-") in finished
                    normalized.add(kind)
            elif kind == "link":
                assert not failures
                assert finished == {"compile-llvm", "compile-crt"}
                assert normalized == {"normalize-llvm", "normalize-crt"}
                pathlib.Path(destination).write_bytes(b"linked-output")
                linked.append(True)
            else:
                raise AssertionError(command)
            return subprocess.CompletedProcess(
                command, failures.get(kind, 0), stdout=b"", stderr=b""
            )

        with mock.patch.object(device_link, "build_commands", side_effect=commands), \
             mock.patch.object(device_link.subprocess, "run", side_effect=run), \
             mock.patch.object(device_link, "run_required_symbol_scan"), \
             mock.patch.object(device_link.os, "cpu_count", return_value=2):
            try:
                device_link.execute_staged_link(args)
                assert not failures
            except subprocess.CalledProcessError as error:
                priority = ("compile-llvm", "normalize-llvm",
                            "compile-crt", "normalize-crt")
                expected = next(kind for kind in priority if kind in failures)
                assert error.cmd[0] == expected, (error.cmd, expected)
        assert finished == {"compile-llvm", "compile-crt"}
        assert not list(root.glob(".kernel.so.staging-*"))
        if failures:
            assert output.read_bytes() == b"previous-output"
            assert not llvm_object.exists() and not crt_object.exists()
            assert not linked
        else:
            assert output.read_bytes() == b"linked-output"
            assert llvm_object.read_bytes() == b"compile-llvm"
            assert crt_object.read_bytes() == b"compile-crt"
            assert len(linked) == 1


for failures in ({}, {"compile-llvm": 1}, {"compile-crt": 2},
                 {"compile-llvm": 1, "compile-crt": 2},
                 {"normalize-llvm": 3}, {"normalize-crt": 4}):
    check_case(failures)
print("parallel object compilation and failure cleanup passed")
