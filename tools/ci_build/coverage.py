#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
#
# This script generates test code coverage for Android.
# The prerequistes:
#     1. The Onnxruntime build with coverage option to compile/link the source files using --coverage optoin
#     2. The tests are run on the target emulator and *.gcda files are available on the emulator
#

import os
import sys
import argparse
from build import run_subprocess, adb_shell

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
REPO_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "..", ".."))

sys.path.append(os.path.join(REPO_DIR, "tools", "python"))


def adb_pull(src, dest, **kwargs):
    return run_subprocess(['adb', 'pull', src, dest], **kwargs)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--build_dir", required=True, help="Path to the build directory.")
    parser.add_argument(
        "--config", default="Debug",
        choices=["Debug", "MinSizeRel", "Release", "RelWithDebInfo"],
        help="Configuration(s) to run code coverage.")
    return parser.parse_args()


def start_android_emulator(args, source_dir):
    run_subprocess([os.path.join(
        source_dir, 'tools', 'ci_build', 'github', 'android',
        'start_android_emulator.sh')])


def main():
    args = parse_arguments()
    script_dir = os.path.realpath(os.path.dirname(__file__))
    source_dir = os.path.normpath(os.path.join(script_dir, "..", ".."))
    cwd = os.path.abspath(os.path.join(args.build_dir, args.config))
    start_android_emulator(args, source_dir)
    adb_shell('ls -ltr /data/local/tmp')
    adb_shell('find /data/local/tmp -name \"*.dir\"')
    adb_shell('cd /data/local/tmp && tar -zcvf gcda_files.tar.gz *.dir')
    adb_pull('/data/local/tmp/gcda_files.tar.gz', cwd)
    run_subprocess("tar -zxvf gcda_files.tar.gz -C CMakeFiles".split(' '))
    run_subprocess("gcovr -s -r {} .".format(os.path.join(source_dir, "onnxruntime")).split(' '),
                   cwd=os.path.join(cwd, "CMakeFiles"))


if __name__ == "__main__":
    main()
