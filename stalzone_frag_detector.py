#!/usr/bin/env python3
"""Compatibility entry point for running the detector as a script."""

from szfd.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
