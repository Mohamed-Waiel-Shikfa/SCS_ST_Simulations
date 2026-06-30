#!/usr/bin/env python3
"""
Magnetic Clamp Simulator — Entry Point
=======================================
Simulates magnetic clamping forces in a symmetric 5-layer sandwich
based on the EPM circuit model from Marchese, Asada & Rus (ICRA 2012).

Usage:
    python main.py

Requirements:
    pip install customtkinter matplotlib numpy scipy magpylib
"""

import sys
import os

# Ensure the package directory is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ui_app import MagneticClampApp


def main():
    app = MagneticClampApp()
    app.mainloop()


if __name__ == "__main__":
    main()
