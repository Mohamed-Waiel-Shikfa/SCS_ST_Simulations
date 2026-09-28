# Magnet simulation package

This folder can be moved as a unit. No files outside this folder are needed,
apart from Python and the three Python libraries listed in `requirements.txt`.

## Run on Windows

1. Extract the ZIP.
2. Install Python 3.10 or newer if it is not already installed.
3. From a terminal in this folder, install the libraries once:

   ```powershell
   python -m pip install -r .\requirements.txt
   ```

4. Double-click `RUN_SIMULATOR.cmd`, or run:

   ```powershell
   python .\magnet_pulse_sim.py
   ```

The program opens exactly two interactive windows:

- A z-center magnetic-field heatmap with isolines, time slider, field selector,
  hover readout, and peak-field button.
- Electrical and magnetic time series, with a checkbox for each quantity.

Closing both plot windows ends the program.

## Contents

- `magnet_pulse_sim.py`: the complete self-contained simulator. All physical
  formulas, parameter provenance, assumptions, and numerical checks are inside.
- `requirements.txt`: Python dependencies.
- `RUN_SIMULATOR.cmd`: Windows launcher; uses paths relative to this folder.
- `results\`: reference console results, time-series CSV, and two static plot
  previews. The previews are not substitutes for the interactive windows.
- `references\`: the three supplied component/material datasheets.
- `references\original_scripts\`: both supplied analytical simulators, preserved
  unchanged for comparison. They are not dependencies of the new simulator.

To run the embedded numerical and widget checks without opening windows:

```powershell
python .\magnet_pulse_sim.py --check
```

To calculate without opening windows:

```powershell
python .\magnet_pulse_sim.py --no-gui
```

The CSV field statistics describe the same cell-centered z=0 slice as the
heatmap. They are not full-volume extrema or fractions of magnet material that
has reversed.

## Model limits

This is a conditional linear-recoil reference, not a verified magnetizer design.
The capacitor's 45 mOhm value is typical at 360 Hz and 60 C. The SCR's
5.2 mOhm slope resistance belongs to its 150 C package power-loss model.
The freewheel diode is idealized because no diode part was specified.

The supplied magnet datasheet provides N52 grade limits but no N52 hysteresis
curve. Its illustrated curve is for N42. The simulation therefore does not
invent a magnetization-reversal trajectory or claim a percentage flipped.
Capacitor surge suitability, SCR pulse safe operating area, insulation,
mechanical restraint, and repetition rate still require physical validation.

The ZIP does not bundle a Python interpreter or a virtual environment. Such
environments are not reliably relocatable; install the listed dependencies
on another computer before running there.
