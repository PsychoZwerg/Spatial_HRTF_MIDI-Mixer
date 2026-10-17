# Spatial MIDI Mixer

A real-time **binaural mixer** that places two musical sources anywhere in 3D around the
listener. Each source is rendered through measured head-related transfer functions (HRTFs);
a hardware head-tracker updates the auditory image live as you turn your head; and the whole
thing can be controlled not just via a GUI but also a MIDI controller with faders, buttons and LED feedback.

Built from scratch in Python as a project for the M.Sc. *Audio Communication and Technology*
at TU Berlin.

## Features

- Real-time binaural rendering of two independent sources via measured HRTFs (SOFA)
- Live head-tracking over MIDI (Supperware Head Tracker), with **click-free HRIR crossfading**
- Per-source azimuth, elevation and distance control from a MIDI controller (AKAI APC mini mk2)
  or on-screen sliders
- Distance cues via level and a per-source reverb send
- Multithreaded engine (audio callback, tracker, MIDI and GUI threads) at 48 kHz
- Optional headphone equalisation via a minimum-phase inverse filter

## Requirements

- Python 3.10+
- `numpy`, `scipy`, `sounddevice`, `mido`, `pyfar`
- A SOFA HRIR dataset (required, not included; see **Resources**)
- Hardware: Supperware Head Tracker, AKAI APC mini mk2

The app sets **two pyfar demo signals per source** (drums / castanets and
guitar / speech), which pyfar downloads on first run. Add your own with `load_wav("file.wav")`;
the track buttons and MIDI mapping adapt to the number of tracks automatically.

## Setup

1. Install the dependencies:
   ```bash
   pip install numpy scipy sounddevice mido pyfar
   ```
2. Download an HRIR dataset and set `SOFA_FILE` to its path (see **Resources**).
3. (optional) Choose your output device: list devices with `python -m sounddevice`, then set
   `sd.default.device` near the top of the script. It is commented out by default, which uses
   the system default device.
4. (optional) Headphone EQ: set `EQ_ACTIVE = True` and point `EQ_IR_FILE` at a headphone
   impulse response. That is a WAV file holding the measured impulse response of your headphone
   (e.g. captured on an ear simulator such as the B&K 5128; mono or stereo, any sample rate, it
   is resampled internally). None is included and the demo works fine without HP EQ.

## Run

```bash
python HRTF_spatialmix_Laurenz_Raehs_2026-10-17.py
```

## Controls

- **Track buttons:** switch the source material per source
- **Faders:** distance, horizontal angle (pan) and vertical angle (tilt) per source, plus master gain.
- **Recenter button:** zero the head-tracker

MIDI is read from an AKAI APC mini mk2 by default; see `MIDI_CONFIG` in the script to remap.

## Resources

- **HRIR dataset:** FABIAN HRTF database (TU Berlin) —
  <https://depositonce.tu-berlin.de/items/bff6568a-5735-4ebc-b3fa-ac10707b7beb>
  (this project uses `FABIAN_HRIR_measured_HATO_0.sofa`)
- **Head-tracker:** Supperware Head Tracker —
  <https://supperware.co.uk/headtracker-overview>
- **MIDI controller:** AKAI APC mini mk2 (its MIDI communication protocol is documented by
  AKAI on the product page) — <https://www.akaipro.com/apc-mini-mk2>

## Notes

This is a course project. The HRIR dataset is licensed separately and is not redistributed here.

## License

Released under the MIT License. See [LICENSE](LICENSE).

## Author

Laurenz Raehs, <https://laurenzraehs.de>
