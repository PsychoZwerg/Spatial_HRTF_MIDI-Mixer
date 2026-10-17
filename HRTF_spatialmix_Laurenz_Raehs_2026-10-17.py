# imports

import numpy as np
import pyfar as pf
import pandas as pd
import sounddevice as sd
from scipy.signal import convolve, fftconvolve, minimum_phase
import tkinter as tk
from tkinter import ttk
import mido
import time
import threading
import warnings
import os

# pyfar warnungen nerven
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
os.environ['PYTHONWARNINGS'] = 'ignore'

# config
fs = 48000
block_size = 1024 *8

#####################################################
# KOPFHÖRER-ENTZERRU
EQ_ACTIVE = False
EQ_IR_FILE = "headphone_ir.wav"  # your own headphone impulse response (see README); only loaded when EQ_ACTIVE

# Parameter
NUM_FRACTIONS     = 3      
MAX_BOOST_DB      = 6     
BASS_CROSSOVER_HZ = 40     
IR_LENGTH         = 1024
N_FFT_INTERNAL    = 16384

if EQ_ACTIVE:
    # IR lade
    eq_ir_sig = pf.io.read_audio(EQ_IR_FILE)
    if eq_ir_sig.sampling_rate != fs:
        eq_ir_sig = pf.dsp.resample(eq_ir_sig, fs)

    # Peak zentrieren
    ir_raw = eq_ir_sig.time[0]
    ir_raw = np.roll(ir_raw, -np.argmax(np.abs(ir_raw)))

    #Glättung
    tmp_sig = pf.Signal(ir_raw[None, :], fs)
    smooth_sig, _ = pf.dsp.smooth_fractional_octave(tmp_sig, num_fractions=NUM_FRACTIONS)
    mag_smooth = np.abs(smooth_sig.freq[0])
    smooth_freqs = smooth_sig.frequencies

    #  normalisieren
    idx_1k = np.argmin(np.abs(smooth_freqs - 1000))
    mag_smooth_norm = mag_smooth / (mag_smooth[idx_1k] + 1e-12)

    #  1/H
    eps = 1e-3
    G_mag = 1.0 / (mag_smooth_norm + eps)

    # test, wegen zu krasser bassanhebung
    if BASS_CROSSOVER_HZ > 0:
        low_cut = 1 / (1 + (BASS_CROSSOVER_HZ / (smooth_freqs + 1e-10))**2)
        G_mag *= low_cut.flatten()

    # Gain Limit
    max_gain_lin = 10**(MAX_BOOST_DB / 20)
    G_mag = np.minimum(G_mag, max_gain_lin)
    G_mag /= (G_mag[idx_1k] + 1e-12)

    # Minimum Phase
    full_spec = np.concatenate([G_mag, G_mag[-2:0:-1]])
    ir_full = np.fft.ifft(full_spec).real

    ir_min = minimum_phase(ir_full, n_fft=len(full_spec))
    ir_min = ir_min[:IR_LENGTH] * np.hanning(IR_LENGTH)

    ir_fft_final = np.fft.rfft(ir_min, n=N_FFT_INTERNAL)
    test_freqs = np.fft.rfftfreq(N_FFT_INTERNAL, 1/fs)
    idx_test_1k = np.argmin(np.abs(test_freqs - 1000))
    ir_min /= (np.abs(ir_fft_final[idx_test_1k]) + 1e-12)

    # Konvertierung
    ir_min_f32 = ir_min.astype(np.float32)
    eq_len = len(ir_min_f32)

    # Overlap-Add Buffer für den globalen EQ
    eq_ovl_l = np.zeros(eq_len + block_size, dtype=np.float32)
    eq_ovl_r = np.zeros(eq_len + block_size, dtype=np.float32)
#####################################################

SOFA_FILE = "FABIAN_HRIR_measured_HATO_0.sofa"
data_ir, source_coords, _ = pf.io.read_sofa(SOFA_FILE)
hrir_len = data_ir.time.shape[-1]

# sounddevice
# sd.default.device = 8  # set to your output device index (list them with: python -m sounddevice); leave commented for the system default

def normalize_rms(audio, target_rms=0.1):
    rms = np.sqrt(np.mean(audio**2))
    if rms > 1e-10:
        audio = audio * (target_rms / rms)
    return audio

def load_wav(path):
    sig = pf.io.read_audio(path)
    if sig.sampling_rate != fs: sig = pf.dsp.resample(sig, fs)
    return normalize_rms(sig.time[0].astype(np.float32))

# Two built-in pyfar demo signals per source (freely available; pyfar downloads them on first run).
# Add your own audio with load_wav(); the track buttons and MIDI mapping adapt automatically, e.g.:
#   s1_tracks.append(load_wav("your_file.wav"))
def _pf_sample(sig):
    return normalize_rms(pf.dsp.resample(sig, fs).time[0].astype(np.float32))

s1_tracks = [_pf_sample(pf.signals.files.drums()),
             _pf_sample(pf.signals.files.castanets())]

s2_tracks = [_pf_sample(pf.signals.files.guitar()),
             _pf_sample(pf.signals.files.speech())]

TRACKER_NAME = "Head Tracker"
START_MSG = [0x00, 0x21, 0x42, 0x00, 0x00, 0x48, 0x01, 0x01]

MIDI_CONFIG = {
    'device_name': "APC mini mk2",
    'notes': {
        'src1_tracks': [56, 48, 40, 32], # Drums A, B, C, D
        'src2_tracks': [59, 51, 43, 35], # Guitar, E-Guit, Sax, Funk Guitar
        'recenter': 63,              # Center Button
        'fader_leds': {48: 0, 51: 3, 54: 6}
    },
    'colors': {
        'track_active': 5,    # Rot
        'track_ready': 3,     # Weiss
        'fader_high': 21,     # Hell-Grun
        'fader_low': 13,      # Dunkel-Grun
        'fader_muted': 5,     # Rot glimm
        'recenter_ready': 21, # Hell-Gruen
        'off': 0
    }
}


# REVERB SETUP 
reverb_sec = 2.0
reverb_len = int(fs * reverb_sec)
t_rev = np.linspace(0, reverb_sec, reverb_len)
# Create it
reverb_ir = np.array([
    np.random.normal(0, 0.05, reverb_len) * np.exp(-4 * t_rev / reverb_sec),
    np.random.normal(0, 0.05, reverb_len) * np.exp(-4 * t_rev / reverb_sec)
]).astype(np.float32)

# zentraler status
master_sample_idx = 0
state = {
    'src1': {
    'tracks': s1_tracks,
    'active_track': 0,
    'hrir': data_ir.time[0],
    'hrir_old': data_ir.time[0],
    'fade_ptr': 1.0,
    'pan': 0.0,
    'tilt': 0.0,
    'gain': 1.0,
    'reverb_send': 0.0,
    'distance': 0.0,
    'ovl_l': np.zeros(hrir_len + block_size),
    'ovl_r': np.zeros(hrir_len + block_size),
    'rev_ovl_l': np.zeros(reverb_len + block_size, dtype=np.float32),
    'rev_ovl_r': np.zeros(reverb_len + block_size, dtype=np.float32)

    },
    'src2': {
    'tracks': s2_tracks,
    'active_track': 0,
    'hrir': data_ir.time[0],
    'hrir_old': data_ir.time[0],
    'fade_ptr': 1.0,
    'pan': 0.0,
    'tilt': 0.0,
    'gain': 1.0,
    'reverb_send': 0.0,
    'distance': 0.0,
    'ovl_l': np.zeros(hrir_len + block_size),
    'ovl_r': np.zeros(hrir_len + block_size),
    'rev_ovl_l': np.zeros(reverb_len + block_size, dtype=np.float32),
    'rev_ovl_r': np.zeros(reverb_len + block_size, dtype=np.float32)
    },
    'master_gain': 1.0
}

tracker_state = {
    'yaw': 0.0, 'pitch': 0.0, 'roll': 0.0, 
    'offset_yaw': 0.0, 'offset_pitch': 0.0, 'offset_roll': 0.0,
    'raw_y': 0.0, 'raw_p': 0.0, 'raw_r': 0.0, 'last_y': 0.0
}


midi_out_port = None
midi_assignment = {}
track_buttons = {'src1': [], 'src2': []}

hrir_lock = threading.Lock()

#hilfsfunkionen-

# MIDI gedöns
def init_midi_output():
    global midi_out_port
    try:
        for name in mido.get_output_names():
            if MIDI_CONFIG['device_name'] in name and "Control" in name:
                midi_out_port = mido.open_output(name)
                send_midi_led(MIDI_CONFIG['notes']['recenter'], MIDI_CONFIG['colors']['recenter_ready'])
                return
    except: pass

def send_midi_led(note, color_code):
    if midi_out_port:
        try: midi_out_port.send(mido.Message('note_on', note=note, velocity=color_code))
        except: pass

def update_led_logic(cc_number, value):
    mapping = MIDI_CONFIG['notes']['fader_leds']
    cols = MIDI_CONFIG['colors']
    if cc_number in mapping:
        note = mapping[cc_number]
        color = cols['fader_muted'] if value <= 0.05 else (cols['fader_low'] if value < 0.35 else cols['fader_high'])
        send_midi_led(note, color)

# TRACKs
def update_track_selection(source, track_idx):
    if track_idx >= len(state[source]['tracks']):
        return  # ignore track buttons beyond the available tracks
    state[source]['active_track'] = track_idx
    cols = MIDI_CONFIG['colors']
    
    # GUI
    for i, btn in enumerate(track_buttons[source]):
        if i == track_idx: btn.config(bg='#FF5555', relief=tk.SUNKEN, fg='white')
        else: btn.config(bg='#d9d9d9', relief=tk.RAISED, fg='black')
            
    # MIDI
    hw_btns = MIDI_CONFIG['notes']['src1_tracks' if source == 'src1' else 'src2_tracks']
    for i, note in enumerate(hw_btns):
        color = cols['track_active'] if i == track_idx else cols['track_ready']
        send_midi_led(note, color)

def register_midi(cc, tk_var, min_val, max_val, callback=None):
    midi_assignment[cc] = {'var': tk_var, 'min': min_val, 'max': max_val, 'callback': callback}

# sptial
def update_spatial(source_name, pan_val, tilt_val):
    try:
        s = state[source_name]
        s['pan'], s['tilt'] = float(pan_val), float(tilt_val)
        rel_pan = (s['pan'] + tracker_state['yaw'] + 180) % 360 - 180
        rel_tilt = (s['tilt'] - tracker_state['pitch'] + 180) % 360 - 180
        idx, *_ = source_coords.find_nearest_k(-rel_pan, -rel_tilt, 1.5, k=1, domain='sph', convention='top_elev', unit='deg')
        new_hrir = data_ir.time[idx]
        if not np.array_equal(new_hrir, s['hrir']):
            with hrir_lock:
                s['hrir_old'] = s['hrir'].copy()
                s['hrir'] = new_hrir
                s['fade_ptr'] = 0.0
    except: pass

def update_distance(source_name, d):
    s = state[source_name]
    d = float(d)

    s['distance'] = d

    # Gain: max 0 dB zu -6 dB
    s['gain'] = 10**(-6 * d / 20)

    # Reverb: ansteigend mit d
    s['reverb_send'] = 0.1 + 0.6 * (d**1.5)

def audio_callback(outdata, frames, time, status):
    if status:
        print(status)
    global master_sample_idx, eq_ovl_l, eq_ovl_r
    outdata.fill(0)

    for name in ['src1', 'src2']:
        s = state[name]

        audio_data = s['tracks'][s['active_track']]
        idx = (np.arange(master_sample_idx, master_sample_idx + frames)) % len(audio_data)
        block = audio_data[idx] * s['gain']

        with hrir_lock:
            hrir     = s['hrir'].copy()
            hrir_old = s['hrir_old'].copy()
            fade_ptr = s['fade_ptr']

        cl_new = fftconvolve(block, hrir[0])
        cr_new = fftconvolve(block, hrir[1])

        if fade_ptr < 1.0:
            # fade step basierend auf block size
            fade_step = frames / (8 * block_size)
            new_ptr = min(s['fade_ptr'] + fade_step, 1.0)

            f_in  = np.sin(np.linspace(s['fade_ptr'], new_ptr, frames) * (np.pi / 2))
            f_out = np.cos(np.linspace(s['fade_ptr'], new_ptr, frames) * (np.pi / 2))

            cl_old = fftconvolve(block, hrir_old[0])
            cr_old = fftconvolve(block, hrir_old[1])

            cl = cl_new.copy()
            cr = cr_new.copy()
            cl[:frames] = cl_old[:frames] * f_out + cl_new[:frames] * f_in
            cr[:frames] = cr_old[:frames] * f_out + cr_new[:frames] * f_in

            with hrir_lock:
                s['fade_ptr'] = new_ptr
        else:
            cl, cr = cl_new, cr_new 

        # overlap-add
        s['ovl_l'][:len(cl)] += cl
        s['ovl_r'][:len(cr)] += cr

        outdata[:, 0] += s['ovl_l'][:frames]
        outdata[:, 1] += s['ovl_r'][:frames]

        s['ovl_l'][:hrir_len] = s['ovl_l'][frames:frames + hrir_len]
        s['ovl_l'][hrir_len:] = 0
        s['ovl_r'][:hrir_len] = s['ovl_r'][frames:frames + hrir_len]
        s['ovl_r'][hrir_len:] = 0

        # REVERB je source
        rev_in = block * s['reverb_send']
        
        # falten block mit Reverb IR
        rl = fftconvolve(rev_in, reverb_ir[0])
        rr = fftconvolve(rev_in, reverb_ir[1])
        
        # overlapt buffer
        s['rev_ovl_l'][:len(rl)] += rl
        s['rev_ovl_r'][:len(rr)] += rr
        
        # Mix in output
        outdata[:, 0] += s['rev_ovl_l'][:frames]
        outdata[:, 1] += s['rev_ovl_r'][:frames]
        
        # shift
        s['rev_ovl_l'][:reverb_len] = s['rev_ovl_l'][frames:frames + reverb_len]
        s['rev_ovl_l'][reverb_len:] = 0
        s['rev_ovl_r'][:reverb_len] = s['rev_ovl_r'][frames:frames + reverb_len]
        s['rev_ovl_r'][reverb_len:] = 0

    master_sample_idx += frames

    if EQ_ACTIVE:
        el = fftconvolve(outdata[:, 0].copy(), ir_min_f32)
        er = fftconvolve(outdata[:, 1].copy(), ir_min_f32)

        print("el max:", np.max(np.abs(el)))

        eq_ovl_l[:len(el)] += el
        eq_ovl_r[:len(er)] += er
        outdata[:, 0] = eq_ovl_l[:frames]
        outdata[:, 1] = eq_ovl_r[:frames]
        eq_ovl_l[:eq_len] = eq_ovl_l[frames:frames + eq_len]
        eq_ovl_l[eq_len:] = 0
        eq_ovl_r[:eq_len] = eq_ovl_r[frames:frames + eq_len]
        eq_ovl_r[eq_len:] = 0

    outdata[:] = np.clip(outdata * state['master_gain'] * 0.2, -1.0, 1.0)

# handlers
def tracker_handler():
    in_name = next((n for n in mido.get_input_names() if TRACKER_NAME in n), None)
    out_name = next((n for n in mido.get_output_names() if TRACKER_NAME in n), None)
    if not in_name: return
    try:
        with mido.open_input(in_name) as inp, mido.open_output(out_name) as outp:
            while inp.receive(block=False): pass
            for _ in range(3):
                outp.send(mido.Message('sysex', data=START_MSG)); time.sleep(0.1)
            while True:
                for msg in inp.iter_pending():
                    if msg.type == 'sysex' and len(msg.data) >= 11 and msg.data[3] == 0x40:
                        def conv(l, h): 
                            v = (l << 7) | h
                            return v - 16384 if v > 8191 else v
                        scale = 180.0 / 6380.0
                        tracker_state['raw_y'] = conv(msg.data[5], msg.data[6]) * scale
                        tracker_state['raw_p'] = conv(msg.data[7], msg.data[8]) * scale
                        if len(msg.data) >= 13: tracker_state['raw_r'] = conv(msg.data[9], msg.data[10]) * scale
                        cy = tracker_state['raw_y'] - (tracker_state['raw_r'] * 0.25)
                        tracker_state['yaw'] = (cy - tracker_state['offset_yaw'] + 180) % 360 - 180
                        tracker_state['pitch'] = (tracker_state['raw_p'] - tracker_state['offset_pitch'] + 180) % 360 - 180
                        if abs(tracker_state['yaw'] - tracker_state.get('last_y', 0)) > 0.5:
                            update_spatial('src1', state['src1']['pan'], state['src1']['tilt'])
                            update_spatial('src2', state['src2']['pan'], state['src2']['tilt'])
                            tracker_state['last_y'] = tracker_state['yaw']
                            head_label.config(text=f"Y: {tracker_state['yaw']:+.1f}° | P: {tracker_state['pitch']:+.1f}°")
                time.sleep(0.005)
    except: pass

def midi_handler():
    inp_name = next((n for n in mido.get_input_names() if MIDI_CONFIG['device_name'] in n and "Control" in n), None)
    if not inp_name: return
    with mido.open_input(inp_name) as inport:
        for msg in inport:
            if msg.type == 'note_on':
                if msg.note in MIDI_CONFIG['notes']['src1_tracks']:
                    root.after(0, lambda: update_track_selection('src1', MIDI_CONFIG['notes']['src1_tracks'].index(msg.note)))
                elif msg.note in MIDI_CONFIG['notes']['src2_tracks']:
                    root.after(0, lambda: update_track_selection('src2', MIDI_CONFIG['notes']['src2_tracks'].index(msg.note)))
                elif msg.note == MIDI_CONFIG['notes']['recenter']:
                    quick_recenter()
            elif msg.type == 'control_change' and msg.control in midi_assignment:
                m = midi_assignment[msg.control]
                val = (msg.value / 127.0) * (m['max'] - m['min']) + m['min']
                root.after(0, lambda v=val, cfg=m: [cfg['var'].set(v), cfg['callback'](v) if cfg['callback'] else None])

def quick_recenter():
    tracker_state['offset_yaw'], tracker_state['offset_pitch'] = tracker_state['raw_y'], tracker_state['raw_p']
    tracker_state['yaw'], tracker_state['pitch'], tracker_state['last_y'] = 0.0, 0.0, -999
    print("Zentriert.")

def on_closing():
    if midi_out_port:
        # Schalte alle relevanten LEDs aus
        notes = MIDI_CONFIG['notes']
        all_notes = notes['src1_tracks'] + notes['src2_tracks'] + list(notes['fader_leds'].values()) + [notes['recenter']]
        for n in all_notes: send_midi_led(n, 0)
        time.sleep(0.1); midi_out_port.close()
    root.destroy()

# gui bausteine
def create_source_controls(parent, name, label_text, start_cc):
    frame = ttk.LabelFrame(parent, text=label_text, padding=10)
    frame.pack(side=tk.LEFT, padx=10, fill=tk.Y)
    btn_frame = ttk.Frame(frame); btn_frame.pack(pady=5, fill=tk.X)
    _names = {'src1': ["Drums", "Castanets"], 'src2': ["Guitar", "Speech"]}.get(name, [])
    _tracks = state[name]['tracks']
    t_names = [_names[i] if i < len(_names) else f"Track {i+1}" for i in range(len(_tracks))]
    for i, t_n in enumerate(t_names):
        b = tk.Button(btn_frame, text=t_n, bg='#d9d9d9', command=lambda idx=i: update_track_selection(name, idx))
        b.pack(fill=tk.X, padx=2, pady=1); track_buttons[name].append(b)
    g_v, p_v, t_v = tk.DoubleVar(value=0.0), tk.DoubleVar(value=0), tk.DoubleVar(value=0)

    ttk.Label(frame, text="Distance").pack()

    ttk.Scale(frame,
            from_=1.0, to=0.0,
            orient=tk.VERTICAL,
            variable=g_v,
            length=150,
            command=lambda v: [
                update_distance(name, v),
                update_led_logic(start_cc, float(v))
            ]).pack()

    register_midi(start_cc, g_v, 0.0, 1.0,
                lambda v: [
                    update_distance(name, v),
                    update_led_logic(start_cc, float(v))
                ])

    # spatial controls
    ttk.Label(frame, text="Pan (L/R)").pack(pady=(10,0))
    ttk.Scale(frame, from_=-90, to=90, variable=p_v, 
              command=lambda v: update_spatial(name, v, t_v.get())).pack(fill=tk.X, padx=5)
    register_midi(start_cc + 1, p_v, -90, 90, lambda v: update_spatial(name, v, t_v.get()))

    ttk.Label(frame, text="Tilt (Up/Down)").pack(pady=(10,0))
    ttk.Scale(frame, from_=-90, to=90, orient=tk.VERTICAL, variable=t_v, length=100,
              command=lambda v: update_spatial(name, p_v.get(), v)).pack(pady=5)
    register_midi(start_cc + 2, t_v, 90, -90, lambda v: update_spatial(name, p_v.get(), v))

# GUI udn AUFRUF
# UI Konstruktion
root = tk.Tk(); root.title("Binaural Mixer"); root.geometry("1400x1000")
root.protocol("WM_DELETE_WINDOW", on_closing)
status_frame = ttk.LabelFrame(root, text="Head Tracker", padding=15); status_frame.pack(fill=tk.X, padx=20, pady=10)
head_label = ttk.Label(status_frame, text="Y: +0.0° | P: +0.0°", font=("Monospace", 16, "bold")); head_label.pack()
ttk.Button(status_frame, text="Zentrieren", command=quick_recenter).pack(pady=5)
mixer_frame = ttk.Frame(root, padding=20); mixer_frame.pack()
create_source_controls(mixer_frame, 'src1', "Source 1", 48)
create_source_controls(mixer_frame, 'src2', "Source 2", 51)

# Master
room_frame = ttk.LabelFrame(root, text="Master", padding=10)
room_frame.pack(fill=tk.X, padx=20, pady=10)

mg_v = tk.DoubleVar(value=0.5)
ttk.Label(room_frame, text="Master Volume").pack(anchor=tk.W)
ttk.Scale(room_frame, from_=0, to=1.0, variable=mg_v, 
          command=lambda v: [state.update({'master_gain': float(v)}), update_led_logic(56, float(v))]).pack(fill=tk.X, padx=5, pady=(0, 10))
register_midi(56, mg_v, 0, 1.0, lambda v: [state.update({'master_gain': float(v)}), update_led_logic(56, float(v))])

# Start
init_midi_output()

threading.Thread(target=tracker_handler, daemon=True).start()
threading.Thread(target=midi_handler, daemon=True).start()
root.after(500, lambda: [update_led_logic(48, 1.0), update_led_logic(51, 1.0), update_led_logic(54, 1.0), update_track_selection('src1', 0), update_track_selection('src2', 0)])

with sd.OutputStream(samplerate=fs, channels=2, callback=audio_callback, blocksize=block_size):
    root.mainloop()

