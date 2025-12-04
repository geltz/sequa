import ctypes
import sys
import os
import time
import numpy as np
import soundfile as sf
from scipy import signal

from PyQt6.QtCore import (Qt, pyqtSignal, QTimer, QRectF, QIODevice, 
                          QByteArray, QPropertyAnimation, QEasingCurve, 
                          QPointF, QUrl)
from PyQt6.QtGui import (QColor, QPainter, QPen, QFont, QPainterPath, 
                         QLinearGradient, QBrush, QRadialGradient, QIcon, 
                         QFontMetrics, QPolygonF)
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QSlider, QPushButton, 
                             QFrame, QMessageBox, QGraphicsOpacityEffect,
                             QSizePolicy, QTabWidget, QGridLayout, QFileDialog)
from PyQt6.QtMultimedia import QAudioSink, QAudioFormat, QMediaDevices

# --- Constants ---
SR = 44100
BPM_DEFAULT = 120
STEPS = 16
BUFFER_MS = 60

# --- Audio Streaming ---

from PyQt6.QtCore import QMutex, QMutexLocker

class LoopGenerator(QIODevice):
    def __init__(self, fmt, parent=None):
        super().__init__(parent)
        self.data = bytes()
        self.pos = 0
        self.fmt = fmt
        self.playing = False
        self.mutex = QMutex()
        self.open(QIODevice.OpenModeFlag.ReadOnly)

    def set_playback_state(self, is_playing):
        with QMutexLocker(self.mutex):
            self.playing = is_playing

    def set_data(self, float_data):
        if float_data is None:
            float_data = np.zeros(SR // 2, dtype=np.float32)
            
        clean_data = np.nan_to_num(float_data, copy=False, nan=0.0)
        # Smooth clip
        audio = np.tanh(clean_data) 
        audio = (audio * 32767).astype(np.int16)
        new_bytes = audio.tobytes()
        
        with QMutexLocker(self.mutex):
            # Atomic swap
            self.data = new_bytes
            # Safety: if position is out of bounds of new data, reset
            if self.pos >= len(self.data):
                self.pos = 0

    def readData(self, maxlen):
        with QMutexLocker(self.mutex):
            if not self.playing or not self.data:
                # Return silence to prevent crackle of underlying buffer garbage
                return b'\x00' * maxlen

            if maxlen % 2 != 0: maxlen -= 1
            total_size = len(self.data)
            if total_size == 0: return b'\x00' * maxlen
            
            chunk = b''
            while len(chunk) < maxlen:
                if self.pos >= total_size: self.pos = 0
                remaining = total_size - self.pos
                needed = maxlen - len(chunk)
                take = min(remaining, needed)
                chunk += self.data[self.pos : self.pos + take]
                self.pos += take
            return chunk

    def writeData(self, data): return 0
    def bytesAvailable(self): return len(self.data) + 4096
    def isSequential(self): return True

class SoundPreview(QIODevice):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.data_bytes = QByteArray()
        self.pos = 0
        self.fmt = QAudioFormat()
        self.fmt.setSampleRate(SR)
        self.fmt.setChannelCount(1)
        self.fmt.setSampleFormat(QAudioFormat.SampleFormat.Int16)
        self.sink = QAudioSink(QMediaDevices.defaultAudioOutput(), self.fmt)
        self.open(QIODevice.OpenModeFlag.ReadOnly)

    def play(self, float_data):
        self.sink.stop()
        if float_data is None: return
        
        # Sanitize to prevent crashes/noise
        audio = np.nan_to_num(float_data, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        audio = np.clip(audio, -1.0, 1.0)
        audio = (audio * 32767).astype(np.int16)
        
        self.data_bytes = QByteArray(audio.tobytes())
        self.pos = 0
        self.sink.start(self)

    def readData(self, maxlen):
        if self.pos >= self.data_bytes.size(): return b''
        
        # Ensure even byte alignment for 16-bit audio
        if maxlen % 2 != 0: maxlen -= 1
        
        chunk = self.data_bytes.mid(self.pos, maxlen)
        self.pos += chunk.size()
        return chunk.data()

    def writeData(self, data): return 0
    def bytesAvailable(self): return self.data_bytes.size() - self.pos
    def isSequential(self): return True

# --- Drum Kits ---

class SimpleDrums:
    @staticmethod
    def generate(drum_type, params):
        p_pitch = params.get('pitch', 0.5)
        p_decay = params.get('decay', 0.5)
        p_tone = params.get('tone', 0.5)
        
        # Slightly longer duration to accommodate soft tails
        duration = 0.6
        t = np.linspace(0, duration, int(SR * duration), endpoint=False)
        y = np.zeros_like(t)
        
        if drum_type == "kick":
            # Round, rubbery kick (Pillowy Sine)
            f_start = 120 + (p_pitch * 50)
            f_end = 40
            freq_env = (f_start - f_end) * np.exp(-t * 12) + f_end
            phase = np.cumsum(freq_env) * 2 * np.pi / SR
            osc = np.sin(phase)
            
            d_amp = 6 + ((1.0 - p_decay) * 40)
            
            # SCOOP: Cut low-mids (250Hz) to remove boxiness
            sos_scoop = signal.butter(1, [200, 350], 'bs', fs=SR, output='sos')
            osc = signal.sosfilt(sos_scoop, osc)
            
            sos_lp = signal.butter(2, 5000, 'lp', fs=SR, output='sos')
            y = signal.sosfilt(sos_lp, osc) * np.exp(-t * d_amp)

        elif drum_type == "snare":
            f_body = 150 + (p_pitch * 25)
            body = np.sin(2 * np.pi * f_body * t) * np.exp(-t * 20)
            
            noise = np.random.uniform(-0.5, 0.5, len(t))
            f_center = 2200 + (p_tone * 1500)
            sos_bp = signal.butter(2, [f_center - 1000, f_center + 1000], 'bp', fs=SR, output='sos')
            noise = signal.sosfilt(sos_bp, noise)
            
            d_noise = 15 + ((1.0 - p_decay) * 40)
            
            # Cut the mid frequencies (500Hz) from the body
            sos_scoop = signal.butter(1, [400, 700], 'bs', fs=SR, output='sos')
            body = signal.sosfilt(sos_scoop, body)
            
            # Reduced body volume relative to noise for a lighter, less mid-heavy sound
            y = (body * 0.6) + (noise * np.exp(-t * d_noise) * 0.45)

        elif "hat" in drum_type or "cymbal" in drum_type:
            # The "Hiss" (Bandpassed Noise)
            noise = np.random.uniform(-1, 1, len(t))
            bp_center = 7000 + (p_tone * 3000)
            sos_bp = signal.butter(2, [bp_center - 2000, bp_center + 2000], 'bp', fs=SR, output='sos')
            hiss = signal.sosfilt(sos_bp, noise)
            
            f_metal = 800 + (p_pitch * 300)
            mod = np.sin(2 * np.pi * (f_metal * 3.5) * t) * 50
            metal = np.sin(2 * np.pi * (f_metal + mod) * t) * 0.2
            
            sig = hiss + metal
            
            if "closed" in drum_type:
                decay = 60 + ((1.0 - p_decay) * 200)
                attack = np.minimum(t * 2000, 1.0)
                y = sig * attack * np.exp(-t * decay)
            else:
                decay = 10 + ((1.0 - p_decay) * 30)
                # FIX: Add attack ramp to remove click (2ms fade-in)
                attack = np.minimum(t * 500, 1.0)
                y = sig * attack * np.exp(-t * decay) * 0.8

        elif drum_type == "clap":
            noise = np.random.uniform(-1, 1, len(t))
            # Shifted higher to avoid mid-range mud
            bp_low = 900 + (p_pitch * 200)
            bp_high = bp_low + 800
            sos = signal.butter(2, [bp_low, bp_high], 'bp', fs=SR, output='sos')
            filt = signal.sosfilt(sos, noise)
            
            env = np.exp(-t * (10 + (1.0 - p_decay) * 30))
            attack = min(len(t), int(SR * 0.015)) 
            env[:attack] *= np.linspace(0, 1, attack)
            y = filt * env

        elif drum_type == "wood" or "perc" in drum_type and "a" in drum_type:
            f_base = 600 + (p_pitch * 300)
            f_env = f_base * (1.0 + 0.1 * np.exp(-t * 50))
            phase = np.cumsum(f_env) * 2 * np.pi / SR
            osc = np.sin(phase)
            decay = 30 + ((1.0 - p_decay) * 100)
            y = osc * np.exp(-t * decay)

        else:
            # Tom: Pure Sine Sweep
            f = 125 + (p_pitch * 100)
            f_env = f * (1.0 - 0.2 * np.exp(-t * 10))
            phase = np.cumsum(f_env) * 2 * np.pi / SR
            y_raw = np.sin(phase) * np.exp(-t * (8 + (1.0 - p_decay) * 25))
            
            # SCOOP: Remove boxy mids at 400Hz
            sos_scoop = signal.butter(1, [300, 500], 'bs', fs=SR, output='sos')
            y = signal.sosfilt(sos_scoop, y_raw)
            
        return SynthEngine.finalize(y)

class PCMDrums:
    @staticmethod
    def degrade(data, sr_target=22050, bit_depth=8):
        # 1. Resample (Aliasing)
        if sr_target < SR:
            factor = int(SR / sr_target)
            down = data[::factor]
            data = np.repeat(down, factor)[:len(data)]
        
        # 2. Bit Crush
        steps = 2 ** bit_depth
        data = np.round(data * steps) / steps
        return data

    @staticmethod
    def generate(drum_type, params):
        p_pitch = params.get('pitch', 0.5)
        p_decay = params.get('decay', 0.5)
        p_tone = params.get('tone', 0.5)
        
        duration = 0.5
        t = np.linspace(0, duration, int(SR * duration), endpoint=False)
        y = np.zeros_like(t)

        if drum_type == "kick":
            # "Thud" Kick: Fixed freq body + Click
            f_body = 60 + (p_pitch * 20)
            body = np.sin(2 * np.pi * f_body * t)
            
            click = np.random.uniform(-1, 1, len(t))
            sos_click = signal.butter(2, 800, 'lp', fs=SR, output='sos')
            click = signal.sosfilt(sos_click, click) * np.exp(-t * 100)
            
            env = np.exp(-t * (10 + (1.0 - p_decay) * 30))
            y = (body * 0.8 + click * 0.4) * env
            y = PCMDrums.degrade(y, sr_target=16000, bit_depth=10)

        elif drum_type == "snare":
            # "Fat" Snare: Square wave + Noise
            f_tone = 160 + (p_pitch * 40)
            tone = np.sign(np.sin(2 * np.pi * f_tone * t)) # Square
            sos_tone = signal.butter(1, 400, 'lp', fs=SR, output='sos')
            tone = signal.sosfilt(sos_tone, tone)
            
            noise = np.random.uniform(-1, 1, len(t))
            sos_mid = signal.butter(2, [600, 2000], 'bp', fs=SR, output='sos')
            noise = signal.sosfilt(sos_mid, noise)
            
            d_val = 15 + ((1.0 - p_decay) * 50)
            y = (tone * 0.4 + noise * 0.8) * np.exp(-t * d_val)
            y = PCMDrums.degrade(y, sr_target=24000, bit_depth=8)
            
        elif "hat" in drum_type:
            # Shaker/Hat
            noise = np.random.uniform(-1, 1, len(t))
            hp_f = 6000 + (p_pitch * 2000)
            sos_hp = signal.butter(2, hp_f, 'hp', fs=SR, output='sos')
            y = signal.sosfilt(sos_hp, noise)
            
            d_val = 80 if "closed" in drum_type else 15
            d_val += (1.0 - p_decay) * 100
            y *= np.exp(-t * d_val)
            y = PCMDrums.degrade(y, sr_target=32000, bit_depth=6)

        elif drum_type == "clap":
            # Digital Clap: Wide, noisy, heavy crush
            noise = np.random.uniform(-1, 1, len(t))
            
            # Wider bandpass (600-3000) for a "trashy" sound
            bp_low = 600 + (p_pitch * 200)
            bp_high = 3000
            sos = signal.butter(2, [bp_low, bp_high], 'bp', fs=SR, output='sos')
            filt = signal.sosfilt(sos, noise)
            
            # "Reverb" tail simulation via decay
            d_val = 10 + ((1.0 - p_decay) * 30)
            
            # Simple flam (just double hit) to distinguish from snare
            env = np.exp(-t * d_val)
            delay_samples = int(0.015 * SR)
            env[delay_samples:] += 0.6 * np.exp(-t[:-delay_samples] * d_val)
            
            y = filt * env
            y = PCMDrums.degrade(y, sr_target=14000, bit_depth=8)

        elif drum_type == "wood" or "perc" in drum_type and "a" in drum_type:
            # Cowbell/Rim: Pulse wave + Bandpass
            f = 500 + (p_pitch * 300)
            y = np.sign(np.sin(2 * np.pi * f * t)) 
            # Bandpass to simulate hollow wood/metal
            sos_bp = signal.butter(2, [f, f+1500], 'bp', fs=SR, output='sos')
            y = signal.sosfilt(sos_bp, y)
            y *= np.exp(-t * (30 + (1.0-p_decay)*250))
            y = PCMDrums.degrade(y, sr_target=18000, bit_depth=8)

        else:
            # Tom: Sampled Tom (Lower pitch, longer decay)
            f = 100 + (p_pitch * 80)
            y = np.sin(2 * np.pi * f * t)
            # Add a little bit of noise for texture
            y += np.random.uniform(-0.1, 0.1, len(t))
            y *= np.exp(-t * (8 + (1.0-p_decay)*20))
            y = PCMDrums.degrade(y, sr_target=12000, bit_depth=9)

        return SynthEngine.finalize(y)

class EightDrums:
    @staticmethod
    def generate(drum_type, params):
        p_pitch = params.get('pitch', 0.5)
        p_decay = params.get('decay', 0.5)
        p_tone = params.get('tone', 0.5)
        
        duration = 0.8
        t = np.linspace(0, duration, int(SR * duration), endpoint=False)
        y = np.zeros_like(t)
        
        rng = np.random.default_rng(int(p_pitch * 1000 + p_tone * 100))

        if drum_type == "kick":
            # 1. Pitch
            f_start = 80 + (p_pitch * 320)
            f_end = 40 + (p_pitch * 10)
            decay_transient = 0.008
            
            freq_env = (f_start - f_end) * np.exp(-t / decay_transient) + f_end
            phase = np.cumsum(freq_env) * 2 * np.pi / SR
            y = np.sin(phase)
            
            # 2. Tone (Click/Thud)
            # FIX: Decay coefficient changed from 0.05 (infinite noise) to 150 (short click)
            thud_cutoff = 200 + (p_tone * 4000)
            thud_noise = np.random.uniform(-1, 1, len(t))
            sos_thud = signal.butter(2, thud_cutoff, 'lp', fs=SR, output='sos')
            thud = signal.sosfilt(sos_thud, thud_noise) * np.exp(-t * 150)
            
            # 3. Decay
            amp_decay = 30 - (p_decay * 28.5)
            y = (y + thud * 0.4) * np.exp(-t * amp_decay)
            
            # 4. Anti-Click Fade Out
            if len(y) > 100:
                y[-100:] *= np.linspace(1, 0, 100)

        elif drum_type == "snare":
            f_body = 140 + (p_pitch * 120)
            tone_osc = np.sin(2 * np.pi * f_body * t) * np.exp(-t * (30 + (1.0-p_decay)*60))
            dust_raw = rng.uniform(-1, 1, len(t))
            filt_center = 1000 + (p_tone * 5000)
            noise = rng.uniform(-1, 1, len(t))
            noise = signal.sosfilt(signal.butter(2, [filt_center, filt_center+2000], 'bp', fs=SR, output='sos'), noise)
            noise_env = np.exp(-t * (30 + ((1.0-p_decay) * 80)))
            y = (tone_osc * 0.5) + (noise * noise_env * 0.8)

        elif drum_type == "closed hat" or drum_type == "open hat":

            base_f = 300 + (p_pitch * 200) 
            ratios = [2.0, 3.0, 4.16, 5.43, 6.79, 8.21]
            metal_sig = np.zeros_like(t)
            
            for r in ratios:
                phase_offset = np.random.rand() * 2 * np.pi
                metal_sig += np.sign(np.sin((2 * np.pi * base_f * r * t) + phase_offset))
            
            metal_sig /= len(ratios)
    
            noise_sig = np.random.uniform(-1, 1, len(t))
            
            # Pre-filter noise: Keep strictly high frequency
            sos_noise_hp = signal.butter(2, 7000, 'hp', fs=SR, output='sos')
            noise_sig = signal.sosfilt(sos_noise_hp, noise_sig)

            mix_ratio = 0.35 - (p_tone * 0.1) 
            sum_sig = (metal_sig * mix_ratio) + (noise_sig * (1 - mix_ratio))

            # --- Main filtering ---

            # Change order from 2 -> 4. This creates a much steeper "cliff",
            # chopping out the 2k-4k range completely.
            hp_freq = 2000 + (p_tone * 3000)
            sos_hp = signal.butter(4, hp_freq, 'hp', fs=SR, output='sos')
            processed = signal.sosfilt(sos_hp, sum_sig)

            bp_low = 6000 + (p_tone * 1000)
            bp_high = bp_low + 8000 
            sos_bp = signal.butter(2, [bp_low, bp_high], 'bp', fs=SR, output='sos')
            processed = signal.sosfilt(sos_bp, processed)

            # --- Envelope ---
            
            if drum_type == "closed hat":
                # Made decay slightly faster
                decay_coef = 90 + ((0.75 - p_decay) * 250)
                env = np.exp(-t * decay_coef)
            else:
                # Open hat
                env = 0.7 * np.exp(-t * (10 + (1-p_decay)*50)) + \
                      0.3 * np.exp(-t * (2 + (1-p_decay)*10))

            # Anti-click attack
            attack_samples = int(SR * 0.0030) 
            if len(processed) > attack_samples:
                smooth_attack = 0.5 * (1 - np.cos(np.linspace(0, np.pi, attack_samples)))
                processed[:attack_samples] *= smooth_attack

            y = processed * env

        elif drum_type == "clap":
            noise = rng.uniform(-1, 1, len(t))

            low = 900 + (p_pitch * 200)
            high = 2500 + (p_pitch * 600)
            filt = signal.sosfilt(signal.butter(2, [low, high], 'bp', fs=SR, output='sos'), noise)
            
            # Envelope: The "Flam" (Multiple sharp impulses)
            env = np.zeros_like(t)
            
            # Tighter spacing (9ms) for a "punchier" feel without adding bass
            pulse_spacing = 0.009
            
            # Decay affects the tail length
            tail_decay = 30 + ((1.0 - p_decay) * 60)
            
            for i in range(4):
                start_idx = int(i * pulse_spacing * SR)
                if start_idx >= len(env): break
                
                # Last hit is the loudest (1.0), pre-hits are transients (0.7)
                amp = 0.7 if i < 3 else 1.0
                
                remaining = len(env) - start_idx
                local_t = np.linspace(0, remaining/SR, remaining)
                
                # Extremely sharp decay for the flam hits
                decay = 250 if i < 3 else tail_decay
                pulse_env = np.exp(-local_t * decay) * amp
                
                env[start_idx:] = np.maximum(env[start_idx:], pulse_env)

            y = filt * env

        elif drum_type == "perc a": 
            f = 400 + (p_pitch * 800)
            fm = np.sin(2 * np.pi * (f * 0.5) * t) * (p_tone * 500)
            y = np.sin(2 * np.pi * (f + fm) * t) 
            d_val = 350 + ((1.0 - p_decay) * 200)
            y *= np.exp(-t * d_val)

        elif drum_type == "perc b":
            base = 60 + (p_pitch * 120)
            p_env_d = 20 + (p_decay * 40)
            freq = base * (1.0 + (0.5 + p_tone) * np.exp(-t * p_env_d))
            phase = np.cumsum(freq) * 2 * np.pi / SR
            y = np.tanh(np.sin(phase) * 1.2) 
            amp_d = 20 + ((1.0 - p_decay) * 80)
            y *= np.exp(-t * amp_d)

        # --- High Frequency Smoothing ---
        # Gentle 1st order Lowpass at 17.5kHz to remove digital harshness/aliasing
        # This increases perceived quality ("studio polish")
        sos_smooth = signal.butter(1, 17500, 'lp', fs=SR, output='sos')
        y = signal.sosfilt(sos_smooth, y)

        peak = np.max(np.abs(y))
        if peak > 0: y /= peak

        # Reduce hat gain relative to other drums (0.85 = 85% volume)
        if "hat" in drum_type:
            y *= 0.85

        return SynthEngine.finalize(y)

class NineDrums:
    @staticmethod
    def generate(drum_type, params):
        p_pitch = params.get('pitch', 0.5)
        p_decay = params.get('decay', 0.5)
        p_tone = params.get('tone', 0.5)
        
        duration = 0.6 # Shortened slightly for tightness
        t = np.linspace(0, duration, int(SR * duration), endpoint=False)
        y = np.zeros_like(t)

        if drum_type == "kick":
            f_start = 150 + (p_pitch * 250)
            f_end = 45 + (p_pitch * 30)
            f_decay = 30 + (p_decay * 50) 
            freq_env = (f_start - f_end) * np.exp(-t * f_decay) + f_end
            phase = np.cumsum(freq_env) * 2 * np.pi / SR
            drive = 1.0 + (p_tone * 4.0)
            osc = np.tanh(np.sin(phase) * drive)
            amp_decay = 5 + ((1.0 - p_decay) * 45)
            click = np.random.normal(0, 0.5, len(t)) * np.exp(-t * 300)
            y = (osc + click * 0.4) * np.exp(-t * amp_decay)

        elif drum_type == "snare":
            # REWORKED: cleaner, crispier, less compressed
            
            # 1. The Wires (High Freq Noise)
            noise = np.random.uniform(-1, 1, len(t))
            # Shift filter higher (1.5k - 8k) for "crispness"
            snap_low = 1500 + (p_tone * 1000)
            snap_high = snap_low + 6000
            sos_snap = signal.butter(2, [snap_low, snap_high], 'bp', fs=SR, output='sos')
            snap = signal.sosfilt(sos_snap, noise)
            
            # 2. The Body (Fundamental Tone) - Defined sine instead of muddy noise
            f_root = 180 + (p_pitch * 40)
            # Fast pitch drop for the "thwack"
            tone_phase = np.cumsum(f_root * (1.0 - 0.1 * t)) * 2 * np.pi / SR
            tone = np.sin(tone_phase)
            
            # 3. Envelopes - Tighter release
            d_snap = 20 + ((1.0 - p_decay) * 50)
            d_body = 25 + ((1.0 - p_decay) * 30)
            
            env_snap = np.exp(-t * d_snap)
            env_body = np.exp(-t * d_body)
            
            # 4. Mix - More snap, less body
            y = (snap * env_snap * 0.9) + (tone * env_body * 0.45)
            
            # 5. Saturation - Gentle drive (1.1x) to glue, not smash (was 1.5x)
            y = np.tanh(y * 1.1)

            # 6. Cleanup - Scoop the "cardboard" boxiness at 400Hz
            sos_scoop = signal.butter(1, [350, 600], 'bs', fs=SR, output='sos')
            y = signal.sosfilt(sos_scoop, y)
            
        elif "hat" in drum_type or "cymbal" in drum_type:
            sig = np.zeros_like(t)
            freqs = [263, 400, 421, 474, 587, 845]
            for f in freqs:
                f_tune = f * (1.0 + p_pitch * 0.2)
                sig += np.sign(np.sin(2 * np.pi * f_tune * t))
            if "closed" in drum_type:
                hp_freq = 7000 + (p_tone * 2000)
                sos_hp = signal.butter(4, hp_freq, 'hp', fs=SR, output='sos')
                sig = signal.sosfilt(sos_hp, sig)
                attack = np.minimum(t * 1500, 1.0) 
                decay_rate = 50 + ((1.0 - p_decay) * 150)
                y = sig * attack * np.exp(-t * decay_rate) * 0.6
            else:
                hp_freq = 6000 + (p_tone * 2000)
                sos_hp = signal.butter(4, hp_freq, 'hp', fs=SR, output='sos')
                sig = signal.sosfilt(sos_hp, sig)
                decay = 10 + ((1.0 - p_decay) * 35)
                y = sig * np.exp(-t * decay)
                y = np.tanh(y * 1.0) * 0.7
            sos_lp = signal.butter(2, 13000, 'lp', fs=SR, output='sos')
            y = signal.sosfilt(sos_lp, y)

        elif drum_type == "clap":
            noise = np.random.uniform(-1, 1, len(t))
            low = 1000 + (p_pitch * 200)
            high = 2400 + (p_pitch * 300)
            filt = signal.sosfilt(signal.butter(2, [low, high], 'bp', fs=SR, output='sos'), noise)
            env = np.zeros_like(t)
            pulse_spacing = 0.009
            for i in range(4):
                start_idx = int(i * pulse_spacing * SR)
                if start_idx >= len(env): break
                amp = 0.8 if i < 3 else 1.0
                remaining = len(env) - start_idx
                local_t = np.linspace(0, remaining/SR, remaining)
                d_val = 300 if i < 3 else (20 + (1.0 - p_decay) * 50)
                pulse_env = np.exp(-local_t * d_val) * amp
                env[start_idx:] = np.maximum(env[start_idx:], pulse_env)
            y = filt * env

        elif drum_type == "wood" or "perc" in drum_type and "a" in drum_type:
            carrier = 800 + (p_pitch * 400)
            mod_f = carrier * 2.41 
            mod_idx = 3.0 * (1.0 - p_tone * 0.5)
            modulator = np.sin(2 * np.pi * mod_f * t) * mod_idx * carrier
            osc = np.sin(2 * np.pi * carrier * t + modulator)
            decay = 80 + ((1.0 - p_decay) * 200)
            y = osc * np.exp(-t * decay)
            sos_hp = signal.butter(2, 400, 'hp', fs=SR, output='sos')
            y = signal.sosfilt(sos_hp, y)

        else:
            f_base = 90 + (p_pitch * 60)
            sweep = np.exp(-t * 15)
            f_inst = f_base * (1.0 + 0.5 * sweep)
            phase = np.cumsum(f_inst) * 2 * np.pi / SR
            y = np.tanh(np.sin(phase) * 1.1)
            click = np.random.uniform(-1, 1, len(t)) * np.exp(-t * 400)
            decay = 8 + ((1.0 - p_decay) * 15)
            y = (y + click * 0.2) * np.exp(-t * decay)

        return SynthEngine.finalize(y)

# --- Synth Engine ---

class SynthEngine:
    KIT_SIMPLE = 0
    KIT_PCM = 1
    KIT_EIGHT = 2
    KIT_NINE = 3
    
    current_kit = KIT_EIGHT

    @staticmethod
    def set_kit(index):
        SynthEngine.current_kit = index % 4

    @staticmethod
    def finalize(data):
        """Standard cleanup for all engines"""
        if len(data) < 200: return data.astype(np.float32) # Early exit, ensure float32
        
        # Declick / Fade out
        fade_len = min(50, len(data) // 4) # Adaptive fade length
        data[-fade_len:] *= np.linspace(1, 0, fade_len)
        
        # High Freq Smooth (Studio Polish)
        sos = signal.butter(1, 18000, 'lp', fs=SR, output='sos')
        data = signal.sosfilt(sos, data)

        peak = np.max(np.abs(data))
        if peak > 0: data /= peak
        return SynthEngine.ensure_zero_crossing(data.astype(np.float32))

    @staticmethod
    def ensure_zero_crossing(data):
        # Keep existing implementation
        if len(data) < 200: return data
        limit = min(len(data) // 4, 2000)
        start_idx = 0
        zc_start = np.where(np.diff(np.sign(data[:limit])))[0]
        if len(zc_start) > 0: start_idx = zc_start[0] + 1
        return data[start_idx:]

    @staticmethod
    def process_sample(raw_data, params):
        if raw_data is None or len(raw_data) == 0:
            return np.zeros(100, dtype=np.float32)

        p_pitch = params.get('pitch', 0.5)
        p_decay = params.get('decay', 0.5)
        p_tone = params.get('tone', 0.5)

        speed = 0.5 + (p_pitch * 1.5)
        new_len = int(len(raw_data) / speed)
        if new_len < 10: new_len = 10
        y = signal.resample(raw_data, new_len)
        t = np.linspace(0, len(y) / SR, len(y))

        if p_tone < 0.45:
            cutoff = 500 + (p_tone * 8000)
            sos = signal.butter(1, cutoff, 'lp', fs=SR, output='sos')
            y = signal.sosfilt(sos, y)
        elif p_tone > 0.55:
            cutoff = 100 + ((p_tone - 0.5) * 4000)
            sos = signal.butter(1, cutoff, 'hp', fs=SR, output='sos')
            y = signal.sosfilt(sos, y)

        decay_coef = 0.5 + ((1.0 - p_decay) * 15)
        env = np.exp(-t * decay_coef)
        y = y * env
        return SynthEngine.finalize(y)

    @staticmethod
    def apply_filter(data, val):
        if 0.45 < val < 0.55: return data
        if val <= 0.45:
            norm = val / 0.45
            cutoff = 150 + (norm**2 * 18000)
            sos = signal.butter(2, cutoff, 'lp', fs=SR, output='sos')
        else:
            norm = (val - 0.55) / 0.45
            cutoff = 20 + (norm**2 * 8000)
            sos = signal.butter(2, cutoff, 'hp', fs=SR, output='sos')
        return signal.sosfilt(sos, data).astype(np.float32)

    @staticmethod
    def generate_drum(drum_type, params):
        try:
            if SynthEngine.current_kit == SynthEngine.KIT_SIMPLE:
                audio = SimpleDrums.generate(drum_type, params)
            elif SynthEngine.current_kit == SynthEngine.KIT_PCM:
                audio = PCMDrums.generate(drum_type, params)
            elif SynthEngine.current_kit == SynthEngine.KIT_NINE:
                audio = NineDrums.generate(drum_type, params)
            else:
                audio = EightDrums.generate(drum_type, params)
            
            # Ensure we always have valid audio data
            if audio is None or len(audio) == 0:
                return SynthEngine.generate_fallback_silence()
            
            return audio
            
        except Exception as e:
            print(f"Error generating {drum_type}: {e}")
            return SynthEngine.generate_fallback_silence()
    
    @staticmethod
    def generate_fallback_silence():
        """Generate a short silence as fallback"""
        return np.zeros(1024, dtype=np.float32)
    
    @staticmethod
    def resample_lofi(data, crush_val):
        if crush_val <= 0.01: return data
        # Gentler reduction (max 6x downsample instead of 20x)
        reduction = 1.0 + (crush_val * 5.0)
        orig_len = len(data)
        target_len = max(1, int(orig_len / reduction))
        lo = signal.resample(data, target_len)
        # Gentler bit reduction (min ~8 bits instead of 3)
        bits = 16 - (crush_val * 8) 
        steps = 2 ** bits
        lo = np.round(lo * steps) / steps
        restored = signal.resample(lo, orig_len).astype(np.float32)
        return SynthEngine.ensure_zero_crossing(np.clip(restored, -1.0, 1.0))
        
    @staticmethod
    def apply_background_reverb(audio_data):
        if len(audio_data) == 0: return audio_data
        wet = np.zeros_like(audio_data)
        
        # 1. Early (40ms)
        d1 = int(SR * 0.04)
        if d1 < len(audio_data):
            r1 = np.roll(audio_data, d1)
            r1[:d1] = 0
            wet += r1 * 0.5 
            
        # 2. Mid (120ms) - Darker
        d2 = int(SR * 0.12)
        if d2 < len(audio_data):
            r2 = np.roll(audio_data, d2)
            r2[:d2] = 0
            sos_lp = signal.butter(1, 3500, 'lp', fs=SR, output='sos')
            r2 = signal.sosfilt(sos_lp, r2)
            wet += r2 * 0.35
            
        # 3. Long (250ms) - Very Dark
        d3 = int(SR * 0.25)
        if d3 < len(audio_data):
            r3 = np.roll(audio_data, d3)
            r3[:d3] = 0
            sos_lp2 = signal.butter(1, 1500, 'lp', fs=SR, output='sos')
            r3 = signal.sosfilt(sos_lp2, r3)
            wet += r3 * 0.20
            
        sos_hp = signal.butter(1, 300, 'hp', fs=SR, output='sos')
        wet = signal.sosfilt(sos_hp, wet)
        
        # Mix 20%
        return audio_data + (wet * 0.20)

class FadeButton(QPushButton):
    def __init__(self, text, parent=None, is_small=False):
        super().__init__(text, parent)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.is_small = is_small
        
        self.hover_fader = HoverFader(self, speed_in=0.2, speed_out=0.1)
        
        # Internal Timer for 60FPS animation
        self.anim_timer = QTimer(self)
        self.anim_timer.setInterval(16)
        self.anim_timer.timeout.connect(self.update_anim)

        if self.is_small:
            self.setFixedSize(30, 18)
            self.base_font = QFont("Segoe UI", 7, QFont.Weight.Bold)
        else:
            self.setFixedSize(90, 26)
            self.base_font = QFont("Segoe UI", 9, QFont.Weight.Bold)

    def update_anim(self):
        if self.hover_fader.update():
            self.update()
        else:
            self.anim_timer.stop()

    def enterEvent(self, e): 
        self.hover_fader.enter()
        self.anim_timer.start()

    def leaveEvent(self, e): 
        self.hover_fader.leave()
        self.anim_timer.start()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        t = self.hover_fader.val
        r = self.rect().adjusted(1, 1, -1, -1)
        
        # Background: White -> Light Blue
        bg_col = QColor(255, 255, 255)
        if t > 0.01:
            bg_col = QColor(
                int(255 + (235 - 255) * t),
                int(255 + (248 - 255) * t),
                int(255 + (255 - 255) * t)
            )

        # Border: Grey -> Blue
        border_col = QColor(203, 213, 224)
        if t > 0.01:
            border_col = QColor(
                int(203 + (144 - 203) * t),
                int(213 + (205 - 213) * t),
                int(224 + (244 - 224) * t)
            )
            
        # Text: Grey -> Blue
        text_col = QColor(160, 174, 192)
        if t > 0.01:
            text_col = QColor(
                int(160 + (49 - 160) * t),
                int(174 + (130 - 174) * t),
                int(192 + (206 - 192) * t)
            )

        painter.setBrush(bg_col)
        painter.setPen(QPen(border_col, 1))
        radius = 13 if not self.is_small else 3
        painter.drawRoundedRect(r, radius, radius)
        
        painter.setPen(text_col)
        painter.setFont(self.base_font)
        painter.drawText(r, Qt.AlignmentFlag.AlignCenter, self.text())

class HoverFader:
    def __init__(self, owner, speed_in=0.25, speed_out=0.1):
        self.val = 0.0
        self.owner = owner
        self.speed_in = speed_in
        self.speed_out = speed_out
        self.hovering = False
        
    def update(self):
        target = 1.0 if self.hovering else 0.0
        if abs(self.val - target) > 0.01:
            step = self.speed_in if target > self.val else self.speed_out
            self.val += (target - self.val) * step
            self.owner.update()
            return True
        elif self.val != target:
            self.val = target
            self.owner.update()
        return False

    def enter(self): self.hovering = True; self.owner.update()
    def leave(self): self.hovering = False; self.owner.update()

class AudioMixer:
    @staticmethod
    def mix_sequence(slots, bpm, swing, clip_val, rev_prob, steps=16):
        sec_beat = 60.0 / bpm
        sec_step = sec_beat / 4.0
        
        total_samples = int(sec_step * steps * SR)
        if total_samples % 2 != 0: total_samples += 1

        swing_offset = int(sec_step * swing * 0.33 * SR)
        
        out = np.zeros(total_samples + int(SR * 0.5), dtype=np.float32)
        
        # 1. Generate Sidechain Map
        sidechain_env = np.ones_like(out)
        for s in slots:
            if "kick" in s.get('label', ''):
                s_pattern = s['pattern']
                duck_len = int(SR * 0.12)
                attack_len = int(SR * 0.005)
                duck_shape = np.ones(duck_len, dtype=np.float32)
                if duck_len > attack_len:
                    duck_shape[:attack_len] = np.linspace(1, 0, attack_len)
                    duck_shape[attack_len:] = np.linspace(0, 1, duck_len - attack_len)
                for i in range(steps):
                    if i < len(s_pattern) and s_pattern[i]:
                        start = int(i * sec_step * SR)
                        if i % 2 != 0: start += swing_offset
                        if start < len(sidechain_env):
                            write_len = min(len(duck_shape), len(sidechain_env) - start)
                            sidechain_env[start:start+write_len] *= duck_shape[:write_len]

        # 2. Process Slots
        for s in slots:
            try:
                raw_data = s['data']
                if raw_data is None or len(raw_data) == 0: continue
                raw_data = np.nan_to_num(raw_data, copy=False)
                is_sliced = s.get('is_sliced', False)
                is_bass = s.get('is_bass', False) 

                # === TRACK BUFFER ===
                # We build the track in isolation first
                track_out = np.zeros_like(out)

                if is_bass:
                    # Bass is pre-sequenced continuous audio, so we just apply sidechain immediately
                    write_len = min(len(raw_data), len(track_out))
                    bass_segment = raw_data[:write_len].copy()
                    bass_segment *= sidechain_env[:write_len]
                    track_out[:write_len] += bass_segment * 0.6
                
                else:
                    # Drums & Reseq Logic
                    s_pattern = s['pattern']
                    s_vels = s['velocities']
                    
                    if not is_sliced:
                        # [One Shot Logic]
                        max_seq_len = total_samples + int(SR * 0.5)
                        if clip_val > 0.0:
                            keep_ratio = 1.0 / (1.0 + (clip_val * 20.0))
                            actual_len = max(150, int(len(raw_data) * keep_ratio))
                            data_fwd = raw_data[:actual_len].copy()
                            fade_samples = min(200, int(actual_len * 0.4))
                            if fade_samples > 0: data_fwd[-fade_samples:] *= np.linspace(1, 0, fade_samples)
                        else:
                            limit = min(len(raw_data), max_seq_len)
                            data_fwd = raw_data[:limit].copy()

                        fade_in = min(100, len(data_fwd) // 10)
                        if fade_in > 0: data_fwd[:fade_in] *= np.linspace(0, 1, fade_in)

                        if rev_prob > 0.0: data_rev = np.ascontiguousarray(data_fwd[::-1])
                        else: data_rev = data_fwd
                        
                        s_len = len(data_fwd)
                        is_kick = "kick" in s.get('label', '')
                        
                        for i in range(steps):
                            if i >= len(s_pattern): break
                            if s_pattern[i]:
                                start_pos = int(i * sec_step * SR)
                                if i % 2 != 0: start_pos += swing_offset
                                if start_pos < len(track_out):
                                    current = data_rev if (rev_prob > 0.0 and np.random.random() < rev_prob) else data_fwd
                                    write_len = min(s_len, len(track_out) - start_pos)
                                    if write_len > 0:
                                        base_gain = 1.0 if is_kick else 0.65
                                        gain = (s_vels[i] ** 1.5) * base_gain
                                        track_out[start_pos:start_pos + write_len] += current[:write_len] * gain
                    else:
                        # [Reseq Logic]
                        # We build the sequence first, THEN apply sidechain below
                        src_step_len = len(raw_data) // steps
                        if src_step_len < 100: continue 

                        for i in range(steps):
                            if i >= len(s_pattern): break
                            if s_pattern[i]:
                                src_start = i * src_step_len
                                src_end = src_start + src_step_len
                                dst_start = int(i * sec_step * SR)
                                if i % 2 != 0: dst_start += swing_offset
                                if src_end > len(raw_data): src_end = len(raw_data)
                                if dst_start >= len(track_out): continue

                                chunk = raw_data[src_start:src_end].copy()
                                chunk *= (s_vels[i] ** 1.5)
                                
                                # Adaptive Fade
                                fade_len = min(200, int(len(chunk) * 0.05))
                                if fade_len > 4:
                                    chunk[:fade_len] *= np.linspace(0, 1, fade_len)
                                    chunk[-fade_len:] *= np.linspace(1, 0, fade_len)

                                write_len = min(len(chunk), len(track_out) - dst_start)
                                if write_len > 0:
                                    track_out[dst_start:dst_start+write_len] += chunk[:write_len] * 0.65

                # === APPLY FX Post-Sequence ===
                if is_sliced:
                    # [SIDECHAIN APPLIED HERE]
                    # We multiply the finished Reseq track by the kick ducking envelope
                    track_out *= sidechain_env
                    
                    # We apply reverb to the CONTINUOUS track buffer.
                    track_out = SynthEngine.apply_background_reverb(track_out)

                # Add to Main Mix
                out += track_out

            except Exception as e:
                print(f"Mix Error: {e}")
                continue

        # Wrap Tail
        tail = out[total_samples:]
        wrap_len = min(len(tail), total_samples)
        out[:wrap_len] += tail[:wrap_len]
        final = out[:total_samples]
        
        peak = np.max(np.abs(final))
        if peak > 0.95: final = np.tanh(final) * 0.95
        
        return final

# --- UI Components ---

class LogoWidget(QWidget):
    kit_changed = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(60, 60)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.grid_size = 4
        self.cell_size = 12
        
        # Initialize with Safe Pastel Blue
        self.current = np.full((4, 4, 3), 240.0) 
        self.targets = np.full((4, 4, 3), 240.0)
        self.randomize_targets()
        
        self.flash_val = 0.0 
        self.kit_index = 2
        self.kit_names = ["simple", "pcm", "eight", "nine"]
        self.text_alpha = 0.0

    def randomize_targets(self):
        # Define 4 subtle variations of the "Pastel Blue/Purple" theme
        # Format: (R_min, R_max), (G_min, G_max), (B_min, B_max)
        # Note: B is always kept high (235+) to ensure the "Blue/Purple" anchor.
        
        palettes = [
            # 1. Pale Lavender (Red bias)
            ([215, 235], [205, 225], [240, 255]),
            # 2. Ice Cyan (Green bias)
            ([195, 215], [225, 245], [240, 255]),
            # 3. Soft Periwinkle (Balanced)
            ([205, 225], [215, 235], [245, 255]),
            # 4. Deep Pale Blue (Slightly darker blue)
            ([190, 210], [210, 230], [240, 255])
        ]
        
        # Shuffle so quadrants swap themes every beat
        # This creates the movement without changing the overall palette
        np.random.shuffle(palettes)
        
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                # Map cell to quadrant (0, 1, 2, 3)
                q_idx = 0
                if r >= self.grid_size // 2: q_idx += 2
                if c >= self.grid_size // 2: q_idx += 1
                
                limits = palettes[q_idx]
                
                # Generate color within strict pastel limits
                red = np.random.randint(limits[0][0], limits[0][1])
                grn = np.random.randint(limits[1][0], limits[1][1])
                blu = np.random.randint(limits[2][0], limits[2][1])
                
                self.targets[r,c] = [red, grn, blu]

    def on_beat(self):
        self.randomize_targets()

    def trigger_flash(self):
        self.flash_val = 0.4

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.kit_index = (self.kit_index + 1) % 4
            self.trigger_flash()
            self.text_alpha = 1.0
            self.kit_changed.emit(self.kit_index)

    def animate(self):
        if self.flash_val > 0.001: self.flash_val *= 0.94
        else: self.flash_val = 0.0
        
        if self.text_alpha > 0.01: self.text_alpha *= 0.95
        
        # DRASTICALLY reduced speed for "easely" fade
        # 0.02 ensures it takes many frames to drift to the new color
        speed = 0.02 
        self.current += (self.targets - self.current) * speed
        
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        grid_w = self.grid_size * self.cell_size
        off_x = (self.width() - grid_w) / 2
        off_y = (self.height() - grid_w) / 2 
        painter.setPen(Qt.PenStyle.NoPen)
        
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                rgb = self.current[r,c].astype(int)
                col = QColor(*rgb)
                if self.flash_val > 0.01: col = col.lighter(int(100 + (self.flash_val * 60)))
                x = off_x + c * self.cell_size
                y = off_y + r * self.cell_size
                rect = QRectF(x + 1, y + 1, self.cell_size - 2, self.cell_size - 2)
                painter.setBrush(col); painter.drawRoundedRect(rect, 3.0, 3.0)

        if self.underMouse() and self.text_alpha < 0.2:
            painter.setPen(QColor(160, 174, 192, 150))
            f = QFont("Segoe UI", 8); f.setBold(True); painter.setFont(f)
            # CHANGED: Adjusted bottom margin from -4 to 2 to move text lower
            painter.drawText(self.rect().adjusted(0,0,0,2), Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignHCenter, "swap")

        if self.text_alpha > 0.01:
            painter.setOpacity(self.text_alpha)
            f = QFont("Segoe UI", 12, QFont.Weight.Bold); painter.setFont(f)
            painter.setPen(QColor("#718096"))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self.kit_names[self.kit_index])

class PlayButton(QWidget):
    clicked = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(90, 26)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._state = 0 
        self.anim_val = 0.0 
        self.icon_morph = 0.0 
        
        self.hover_fader = HoverFader(self, speed_in=0.2, speed_out=0.1)
        
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.animate)
        self.timer.start(20)

    def set_playing(self, state):
        self._state = 1 if state else 0
        self.update()

    def mousePressEvent(self, e): self.clicked.emit()
    def enterEvent(self, e): self.hover_fader.enter()
    def leaveEvent(self, e): self.hover_fader.leave()

    def animate(self):
        target_anim = 0.0 if self._state == 0 else 1.0
        if abs(self.anim_val - target_anim) > 0.001:
            self.anim_val += (target_anim - self.anim_val) * 0.25
        
        target_morph = 1.0 if self._state == 1 else 0.0
        if abs(self.icon_morph - target_morph) > 0.001:
            self.icon_morph += (target_morph - self.icon_morph) * 0.25
            
        # Drive the hover fade here since we already have a timer running
        self.hover_fader.update()
        
        self.update()

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        r = self.rect()
        c = QRectF(r).center()
        
        self.hover_fader.update()
        t = self.hover_fader.val

        # Background: #e2e8f0 -> #cbd5e0
        # Interpolate
        bg_r = int(226 + (203 - 226) * t)
        bg_g = int(232 + (213 - 232) * t)
        bg_b = int(240 + (224 - 240) * t)
        
        p.setBrush(QColor(bg_r, bg_g, bg_b))
        p.setPen(Qt.PenStyle.NoPen)
        p.drawRoundedRect(r, 13, 13)
        
        # Foreground (Icon)
        fg = QColor("#4a5568")
        
        # Icon Morph
        p.translate(c)
        scale_anim = 0.8 + (0.2 * self.anim_val)
        p.scale(scale_anim, scale_anim)
        morph_t = self.icon_morph
        
        p1x = -3.0 * (1.0-morph_t) + (-5.0) * morph_t; p1y = -5.0 
        p2x = -3.0 * (1.0-morph_t) + (-5.0) * morph_t; p2y = 5.0 
        p3x = 6.0 * (1.0-morph_t) + (-2.0) * morph_t; p3y = 0.0 * (1.0-morph_t) + (5.0) * morph_t
        p4x = 6.0 * (1.0-morph_t) + (-2.0) * morph_t; p4y = 0.0 * (1.0-morph_t) + (-5.0) * morph_t
        
        path = QPainterPath()
        path.moveTo(p1x, p1y); path.lineTo(p2x, p2y); path.lineTo(p3x, p3y); path.lineTo(p4x, p4y)
        path.closeSubpath()
        p.setBrush(fg); p.drawPath(path)
        
        if morph_t > 0.01:
            current_alpha = int(255 * morph_t)
            fg.setAlpha(current_alpha)
            p.setBrush(fg)
            off = (1.0 - morph_t) * 2.0
            p.drawRect(QRectF(2.0 + off, -5.0, 3.0, 10.0))

class ClearButton(QPushButton):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(20, 20)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.hover_fader = HoverFader(self, speed_in=0.2, speed_out=0.1)

    def enterEvent(self, e): self.hover_fader.enter()
    def leaveEvent(self, e): self.hover_fader.leave()
        
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.hover_fader.update()
        t = self.hover_fader.val
        
        s = min(self.width(), self.height()); margin = 1
        rect = QRectF(margin, margin, s - (margin * 2), s - (margin * 2))
        
        # Background Fade
        bg = QColor(255, 255, 255, 0)
        if t > 0.01: bg = QColor("#fff5f5")

        # Border: Increased Base Opacity (80 -> 140)
        r = int(254 + (229 - 254) * t)
        g = int(178 + (62 - 178) * t)
        b = int(178 + (62 - 178) * t)
        a = int(140 + (255 - 140) * t) # Base 140
        border_c = QColor(r, g, b, a)

        painter.setBrush(bg); painter.setPen(QPen(border_c, 1))
        painter.drawEllipse(rect)
        
        # Dot: Increased Base Opacity (80 -> 140)
        dot_c = QColor(229, 62, 62) 
        dot_c.setAlpha(int(140 + (115 * t))) # Base 140
            
        painter.setBrush(dot_c); painter.setPen(Qt.PenStyle.NoPen)
        cx, cy = s / 2.0, s / 2.0; r = s / 7.0 
        painter.drawEllipse(QPointF(cx, cy), r, r)

class RedFadeButton(QPushButton):
    def __init__(self, text, parent=None):
        super().__init__(text, parent)
        self.setFixedSize(90, 26)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.hover_fader = HoverFader(self, speed_in=0.2, speed_out=0.1)
        self.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        
        self.anim_timer = QTimer(self)
        self.anim_timer.setInterval(16)
        self.anim_timer.timeout.connect(self.update_anim)

    def update_anim(self):
        if self.hover_fader.update(): self.update()
        else: self.anim_timer.stop()

    def enterEvent(self, e): 
        self.hover_fader.enter()
        self.anim_timer.start()

    def leaveEvent(self, e): 
        self.hover_fader.leave()
        self.anim_timer.start()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        t = self.hover_fader.val
        r = self.rect().adjusted(1, 1, -1, -1)
        
        # Background: Subtle Red (Default) -> Richer Red (Hover)
        # Default: #fff1f2 (Rose-50, very subtle red)
        # Hover:   #ffe4e6 (Rose-100)
        
        start_bg = QColor(255, 255, 255)
        end_bg   = QColor(254, 226, 226)
        
        r_val = int(start_bg.red() + (end_bg.red() - start_bg.red()) * t)
        g_val = int(start_bg.green() + (end_bg.green() - start_bg.green()) * t)
        b_val = int(start_bg.blue() + (end_bg.blue() - start_bg.blue()) * t)
        bg_col = QColor(r_val, g_val, b_val)

        # Border: Soft Red -> Stronger Red
        # Default: #fecdd3
        start_border = QColor(254, 205, 211)
        end_border   = QColor(244, 63, 94) # Rose-500
        
        br = int(start_border.red() + (end_border.red() - start_border.red()) * t)
        bg = int(start_border.green() + (end_border.green() - start_border.green()) * t)
        bb = int(start_border.blue() + (end_border.blue() - start_border.blue()) * t)
        
        # Text: Rose-400 -> Rose-600
        start_text = QColor(247, 150, 165)
        end_text   = QColor(225, 29, 72)
        
        tr = int(start_text.red() + (end_text.red() - start_text.red()) * t)
        tg = int(start_text.green() + (end_text.green() - start_text.green()) * t)
        tb = int(start_text.blue() + (end_text.blue() - start_text.blue()) * t)

        painter.setBrush(bg_col)
        painter.setPen(QPen(QColor(br, bg, bb), 1))
        painter.drawRoundedRect(r, 13, 13)
        painter.setPen(QColor(tr, tg, tb))
        painter.drawText(r, Qt.AlignmentFlag.AlignCenter, self.text())

class ArrowButton(QPushButton):
    def __init__(self, direction, parent=None):
        super().__init__(parent)
        self.direction = direction # 'left' or 'right'
        # Increased height to 22px to match labels
        self.setFixedSize(14, 22)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.hover_fader = HoverFader(self, speed_in=0.2, speed_out=0.1)

    def enterEvent(self, e): self.hover_fader.enter()
    def leaveEvent(self, e): self.hover_fader.leave()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.hover_fader.update()
        t = self.hover_fader.val
        r = self.rect().adjusted(0, 0, -1, -1)

        # Background/Border Logic
        bg_col = QColor(255, 255, 255)
        if t > 0.01: 
            target = QColor("#ebf8ff")
            bg_col = QColor(int(255+(target.red()-255)*t), int(255+(target.green()-255)*t), int(255+(target.blue()-255)*t))
        
        border_col = QColor("#cbd5e0")
        if t > 0.01:
             target_b = QColor("#90cdf4")
             border_col = QColor(int(203+(144-203)*t), int(213+(205-213)*t), int(224+(244-224)*t))

        painter.setBrush(bg_col)
        painter.setPen(QPen(border_col, 1))
        painter.drawRoundedRect(r, 2, 2)

        # Draw Arrow
        arrow_col = QColor("#9ba5b2")
        if t > 0.01:
             target_a = QColor("#3182ce")
             arrow_col = QColor(int(160+(49-160)*t), int(174+(130-174)*t), int(192+(206-192)*t))

        painter.setBrush(arrow_col)
        painter.setPen(Qt.PenStyle.NoPen)
        
        # Centering Logic
        cx, cy = r.center().x(), r.center().y()
        path = QPainterPath()
        
        # Size radius (distance from center to edge of arrow)
        s_x = 2.0 
        s_y = 3.0
        
        if self.direction == 'left':
            # Tip at left, Base at right
            path.moveTo(cx - s_x, cy)           # Tip
            path.lineTo(cx + s_x, cy - s_y)     # Top Base
            path.lineTo(cx + s_x, cy + s_y)     # Bottom Base
        else:
            # Tip at right, Base at left
            path.moveTo(cx + s_x, cy)           # Tip
            path.lineTo(cx - s_x, cy - s_y)     # Top Base
            path.lineTo(cx - s_x, cy + s_y)     # Bottom Base
            
        painter.drawPath(path)

class StepPad(QWidget):
    # Updated Signals: include Index (int)
    toggled = pyqtSignal(int, bool, float)
    velocity_changed = pyqtSignal(int, float)

    def __init__(self, index, base_hue, parent=None):
        super().__init__(parent)
        self.index = index # This index is updated by the Main Window when switching views
        self.base_hue = base_hue 
        self.active = False
        self.velocity = 0.8
        self.playhead_opacity = 0.0
        self.is_downbeat = (index % 4 == 0)

        # Animation State
        self.active_level = 0.0
        self.hover_level = 0.0
        self.is_hovering = False
        
        self.anim_timer = QTimer(self)
        self.anim_timer.setInterval(20)
        self.anim_timer.timeout.connect(self.update_anim)

        self.setFixedWidth(28)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setMouseTracking(True)
        self.is_drag_active = False

    def force_visual_state(self, is_active, velocity):
        """Instantly sets state and KILLS animation to prevent ghosting."""
        self.anim_timer.stop()
        self.active = is_active
        self.velocity = velocity
        self.active_level = 1.0 if is_active else 0.0
        self.hover_level = 1.0 if self.underMouse() else 0.0
        self.update()

    def update_anim(self):
        changed = False
        target_active = 1.0 if self.active else 0.0
        if abs(self.active_level - target_active) > 0.01:
            step = 0.25
            self.active_level += (target_active - self.active_level) * step
            changed = True
        else:
            self.active_level = target_active

        target_hover = 1.0 if self.is_hovering else 0.0
        if abs(self.hover_level - target_hover) > 0.01:
            self.hover_level += (target_hover - self.hover_level) * 0.2
            changed = True
        else:
            self.hover_level = target_hover

        if changed: self.update()
        else: self.anim_timer.stop()

    def enterEvent(self, e): 
        self.is_hovering = True
        if not self.anim_timer.isActive(): self.anim_timer.start()

    def leaveEvent(self, e): 
        self.is_hovering = False
        if not self.anim_timer.isActive(): self.anim_timer.start()

    def set_playing_pos(self, playhead_float):
        # Normalize our absolute data index (0-31) to local visual space (0-16)
        local_idx = self.index % 16
        
        # Calculate distance
        dist = abs(local_idx - playhead_float)
        
        # Highlight if close
        if dist < 0.8:
            new_op = 1.0 - (dist / 0.8)**2
            if abs(self.playhead_opacity - new_op) > 0.05:
                self.playhead_opacity = new_op
                self.update()
        elif self.playhead_opacity > 0.01:
            self.playhead_opacity = 0.0
            self.update()

    def process_mouse_input(self, local_pos, force_state=None, trigger_signal=True):
        h = self.height()
        y = max(0.0, min(float(h), local_pos.y()))
        new_vel = max(0.1, min(1.0, 1.0 - (y / h)))
        
        should_emit = False

        if force_state is not None:
            if self.active != force_state:
                self.active = force_state
                if self.active: self.velocity = new_vel
                self.active_level = 1.0 if self.active else 0.0
                should_emit = True
            elif self.active:
                if abs(new_vel - self.velocity) > 0.01:
                    self.velocity = new_vel
                    should_emit = True
        else:
            self.active = not self.active
            if self.active: self.velocity = new_vel
            self.active_level = 1.0 if self.active else 0.0
            should_emit = True
            
        self.update()
        
        if trigger_signal and should_emit:
            # PASS SELF.INDEX (The true data index)
            if force_state is None:
                self.toggled.emit(self.index, self.active, self.velocity)
            else:
                self.velocity_changed.emit(self.index, self.velocity)
                self.toggled.emit(self.index, self.active, self.velocity)

    def mousePressEvent(self, event):
        if event.button() in (Qt.MouseButton.LeftButton, Qt.MouseButton.RightButton):
            self.drag_start_pos = event.position() if hasattr(event, 'position') else event.localPos()
            self.was_active_at_press = self.active
            self.is_drag_active = False
            
            if not self.active:
                self.process_mouse_input(event.pos(), force_state=True, trigger_signal=True)
            
            if self.parent() and hasattr(self.parent(), 'start_painting'):
                target = not self.was_active_at_press if self.was_active_at_press else True
                self.parent().start_painting(target)

    def mouseMoveEvent(self, event):
        if event.buttons() & (Qt.MouseButton.LeftButton | Qt.MouseButton.RightButton):
            if not self.is_drag_active:
                cur = event.position() if hasattr(event, 'position') else event.localPos()
                if (cur - self.drag_start_pos).manhattanLength() > 5:
                    self.is_drag_active = True
            
            if self.rect().contains(event.pos()):
                if self.is_drag_active:
                    self.process_mouse_input(event.pos(), force_state=True, trigger_signal=False)
            else:
                parent = self.parent()
                if parent:
                    pos_in_parent = self.mapTo(parent, event.pos())
                    child = parent.childAt(pos_in_parent)
                    if isinstance(child, StepPad) and child != self:
                         if hasattr(parent, 'paint_state'):
                            child.process_mouse_input(child.mapFrom(parent, pos_in_parent), force_state=parent.paint_state, trigger_signal=True)

    def mouseReleaseEvent(self, event):
        if self.is_drag_active and self.active:
             if self.rect().contains(event.pos()):
                 self.velocity_changed.emit(self.index, self.velocity)
        elif not self.is_drag_active and self.was_active_at_press and self.active:
             if self.rect().contains(event.pos()):
                self.process_mouse_input(event.pos(), force_state=False, trigger_signal=True)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        r = self.rect().adjusted(1, 1, -1, -1)
        
        c_inactive = QColor("#e2e8f0") if self.is_downbeat else QColor("#edf2f7")
        
        if self.hover_level > 0.01:
            c_hover = QColor("#cbd5e0")
            c_inactive = QColor(
                int(c_inactive.red() + (c_hover.red() - c_inactive.red()) * self.hover_level),
                int(c_inactive.green() + (c_hover.green() - c_inactive.green()) * self.hover_level),
                int(c_inactive.blue() + (c_hover.blue() - c_inactive.blue()) * self.hover_level)
            )

        if self.active_level > 0.01:
            t = time.time() * 0.5 
            hue_offset = np.sin(t + (self.index * 0.2)) * 12
            current_hue = int((self.base_hue + hue_offset) % 360)
            c_active = QColor.fromHsl(current_hue, 160, 140)
            
            final_bg = QColor(
                int(c_inactive.red() + (c_active.red() - c_inactive.red()) * self.active_level),
                int(c_inactive.green() + (c_active.green() - c_inactive.green()) * self.active_level),
                int(c_inactive.blue() + (c_active.blue() - c_inactive.blue()) * self.active_level)
            )
        else:
            final_bg = c_inactive

        painter.setBrush(final_bg)
        painter.setPen(QPen(QColor("#cbd5e0"), 1))
        painter.drawRoundedRect(r, 3, 3)

        if self.active_level > 0.1:
            ly = max(r.y()+2, min(r.bottom()-2, int(r.y() + r.height() * (1.0 - self.velocity))))
            alpha = int(200 * self.active_level)
            painter.setPen(QPen(QColor(255, 255, 255, alpha), 1))
            painter.drawLine(r.x()+4, ly, r.right()-4, ly)

        if self.playhead_opacity > 0.01:
             alpha = int(self.playhead_opacity * 40)
             painter.setBrush(QColor(148, 163, 184, alpha))
             painter.setPen(Qt.PenStyle.NoPen)
             painter.drawRoundedRect(r, 3, 3)

class DraggableLabel(QLabel):
    file_dropped = pyqtSignal(str)

    def __init__(self, text, parent=None):
        super().__init__(text, parent)
        self.setAcceptDrops(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setToolTip("Drag & Drop or Click to load audio file")
        
        self.hover_fader = HoverFader(self, speed_in=0.2, speed_out=0.1)
        font = QFont("Segoe UI", 9, QFont.Weight.Bold)
        self.setFont(font)
        self.setStyleSheet("") 
        
        # Fixed size: Wide enough for text, separated from neighbors
        self.setFixedSize(40, 40)

        # Timer for smooth animation
        self.anim_timer = QTimer(self)
        self.anim_timer.setInterval(20)
        self.anim_timer.timeout.connect(self.update_anim)

    def update_anim(self):
        if self.hover_fader.update():
            self.update()
        else:
            self.anim_timer.stop()

    def enterEvent(self, e): 
        self.hover_fader.enter()
        self.anim_timer.start()

    def leaveEvent(self, e): 
        self.hover_fader.leave()
        self.anim_timer.start()

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        if urls:
            path = urls[0].toLocalFile()
            self.file_dropped.emit(path)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls(): event.accept()
        else: event.ignore()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            fname, _ = QFileDialog.getOpenFileName(self, "open", "", "Audio Files (*.wav *.mp3 *.aif *.flac)")
            if fname: self.file_dropped.emit(fname)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Disable clipping to prevent anti-aliasing cut-off at the edge
        painter.setClipping(False)
        
        t = self.hover_fader.val
        
        # Background
        if t > 0.01:
            bg_col = QColor(235, 248, 255, int(255 * t))
            painter.setBrush(bg_col)
            border_col = QColor(144, 205, 244, int(255 * t))
            painter.setPen(QPen(border_col, 1))
            painter.drawRoundedRect(self.rect().adjusted(0,0,-1,-1), 4, 4)
            
        # Text
        r = int(113 + (49 - 113) * t)
        g = int(128 + (130 - 128) * t)
        b = int(150 + (206 - 150) * t)
        
        painter.setPen(QColor(r, g, b))
        
        # FIX: Reset adjustment to 0 to prevent clipping off-screen.
        # AlignLeft puts it at x=0. setClipping(False) ensures edges aren't sharp-cut.
        rect = self.rect().adjusted(0, 0, 0, 0)
        painter.drawText(rect, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, self.text())
    
class GradientLabel(QWidget):
    clicked = pyqtSignal()

    def __init__(self, text, parent=None):
        super().__init__(parent)
        self.text = text
        self.phase = 0.0
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.animate)
        self.timer.start(50)
        
        font = QFont("Segoe UI", 9, QFont.Weight.Bold)
        self.setFont(font)
        
        # Fixed size matching DraggableLabel
        self.setFixedSize(46, 22)

        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Preferred)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def mousePressEvent(self, event):
        self.clicked.emit()

    def animate(self):
        self.phase = (self.phase + 0.005) % 1.0
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        grad = QLinearGradient(0, 0, self.width(), 0)
        c1 = QColor.fromHslF(self.phase, 0.6, 0.7)
        c2 = QColor.fromHslF((self.phase + 0.3) % 1.0, 0.6, 0.7)
        grad.setColorAt(0, c1)
        grad.setColorAt(1, c2)
        
        painter.setPen(QPen(QBrush(grad), 0))
        
        # Adjusted left +4 pixels to align with the drum labels above
        r = self.rect().adjusted(4, 0, 0, 0)
        painter.drawText(r, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, self.text)

class SplitPill(QWidget):
    view_changed = pyqtSignal(int) 

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(90, 26)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.view_idx = 0     
        self.is_extended = False 
        self.hover_side = -1 
        self.setMouseTracking(True)
        self.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        self.hover_fader = HoverFader(self, speed_in=0.2, speed_out=0.1)

    def set_extended_state(self, is_extended):
        if self.is_extended != is_extended:
            self.is_extended = is_extended
            self.update()

    def enterEvent(self, event): self.hover_fader.enter(); self.update()
    def leaveEvent(self, event): self.hover_fader.leave(); self.hover_side = -1; self.update()

    def mouseMoveEvent(self, event):
        mid = self.width() / 2
        self.hover_side = 0 if event.pos().x() < mid else 1
        self.update()

    def mousePressEvent(self, event):
        mid = self.width() / 2
        self.view_idx = 0 if event.pos().x() < mid else 1
        self.view_changed.emit(self.view_idx)
        self.update()

    def paintEvent(self, event):
        p = QPainter(self); p.setRenderHint(QPainter.RenderHint.Antialiasing)
        r = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        w, h = r.width(), r.height(); mid = w / 2

        self.hover_fader.update(); t = self.hover_fader.val

        path = QPainterPath(); path.addRoundedRect(r, 13, 13)
        p.setBrush(QColor("white")); p.setPen(Qt.PenStyle.NoPen); p.drawPath(path)

        if self.hover_side != -1:
            p.setClipPath(path)
            h_rect = QRectF(0, 0, mid, h) if self.hover_side == 0 else QRectF(mid, 0, mid, h)
            p.setBrush(QColor(66, 153, 225, 20)); p.drawRect(h_rect); p.setClipping(False)

        p.setClipPath(path)
        active_rect = QRectF(0, 0, mid, h) if self.view_idx == 0 else QRectF(mid, 0, mid, h)
        p.setBrush(QColor("#ebf8ff")); p.drawRect(active_rect); p.setClipping(False)

        border_r = int(203 + (144 - 203) * t)
        border_g = int(213 + (205 - 213) * t)
        border_b = int(224 + (244 - 224) * t)
        p.setBrush(Qt.BrushStyle.NoBrush); p.setPen(QPen(QColor(border_r, border_g, border_b), 1))
        p.drawPath(path); p.drawLine(int(mid), 0, int(mid), int(h))

        c_active = QColor("#3182ce")
        inact_r = int(113 + (49 - 113) * t)
        inact_g = int(128 + (130 - 128) * t)
        inact_b = int(150 + (206 - 150) * t)
        c_inactive = QColor(inact_r, inact_g, inact_b)
        
        p.setPen(c_active if self.view_idx == 0 else c_inactive)
        p.drawText(QRectF(0, 0, mid, h), Qt.AlignmentFlag.AlignCenter, "1")

        col_2 = c_active if self.view_idx == 1 else c_inactive
        if not self.is_extended and self.view_idx != 1: col_2.setAlpha(100)
        else: col_2.setAlpha(255)
        p.setPen(col_2)
        p.drawText(QRectF(mid, 0, mid, h), Qt.AlignmentFlag.AlignCenter, "2")

class PlaceholderButton(QPushButton):
    def __init__(self, parent=None):
        super().__init__(parent)
        # Matches SlotRow.create_btn dimensions exactly
        self.setFixedSize(30, 18)
        
        # Matches SlotRow.create_btn stylesheet exactly
        self.setStyleSheet("""
            QPushButton { 
                background: white; 
                border: 1px solid #cbd5e0; 
                border-radius: 3px;
                margin: 0px;
                padding: 0px;
            }
        """)

class ReseqWaveform(QWidget):
    pattern_changed = pyqtSignal()
    file_dropped = pyqtSignal(str)

    def __init__(self, steps=16, parent=None): 
        super().__init__(parent)
        self.steps = steps
        # Initialize 32 steps (2 bars) immediately
        self.pattern = [False] * 32
        self.velocities = [0.8] * 32
        self.active_levels = [0.0] * 32
        
        self.view_offset = 0
        self.hue_phase = 0.0 
        
        self.waveform_data = None
        self.setAcceptDrops(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setMouseTracking(True)
        self.setFixedWidth(448)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)
        
        self.playhead_pos = -10.0
        self.hover_step = -1
        self.cached_poly = None
        
        self.is_dragging = False
        self.drag_start_y = 0
        self.drag_state = True
        self.last_step_idx = -1
        self.initial_vel_at_click = 0.8

    def animate_visuals(self):
        # Slower Hue
        self.hue_phase += 0.2 
        if self.hue_phase > 360: self.hue_phase = 0.0
        
        changed = False
        # Iterate over ALL available steps (32), not just the visible 16
        for i in range(len(self.pattern)):
            # Ensure active_levels has capacity (safety check)
            while len(self.active_levels) <= i: self.active_levels.append(0.0)
            
            target = 1.0 if self.pattern[i] else 0.0
            current = self.active_levels[i]
            
            if abs(current - target) > 0.01:
                speed = 0.2 if target > current else 0.08
                self.active_levels[i] += (target - current) * speed
                changed = True
            else:
                self.active_levels[i] = target
        
        if changed or self.hue_phase is not None:
            self.update()

    def set_data(self, data):
        if data is None: 
            self.waveform_data = None
            self.cached_poly = None
        else:
            target_width = 800
            step = max(1, len(data) // target_width)
            self.waveform_data = data[::step]
            mx = np.max(np.abs(self.waveform_data))
            if mx > 0: self.waveform_data /= mx
            self.cached_poly = None
        self.update()

    def set_playing_pos(self, step_float):
        self.playhead_pos = step_float
        self.update()

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls(): event.accept()
        else: event.ignore()

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        if urls: self.file_dropped.emit(urls[0].toLocalFile())

    def mousePressEvent(self, event):
        if self.waveform_data is None:
            fname, _ = QFileDialog.getOpenFileName(self, "load sample", "", "Audio (*.wav *.mp3 *.flac)")
            if fname: self.file_dropped.emit(fname)
            return

        w = self.width()
        step_w = w / self.steps
        # Calculate index with offset (0-15 + offset)
        idx = int(event.pos().x() / step_w) + self.view_offset
        
        if 0 <= idx < len(self.pattern):
            self.pattern[idx] = not self.pattern[idx]
            self.drag_state = self.pattern[idx]
            
            self.is_dragging = True
            self.last_step_idx = idx
            self.drag_start_y = event.pos().y()
            self.initial_vel_at_click = self.velocities[idx]
            
            self.pattern_changed.emit()
            self.update()

    def mouseMoveEvent(self, event):
        w = self.width()
        step_w = w / self.steps
        
        # Visual Hover (0-15) for UI feedback
        visual_idx = int(event.pos().x() / step_w)
        if visual_idx != self.hover_step:
            self.hover_step = visual_idx
            self.update()

        # Data Index (0-31) for logic
        idx = visual_idx + self.view_offset

        if self.is_dragging and 0 <= idx < len(self.pattern):
            if idx != self.last_step_idx:
                self.pattern[idx] = self.drag_state
                self.drag_start_y = event.pos().y() 
                self.initial_vel_at_click = self.velocities[idx]
                self.last_step_idx = idx
                self.pattern_changed.emit()
            
            if self.pattern[idx]:
                dy = self.drag_start_y - event.pos().y()
                vel_delta = dy / 200.0 
                new_vel = np.clip(self.initial_vel_at_click + vel_delta, 0.1, 1.0)
                if abs(new_vel - self.velocities[idx]) > 0.01:
                    self.velocities[idx] = new_vel
                    self.update()

    def mouseReleaseEvent(self, event):
        self.is_dragging = False
        self.pattern_changed.emit()

    def leaveEvent(self, event):
        self.hover_step = -1
        self.update()

    def resizeEvent(self, event):
        self.cached_poly = None
        super().resizeEvent(event)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        r = self.rect().adjusted(1, 1, -1, -1)
        w, h = r.width(), r.height()
        
        # 1. Background Frame
        painter.setBrush(QColor("#edf2f7"))
        painter.setPen(QPen(QColor("#cbd5e0"), 1))
        painter.drawRoundedRect(r, 3, 3)

        # 2. Empty State Text
        if self.waveform_data is None:
            painter.setPen(QColor("#9ba5b2"))
            painter.setFont(QFont("Segoe UI", 9))
            painter.drawText(r, Qt.AlignmentFlag.AlignCenter, "drag audio here")
            return

        # 3. Waveform Generation (Cached)
        # Note: The waveform represents the SOURCE sample, so it remains static 
        # in the background regardless of which bar (1 or 2) is being sequenced.
        if self.cached_poly is None and self.waveform_data is not None:
            pts = []
            cy = h / 2
            amp = (h / 2) * 0.95
            x_step = w / len(self.waveform_data)
            
            pts.append(QPointF(0, cy))
            for i, val in enumerate(self.waveform_data):
                pts.append(QPointF(i * x_step, cy + abs(val) * amp))
            pts.append(QPointF(w, cy))
            
            for i in range(len(self.waveform_data)-1, -1, -1):
                val = self.waveform_data[i]
                pts.append(QPointF(i * x_step, cy - abs(val) * amp))
            
            self.cached_poly = QPolygonF(pts)

        # 4. Draw Waveform (Gradient)
        painter.save()
        painter.translate(r.topLeft())

        if self.cached_poly:
            path = QPainterPath()
            path.addRoundedRect(0, 0, w, h, 3, 3)
            painter.setClipPath(path)
            
            grad = QLinearGradient(0, 0, w, 0)
            h1 = (self.hue_phase) % 360
            h2 = (self.hue_phase + 60) % 360
            c1 = QColor.fromHsl(int(h1), 180, 210) 
            c2 = QColor.fromHsl(int(h2), 180, 200)
            grad.setColorAt(0, c1) 
            grad.setColorAt(1, c2) 
            
            painter.setBrush(QBrush(grad))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawPolygon(self.cached_poly)

        # 5. Draw Steps (Pattern Overlay)
        step_w = w / self.steps
        # Get the offset (0 or 16), default to 0 if not yet set
        offset = getattr(self, 'view_offset', 0) 
        
        for i in range(self.steps):
            # Calculate the actual index in the data arrays
            data_idx = i + offset
            
            # Boundary check
            if data_idx >= len(self.active_levels): break
            
            x = i * step_w
            anim_lvl = self.active_levels[data_idx]
            rect = QRectF(x, 0, step_w, h)
            
            # A. Inactive/Mask Overlay
            # This dims the waveform parts that aren't active
            mask_alpha = int((1.0 - anim_lvl) * 210)
            if mask_alpha > 0:
                painter.setBrush(QColor(240, 244, 248, mask_alpha))
                painter.setPen(Qt.PenStyle.NoPen)
                painter.drawRect(rect)
                
            # B. Velocity Overlay (White tint on active steps)
            if anim_lvl > 0.01:
                # Retrieve velocity for the specific bar/step
                vel = self.velocities[data_idx] if data_idx < len(self.velocities) else 0.8
                vel_alpha = int((1.0 - vel) * 180 * anim_lvl)
                if vel_alpha > 0:
                    painter.setBrush(QColor(255, 255, 255, vel_alpha))
                    painter.setPen(Qt.PenStyle.NoPen)
                    painter.drawRect(rect)
            
            # C. Hover Highlight
            if i == self.hover_step and anim_lvl < 0.9:
                painter.setBrush(QColor(255, 255, 255, 100))
                painter.drawRect(rect)
            
            # D. Grid Lines
            if i > 0:
                painter.setPen(QPen(QColor(255, 255, 255, 120), 1))
                painter.drawLine(int(x), 0, int(x), int(h))

        painter.restore()

        # 6. Smooth Playhead Overlay
        # self.playhead_pos is calculated relative to the VISUAL window (0 to 16)
        # by the main window, so no offset logic is needed here.
        if self.playhead_pos >= 0:
            cx = self.playhead_pos * step_w + (step_w / 2)
            bar_w = step_w * 1.2
            r_play = QRectF(cx - (bar_w/2), 0, bar_w, h)
            
            grad = QLinearGradient(r_play.topLeft(), r_play.topRight())
            c_center = QColor(148, 163, 184, 45)
            c_edge = QColor(148, 163, 184, 0)
            
            grad.setColorAt(0, c_edge)
            grad.setColorAt(0.5, c_center)
            grad.setColorAt(1, c_edge)
            painter.setBrush(grad)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawRect(r_play)

class AnimToggle(QPushButton):
    def __init__(self, text, anim_type='pulse', parent=None):
        super().__init__(text, parent)
        self.setCheckable(True)
        self.setFixedSize(30, 18)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.anim_type = anim_type
        self.timer = QTimer(self); self.timer.timeout.connect(self.update); self.timer.start(40)
        self.phase = 0.0
        # FIX: Changed :checked border-color from #9ba5b2 (Grey) to #3182ce (Blue)
        self.setStyleSheet("""
            QPushButton { 
                background: white; border: 1px solid #cbd5e0; border-radius: 3px;
                margin: 0px; padding: 0px; color: #9ba5b2; 
                font-size: 10px; font-weight: bold; font-family: 'Segoe UI';
            }
            QPushButton:hover { background: #ebf8ff; color: #3182ce; border-color: #90cdf4; }
            QPushButton:checked { border-color: #3182ce; color: #3182ce; } 
        """)

    def paintEvent(self, event):
        super().paintEvent(event)
        if not self.isChecked(): return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = self.rect(); self.phase += 0.15
        cx, cy = rect.center().x(), rect.center().y()

        if self.anim_type == 'pulse':
            alpha = int(80 + (np.sin(self.phase) * 40))
            color = QColor(159, 122, 234, alpha) 
            grad = QRadialGradient(cx, cy, rect.width() * 0.6)
            grad.setColorAt(0, color); grad.setColorAt(1, QColor(255, 255, 255, 0))
            painter.setBrush(grad); painter.setPen(Qt.PenStyle.NoPen)
            painter.drawRoundedRect(rect, 3, 3)
            
        elif self.anim_type == 'ripple':
            prog = (self.phase % 4.0) / 4.0
            radius = prog * rect.width()
            alpha = int((1.0 - prog) * 180)
            color = QColor(66, 153, 225, alpha)
            painter.setPen(QPen(color, 2)); painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawEllipse(QPointF(cx, cy), radius, radius * 0.6)

class AnimRevButton(QPushButton):
    def __init__(self, text, parent=None):
        super().__init__(text, parent)
        self.setCheckable(True)
        self.setFixedSize(30, 18)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.timer = QTimer(self); self.timer.timeout.connect(self.update); self.timer.start(25)
        self.scan_pos = 1.0 
        self.setStyleSheet("""
            QPushButton { 
                background: white; border: 1px solid #cbd5e0; border-radius: 3px;
                margin: 0px; padding: 0px; color: #9ba5b2; 
                font-size: 10px; font-weight: bold; font-family: 'Segoe UI';
            }
            QPushButton:hover { background: #ebf8ff; color: #3182ce; border-color: #90cdf4; }
            QPushButton:checked { color: #3182ce; }
        """)

    def paintEvent(self, event):
        super().paintEvent(event)
        if not self.isChecked(): return
        painter = QPainter(self); painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = self.rect(); w, h = rect.width(), rect.height()
        self.scan_pos -= 0.08
        if self.scan_pos < -0.3: self.scan_pos = 1.0
        x_base = self.scan_pos * w
        for i in range(4):
            x = x_base + (i * 2.5)
            if x > w or x < 0: continue
            alpha = 255 if i == 0 else int(160 - (i * 50))
            if alpha < 0: alpha = 0
            pen = QPen(QColor(159, 122, 234, alpha)); pen.setWidthF(1.5 if i == 0 else 1.0)
            painter.setPen(pen); painter.drawLine(QPointF(x, 2), QPointF(x, h-2))

class ReseqEngine:
    @staticmethod
    def process(raw_data, bpm, params, steps=16):
        if raw_data is None or len(raw_data) == 0:
            return np.zeros(int(SR * 2), dtype=np.float32)

        # Calculate target length based on bar count (steps / 16)
        bars = max(1, steps // 16)
        target_len = int((60.0 / bpm) * 4.0 * bars * SR) 
        
        working = raw_data.astype(np.float32)

        # 1. Pitch
        p_pitch = params.get('pitch', 0.5)
        speed = 0.5 + (p_pitch * 1.5)
        if abs(speed - 1.0) > 0.001:
            old_indices = np.arange(len(working))
            new_indices = np.linspace(0, len(working) - 1, int(len(working) / speed))
            working = np.interp(new_indices, old_indices, working).astype(np.float32)

        # 2. Tone
        p_tone = params.get('tone', 0.5)
        if abs(p_tone - 0.5) > 0.05:
            if p_tone < 0.5:
                cutoff = 400 + (p_tone * 10000)
                sos = signal.butter(1, cutoff, 'lp', fs=SR, output='sos')
            else:
                cutoff = 100 + ((p_tone - 0.5) * 5000)
                sos = signal.butter(1, cutoff, 'hp', fs=SR, output='sos')
            working = signal.sosfilt(sos, working)

        # 3. Tiling (Fill the target length)
        if len(working) < target_len:
            fade_edge = min(50, len(working)//8)
            if fade_edge > 0:
                working[:fade_edge] *= np.linspace(0, 1, fade_edge)
                working[-fade_edge:] *= np.linspace(1, 0, fade_edge)

            mirrored = np.concatenate([working, working[::-1]])
            repeats = (target_len // len(mirrored)) + 2
            working = np.tile(mirrored, repeats)

        # Crop to exactly target length
        working = working[:target_len]

        # 4. Loop Seam
        seam_fade = 64
        working[:seam_fade] *= np.linspace(0, 1, seam_fade)
        working[-seam_fade:] *= np.linspace(1, 0, seam_fade)

        # 5. Filter & Crush
        p_filt = params.get('filter', 0.5)
        working = SynthEngine.apply_filter(working, p_filt)
        
        p_crush = params.get('crush', 0.0)
        if p_crush > 0.01:
            working = SynthEngine.resample_lofi(working, p_crush)

        # 6. Trance Gate (Dynamic Steps)
        p_decay = params.get('decay', 0.5)
        if p_decay < 0.9:
            gate_len = target_len // steps
            gate_env = np.ones(gate_len, dtype=np.float32)
            tail = int(gate_len * (1.0 - p_decay))
            if tail > 0:
                gate_env[-tail:] = np.linspace(1, 0, tail)
            
            full_gate = np.tile(gate_env, steps)
            if len(full_gate) > len(working): full_gate = full_gate[:len(working)]
            elif len(full_gate) < len(working): full_gate = np.pad(full_gate, (0, len(working)-len(full_gate)), constant_values=1)
            
            working *= full_gate

        # 7. Normalize
        peak = np.max(np.abs(working))
        if peak > 0.01: 
            working *= (0.6 / peak)
        
        return working

class ReseqRow(QFrame):
    pattern_changed = pyqtSignal()
    preview_req = pyqtSignal(object)
    saved_msg = pyqtSignal(str) 

    def __init__(self, label_text, bpm_ref, parent=None):
        super().__init__(parent)
        self.label_text = label_text
        self.bpm = bpm_ref 
        
        # --- DUAL STORAGE for Side 1 and Side 2 ---
        self.full_sources = [None, None]  
        self.raw_samples = [None, None]   
        # ------------------------------------------
        
        self.current_data = None     
        self.is_reversed = False
        self.mod_active = False
        self.mod_envelopes = [] 
        self.spatial_active = False
        self.spatial_params = {}
        
        self.update_timer = QTimer(self)
        self.update_timer.setSingleShot(True)
        self.update_timer.setInterval(50)
        self.update_timer.timeout.connect(self.process_audio)

        self.layout = QHBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0) 
        self.layout.setSpacing(0)

        # Label
        lbl_frame = QWidget(); lbl_frame.setFixedWidth(55)
        lf_layout = QHBoxLayout(lbl_frame); lf_layout.setContentsMargins(0,0,0,0)
        self.lbl = GradientLabel(label_text)
        self.lbl.clicked.connect(self.clear_sample)
        lf_layout.addWidget(self.lbl, 0, Qt.AlignmentFlag.AlignLeft)
        self.layout.addWidget(lbl_frame)

        # Controls
        ctrl_frame = QWidget(); ctrl_frame.setFixedWidth(205)
        ctrl_main_layout = QHBoxLayout(ctrl_frame); ctrl_main_layout.setContentsMargins(0, 0, 0, 0); ctrl_main_layout.setSpacing(0)

        btn_container = QWidget(); btn_container.setFixedWidth(65)
        btn_grid = QGridLayout(btn_container); btn_grid.setContentsMargins(0, 0, 4, 0)
        btn_grid.setSpacing(1); btn_grid.setVerticalSpacing(1)
        
        self.btn_snap = FadeButton("fit", is_small=True)
        self.btn_snap.clicked.connect(self.snap_to_transient)
        self.btn_rev = AnimRevButton("rev", parent=self)
        self.btn_rev.toggled.connect(self.toggle_reverse)
        self.btn_mod = AnimToggle("flt", anim_type='pulse', parent=self)
        self.btn_mod.toggled.connect(self.toggle_mod)
        self.btn_space = AnimToggle("del", anim_type='ripple', parent=self)
        self.btn_space.toggled.connect(self.toggle_spatial)

        btn_grid.addWidget(self.btn_snap, 0, 0); btn_grid.addWidget(self.btn_rev, 0, 1)
        btn_grid.addWidget(self.btn_mod, 1, 0); btn_grid.addWidget(self.btn_space, 1, 1)
        btn_grid.setAlignment(Qt.AlignmentFlag.AlignVCenter)
        ctrl_main_layout.addWidget(btn_container)

        crush_container = QWidget()
        cc_layout = QHBoxLayout(crush_container); cc_layout.setContentsMargins(2, 0, 2, 0); cc_layout.setSpacing(2)
        
        def make_v_slider(val, tip, hue_offset, def_val=50):
            sl = CircleSlider(Qt.Orientation.Vertical, base_hue=(260 + hue_offset) % 360, default_value=def_val)
            sl.setRange(0, 100); sl.setValue(val); sl.setFixedWidth(20)
            sl.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)
            sl.setToolTip(tip); sl.valueChanged.connect(self.on_fx_change)
            return sl

        self.sl_vol = make_v_slider(85, "Volume", 0, 85)
        self.sl_crush = make_v_slider(0, "Bitcrush", 0, 0)
        self.sl_filt = make_v_slider(50, "Filter", 8, 50)
        self.sl_pitch = make_v_slider(50, "Pitch", 16, 50)
        self.sl_decay = make_v_slider(50, "Gate/Decay", 24, 50)
        self.sl_tone = make_v_slider(50, "Tone", 32, 50)

        cc_layout.addWidget(self.sl_vol); cc_layout.addWidget(self.sl_crush)
        cc_layout.addWidget(self.sl_filt); cc_layout.addWidget(self.sl_pitch)
        cc_layout.addWidget(self.sl_decay); cc_layout.addWidget(self.sl_tone)
        ctrl_main_layout.addWidget(crush_container)
        self.layout.addWidget(ctrl_frame)

        self.wave_viz = ReseqWaveform(steps=16, parent=self)
        self.wave_viz.file_dropped.connect(self.load_sample)
        self.wave_viz.pattern_changed.connect(lambda: self.pattern_changed.emit())
        self.wave_viz.setFixedWidth(448)
        self.wave_viz.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)
        self.layout.addWidget(self.wave_viz)

        self.layout.addSpacing(6)
        self.btn_clr = ClearButton(self)
        self.btn_clr.clicked.connect(self.clear)
        self.layout.addWidget(self.btn_clr)
        self.layout.addStretch(1)
        self.process_audio()
    
    def update_view_state(self):
        side = 0 if self.wave_viz.view_offset < 16 else 1
        self.wave_viz.set_data(self.raw_samples[side])
    
    def schedule_update(self): self.update_timer.start()
    
    @property
    def pattern(self): return self.wave_viz.pattern
    
    @property
    def velocities(self): return self.wave_viz.velocities

    def create_btn(self, text, checkable=False):
        b = QPushButton(text); b.setFixedSize(30, 18); b.setCheckable(checkable)
        b.setCursor(Qt.CursorShape.PointingHandCursor)
        return b
    
    def clear(self):
        self.wave_viz.pattern = [False] * STEPS
        self.pattern_changed.emit(); self.wave_viz.update()
    
    def highlight(self, step_float): self.wave_viz.set_playing_pos(step_float)
    
    def update_sound(self, play=False):
        self.process_audio()
        if play and self.current_data is not None: self.preview_req.emit(self.current_data)

    def load_sample(self, path):
        try:
            side = 0 if self.wave_viz.view_offset < 16 else 1
            
            data, f_sr = sf.read(path)
            if len(data.shape) > 1: data = np.mean(data, axis=1)
            
            if f_sr != SR:
                num_samples = int(len(data) * SR / f_sr)
                if num_samples > SR * 10: 
                    data = data[:f_sr*10] 
                    num_samples = int(len(data) * SR / f_sr)
                data = signal.resample(data, num_samples)
                
            peak = np.max(np.abs(data))
            if peak > 0: data /= peak
            
            self.full_sources[side] = data.astype(np.float32)
            self.full_sources[side] = np.nan_to_num(self.full_sources[side])
            
            self.saved_msg.emit(f"reseq {side+1}: {path.split('/')[-1]}")
            self.snap_to_transient()
        except Exception as e:
            print(e); self.saved_msg.emit("err loading sample")

    def snap_to_transient(self):
        side = 0 if self.wave_viz.view_offset < 16 else 1
        source = self.full_sources[side]
        if source is None: return
        
        try:
            sec_beat = 60.0 / self.bpm; sec_step = sec_beat / 4.0
            target_length = int(sec_step * 16 * SR)
            
            clean_source = np.nan_to_num(source, copy=True)
            source_len = len(clean_source)
            
            # 1. Transient Search (Randomized candidates restored)
            best_start = 0
            min_usable = int(SR * 0.05) 
            limit = source_len - min_usable
            
            if limit > 0:
                max_energy = -1.0
                # FIX: Restored randomization so the button actually does something different each click
                for _ in range(8):
                    cand = np.random.randint(0, limit)
                    window = clean_source[cand : cand + 1024]
                    energy = np.sum(np.abs(window))
                    if energy > max_energy: 
                        max_energy = energy
                        best_start = cand
                
                scan_start = max(0, best_start - 1000)
                scan_window = clean_source[scan_start : best_start + 100]
                zcs = np.where(np.diff(np.sign(scan_window)))[0]
                if len(zcs) > 0:
                    local_peak_idx = 1000 
                    best_zc = zcs[np.argmin(np.abs(zcs - local_peak_idx))]
                    best_start = scan_start + best_zc

            # 2. Extract Logic (Crossfade Loop)
            remaining = source_len - best_start
            
            if remaining >= target_length:
                out_buffer = clean_source[best_start : best_start + target_length]
            else:
                chunk = clean_source[best_start:]
                min_loop = int(SR * 0.05)
                
                if len(chunk) < min_loop:
                    out_buffer = np.zeros(target_length, dtype=np.float32)
                    out_buffer[:len(chunk)] = chunk
                else:
                    xfade_len = min(int(SR * 0.01), len(chunk) // 8)
                    if xfade_len > 0:
                        chunk[:xfade_len] *= np.linspace(0, 1, xfade_len)
                        chunk[-xfade_len:] *= np.linspace(1, 0, xfade_len)
                    
                    repeats = (target_length // len(chunk)) + 2
                    tiled = np.tile(chunk, repeats)
                    out_buffer = tiled[:target_length]

            # 3. Final Cleanup
            out_buffer = np.nan_to_num(out_buffer)
            sos_hp = signal.butter(1, 20, 'hp', fs=SR, output='sos')
            out_buffer = signal.sosfilt(sos_hp, out_buffer)
            
            peak = np.max(np.abs(out_buffer))
            if peak > 0.01: out_buffer *= (0.89 / peak)
            
            self.raw_samples[side] = out_buffer.astype(np.float32)
            self.wave_viz.set_data(self.raw_samples[side])
            self.process_audio()
            self.saved_msg.emit(f"fit sample to length")
            
        except Exception as e:
            print(f"Snap error: {e}")

    def apply_mod_filter(self, audio_data, steps=16):
        if not self.mod_active or not self.mod_envelopes or len(audio_data) == 0: return audio_data
        step_len = len(audio_data) // steps
        if step_len < 10: return audio_data

        current_envs = list(self.mod_envelopes)
        while len(current_envs) < steps:
            current_envs.extend(self.mod_envelopes[:16])
        
        processed_chunks = []
        for i in range(steps):
            start = i * step_len
            end = (i + 1) * step_len if i < steps - 1 else len(audio_data)
            chunk = audio_data[start:end]
            if len(chunk) == 0: continue
            env = current_envs[i]
            freq = (env['start'] + env['end']) * 0.5; freq = np.clip(freq, 50, 16000)
            try:
                if env['type'] == 'lp': sos = signal.butter(2, freq, 'lp', fs=SR, output='sos')
                elif env['type'] == 'hp': sos = signal.butter(2, freq, 'hp', fs=SR, output='sos')
                else: 
                    width = freq * 0.5
                    sos = signal.butter(2, [max(20, freq - width), min(SR/2-1, freq + width)], 'bp', fs=SR, output='sos')
                chunk = signal.sosfilt(sos, chunk)
            except: pass 
            processed_chunks.append(chunk)
        return np.concatenate(processed_chunks)

    def process_audio(self, steps=32, notify=True):
        params = {
            'pitch': self.sl_pitch.value() / 100.0,
            'tone': self.sl_tone.value() / 100.0,
            'filter': self.sl_filt.value() / 100.0,
            'crush': self.sl_crush.value() / 100.0,
            'decay': self.sl_decay.value() / 100.0
        }
        
        while len(self.pattern) < 32: self.pattern.append(False)
        while len(self.velocities) < 32: self.velocities.append(0.8)

        if self.raw_samples[0] is not None:
            res_a = ReseqEngine.process(self.raw_samples[0], self.bpm, params, steps=16)
        else:
            len_a = int((60.0 / self.bpm) * 4.0 * SR) 
            res_a = np.zeros(len_a, dtype=np.float32)

        if self.raw_samples[1] is not None:
            res_b = ReseqEngine.process(self.raw_samples[1], self.bpm, params, steps=16)
        else:
            len_b = int((60.0 / self.bpm) * 4.0 * SR) 
            res_b = np.zeros(len_b, dtype=np.float32)

        if self.is_reversed:
            res_a = res_a[::-1]
            res_b = res_b[::-1]

        processed = np.concatenate([res_a, res_b])

        if self.mod_active: processed = self.apply_mod_filter(processed, steps=32)
        if self.spatial_active: processed = self.apply_spatial_effects(processed)
        
        vol = (self.sl_vol.value() / 100.0) ** 2
        processed *= vol
        
        self.current_data = np.nan_to_num(processed.astype(np.float32), copy=False)
        
        if notify:
            self.pattern_changed.emit()

    def toggle_reverse(self, state):
        self.is_reversed = state; self.schedule_update()
    
    def toggle_mod(self, state):
        self.mod_active = state
        if state:
            self.mod_envelopes = []
            for i in range(32):
                f_type = np.random.choice(['lp', 'bp', 'hp'], p=[0.5, 0.4, 0.1])
                start = np.random.uniform(200, 7000)
                end = np.random.uniform(200, 7000)
                self.mod_envelopes.append({'type': f_type, 'start': start, 'end': end})
        self.schedule_update()

    def toggle_spatial(self, state):
        self.spatial_active = state
        if state:
            self.spatial_params = {'delay_mult': np.random.choice([0.25, 0.5, 0.75]), 
                                   'feedback': np.random.uniform(0.4, 0.7), 'reverb_mix': np.random.uniform(0.25, 0.5)}
        self.schedule_update()

    def clear_sample(self):
        side = 0 if self.wave_viz.view_offset < 16 else 1
        self.full_sources[side] = None
        self.raw_samples[side] = None
        self.wave_viz.set_data(None)
        self.process_audio()
        self.saved_msg.emit(f"reseq {side+1}: cleared")
    
    def on_fx_change(self):
        self.current_data = None; self.process_audio(); self.pattern_changed.emit()
    
    def drift_params(self, amount):
        if amount <= 0.01: return
        targets = [self.sl_crush, self.sl_filt, self.sl_decay, self.sl_tone]
        for sl in targets:
            if not hasattr(sl, '_f_val'): sl._f_val = float(sl.value())
            if not hasattr(sl, '_vel'): sl._vel = np.random.uniform(-0.05, 0.05)
            force = (np.random.random() - 0.5) * 0.04 * amount
            if sl._f_val < 10: force += 0.03 * amount
            elif sl._f_val > 90: force -= 0.03 * amount
            sl._vel += force; sl._vel *= 0.985
            new_val = sl._f_val + sl._vel; sl._f_val = np.clip(new_val, 0.0, 100.0)
            if int(sl._f_val) != sl.value(): sl.setValue(int(sl._f_val))

    def evolve_fx_params(self):
        changed = False
        if self.mod_active:
            for env in self.mod_envelopes:
                drift = np.random.uniform(0.9, 1.1)
                env['start'] = np.clip(env['start'] * drift, 80, 14000)
                env['end'] = np.clip(env['end'] * drift, 80, 14000)
            changed = True
        if self.spatial_active:
            self.spatial_params['feedback'] = np.clip(self.spatial_params['feedback'] + np.random.uniform(-0.04, 0.04), 0.2, 0.75)
            self.spatial_params['reverb_mix'] = np.clip(self.spatial_params['reverb_mix'] + np.random.uniform(-0.03, 0.03), 0.05, 0.45)
            changed = True
        if changed: self.on_fx_change()
    
    def update_bpm(self, new_bpm):
        self.bpm = new_bpm
        self.process_audio(steps=32)

    def apply_spatial_effects(self, audio_data):
        mix = self.spatial_params.get('reverb_mix', 0.4); feedback = self.spatial_params.get('feedback', 0.5)
        delay_seconds = (60.0 / self.bpm) * self.spatial_params.get('delay_mult', 0.5)
        d_samples = int(delay_seconds * SR)
        wet_delay = np.zeros_like(audio_data)
        if d_samples > 0 and len(audio_data) > 0:
            d1 = np.roll(audio_data, d_samples)
            d2 = np.roll(audio_data, d_samples * 2)
            raw_delay = (d1 * feedback) + (d2 * (feedback * 0.8))
            wet_delay = signal.sosfilt(signal.butter(1, [400, 3000], 'bp', fs=SR, output='sos'), raw_delay)
        return audio_data + (wet_delay * mix * 3.5) 

class BassEngine:
    SCALES = {
        'min': [0, 2, 3, 5, 7, 8, 10, 12],
        'maj': [0, 2, 4, 5, 7, 9, 11, 12],
        'pent': [0, 3, 5, 7, 10, 12, 15, 17],
        'chrom': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    }
    
    @staticmethod
    def mto_freq(midi_note):
        return 440.0 * (2.0 ** ((midi_note - 69) / 12.0))

    @staticmethod
    def generate_sequence(pattern, values, bpm, params, steps=16):
        sec_beat = 60.0 / bpm
        sec_step = sec_beat / 4.0
        total_samples = int(sec_step * steps * SR)
        out = np.zeros(total_samples + int(SR), dtype=np.float32)
        
        base_note = 24 
        p_root = base_note + params.get('root_note', 0)
        p_decay = params.get('decay', 0.5)
        p_glide = params.get('glide', 0.0)
        p_release = params.get('release', 0.5) 
        scale_notes = BassEngine.SCALES[params.get('scale_name', 'min')]

        last_freq = -1.0
        
        for i in range(steps):
            if i >= len(pattern): break
            if not pattern[i]: continue
            
            val = values[i]
            note_idx = int(val * (len(scale_notes) - 1))
            note_idx = max(0, min(note_idx, len(scale_notes)-1))
            target_freq = BassEngine.mto_freq(p_root + scale_notes[note_idx])
            
            if last_freq < 0: last_freq = target_freq
            
            duration_scalar = 0.2 + (p_release * 2.3)
            duration = sec_step * duration_scalar
            t = np.linspace(0, duration, int(duration * SR))
            
            if p_glide > 0.01 and abs(target_freq - last_freq) > 0.1:
                slide_speed = 30.0 * (1.0 - p_glide)
                freq_env = target_freq + (last_freq - target_freq) * np.exp(-t * slide_speed)
                phase = np.cumsum(freq_env) * 2 * np.pi / SR
            else:
                phase = 2 * np.pi * target_freq * t

            last_freq = target_freq
            osc = np.sin(phase)
            env_decay = 2.0 + ((1.0 - p_decay) * 25.0) 
            env = np.exp(-t * env_decay)
            
            sig = np.tanh(osc * 1.5 * env)
            cutoff = 300 + (params.get('tone', 0.5) * 5000)
            sig = signal.sosfilt(signal.butter(1, cutoff, 'lp', fs=SR, output='sos'), sig)

            fade_len = min(int(SR * 0.02), len(sig) // 2) 
            if fade_len > 0:
                sig[:64] *= np.linspace(0, 1, 64)
                sig[-fade_len:] *= 0.5 * (1.0 + np.cos(np.linspace(0, np.pi, fade_len)))

            start = int(i * sec_step * SR)
            dest_len = min(len(sig), len(out) - start)
            out[start:start+dest_len] += sig[:dest_len]

        out = out[:total_samples]
        loop_fade = 256
        if len(out) > loop_fade: out[-loop_fade:] *= np.linspace(1, 0, loop_fade)
        peak = np.max(np.abs(out))
        if peak > 0.9: out = (out / peak) * 0.9
        return out.astype(np.float32)

class BassPianoRoll(QWidget):
    pattern_changed = pyqtSignal()

    def __init__(self, steps=16, base_hue=170, parent=None):
        super().__init__(parent)
        self.steps = steps
        self.base_hue = base_hue 
        
        # Initialize 32 steps (2 bars) immediately
        self.pattern = [False] * 32
        self.values = [0.5] * 32
        self.note_levels = [0.0] * 32
        
        self.view_offset = 0 
        self.hover_fader = HoverFader(self)
        
        # Exact width to match 16 * 28px pads
        self.setFixedWidth(448) 
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setMouseTracking(True)
        
        self.hover_step = -1
        self.playhead_pos = -10.0
        self.is_dragging = False
        self.drag_mode = "paint" 
        self.last_idx = -1
        
        self.anim_timer = QTimer(self)
        self.anim_timer.timeout.connect(self.update_anim)
        self.anim_timer.start(20)
    
    def update_anim(self):
        # Update Note Fades
        changed = self.hover_fader.update()
        
        # Iterate over ALL available steps (32), not just the visible 16
        for i in range(len(self.pattern)):
            # Safety capacity check
            while len(self.note_levels) <= i: self.note_levels.append(0.0)
            
            target = 1.0 if self.pattern[i] else 0.0
            if abs(self.note_levels[i] - target) > 0.01:
                # Slower out (0.1), fast in (0.3)
                speed = 0.3 if target > self.note_levels[i] else 0.1
                self.note_levels[i] += (target - self.note_levels[i]) * speed
                changed = True
            else:
                self.note_levels[i] = target

        if changed: self.update()

    def set_playing_pos(self, step_float):
        if abs(self.playhead_pos - step_float) > 0.001:
            self.playhead_pos = step_float
            self.update()

    def mousePressEvent(self, event):
        x_step = self.width() / self.steps
        # Add view_offset to write to the correct bank (A or B)
        idx = int(event.pos().x() / x_step) + self.view_offset
        
        # Ensure we don't go out of bounds (0-31)
        if 0 <= idx < 32:
            # Ensure capacity
            while len(self.pattern) <= idx: self.pattern.append(False)
            while len(self.values) <= idx: self.values.append(0.5)

            self.drag_mode = (event.button() == Qt.MouseButton.LeftButton)
            
            if self.drag_mode and self.pattern[idx]:
                 self.drag_mode = False

            self.pattern[idx] = self.drag_mode
            if self.drag_mode:
                self.update_pitch_from_mouse(idx, event.pos().y())
            
            self.is_dragging = True
            self.last_idx = idx
            
            self.pattern_changed.emit()
            self.update()

    def mouseMoveEvent(self, event):
        if not self.rect().contains(event.pos()):
            if self.hover_step != -1:
                self.hover_step = -1
                self.update()
            return

        x_step = self.width() / self.steps
        # Visual hover index (0-15)
        visual_idx = int(event.pos().x() / x_step)
        
        if visual_idx != self.hover_step:
            self.hover_step = visual_idx
            self.update()

        # Data index (0-31)
        idx = visual_idx + self.view_offset

        if self.is_dragging and 0 <= idx < 32:
            # Ensure capacity
            while len(self.pattern) <= idx: self.pattern.append(False)
            while len(self.values) <= idx: self.values.append(0.5)

            changed = False
            if idx != self.last_idx:
                self.pattern[idx] = self.drag_mode 
                if self.drag_mode:
                    self.update_pitch_from_mouse(idx, event.pos().y(), repaint=False)
                self.last_idx = idx
                changed = True
            else:
                if self.drag_mode:
                    if self.update_pitch_from_mouse(idx, event.pos().y(), repaint=False):
                        changed = True
            
            if changed: self.update()
    
    def leaveEvent(self, event):
        self.hover_fader.leave()
        # FIX: Explicitly clear hover step
        self.hover_step = -1
        self.update()

    def enterEvent(self, e): self.hover_fader.enter()
    
    def mouseReleaseEvent(self, event):
        if self.is_dragging:
            self.is_dragging = False
            # Emit ONLY when user lets go
            self.pattern_changed.emit()
    
    def update_pitch_from_mouse(self, idx, y, repaint=True):
        h = self.height()
        grid_steps = 8
        raw_val = 1.0 - (y / h)
        step_idx = int(raw_val * grid_steps)
        step_idx = max(0, min(step_idx, grid_steps - 1))
        new_val = (step_idx + 0.5) / grid_steps
        
        if abs(self.values[idx] - new_val) > 0.001:
            self.values[idx] = new_val
            if repaint: self.update()
            return True
        return False

    def update_sound(self, play=False): self.process_sequence()
    
    def clear(self):
            # Target the currently visible bank (0 or 16)
            offset = self.view_offset
            
            # Ensure capacity
            while len(self.pattern) < offset + 16: self.pattern.append(False)
            while len(self.note_levels) < offset + 16: self.note_levels.append(0.0)
            
            for i in range(16):
                idx = offset + i
                self.pattern[idx] = False
                self.note_levels[idx] = 0.0
                
            self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        r = self.rect().adjusted(1, 1, -1, -1)
        w, h = r.width(), r.height()
        step_w = w / self.steps

        # 1. Background Grid
        painter.setBrush(QColor("#f7fafc"))
        painter.setPen(QPen(QColor("#e2e8f0"), 1))
        painter.drawRoundedRect(r, 3, 3) 
        
        painter.save()
        path = QPainterPath(); path.addRoundedRect(QRectF(r), 3, 3); painter.setClipPath(path)
        for i in range(1, self.steps):
            x = i * step_w
            painter.drawLine(int(x), 0, int(x), int(h))
        painter.setPen(QPen(QColor("#edf2f7"), 1))
        for i in range(1, 8):
            y = i * (h / 8)
            painter.drawLine(0, int(y), w, int(y))
        painter.restore()

        # 2. Notes
        painter.setPen(Qt.PenStyle.NoPen)
        t = time.time() * 0.5
        for i in range(self.steps): # Loop 0-15 (Visual columns)
            data_idx = i + self.view_offset
            
            # Safety check
            if data_idx >= len(self.note_levels): break
            
            lvl = self.note_levels[data_idx]
            if lvl > 0.01:
                x = i * step_w
                val = self.values[data_idx]
                note_h = h / 8
                note_y = (1.0 - val) * h 
                grid_y = int(note_y / note_h) * note_h
                rect_note = QRectF(x + 1, grid_y + 1, step_w - 2, note_h - 2)
                
                hue_offset = np.sin(t + (i * 0.3)) * 15
                current_hue = int((self.base_hue + hue_offset) % 360)
                
                c = QColor.fromHsl(current_hue, 160, 160)
                c.setAlpha(int(255 * lvl))
                painter.setBrush(c)
                painter.drawRoundedRect(rect_note, 2, 2)

        # 3. Playhead
        if self.playhead_pos >= 0:
            cx = self.playhead_pos * step_w + (step_w / 2)
            bar_w = step_w * 1.2
            r_play = QRectF(cx - (bar_w/2), 0, bar_w, h)
            grad = QLinearGradient(r_play.topLeft(), r_play.topRight())
            c_center = QColor(148, 163, 184, 45)
            c_edge = QColor(148, 163, 184, 0)
            grad.setColorAt(0, c_edge); grad.setColorAt(0.5, c_center); grad.setColorAt(1, c_edge)
            painter.setBrush(grad); painter.setPen(Qt.PenStyle.NoPen)
            painter.drawRect(r_play)
            
        # 4. Hover
        if self.hover_step >= 0:
            x = self.hover_step * step_w
            painter.setBrush(QColor(200, 210, 225, 40))
            painter.drawRect(QRectF(x, 0, step_w, h))

class BassRow(QFrame):
    pattern_changed = pyqtSignal()
    preview_req = pyqtSignal(object)
    saved_msg = pyqtSignal(str) 

    def __init__(self, bpm_ref, base_hue=300, parent=None):
        super().__init__(parent)
        self.bpm = bpm_ref
        self.base_hue = base_hue 
        self.scale_keys = list(BassEngine.SCALES.keys())
        self.current_scale_idx = 0
        self.roots = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        self.current_root_idx = 0
        self.current_data = None

        self.layout = QHBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(0)

        # --- Label Layout ---
        lbl_frame = QWidget(); lbl_frame.setFixedWidth(55)
        lf_layout = QHBoxLayout(lbl_frame); lf_layout.setContentsMargins(0,0,0,0)
        
        self.lbl = GradientLabel(" bass")
        self.lbl.clicked.connect(self.randomize_pattern)
        
        # Align Left (0 stretch)
        lf_layout.addWidget(self.lbl, 0, Qt.AlignmentFlag.AlignLeft)
        
        self.layout.addWidget(lbl_frame)
        # ---------------------------------

        ctrl_frame = QWidget(); ctrl_frame.setFixedWidth(205)
        c_layout = QHBoxLayout(ctrl_frame); c_layout.setContentsMargins(0, 0, 0, 0); c_layout.setSpacing(0)

        btn_container = QWidget(); btn_container.setFixedWidth(65)
        bc_layout = QVBoxLayout(btn_container); bc_layout.setContentsMargins(0, 0, 4, 0)
        bc_layout.setSpacing(1); bc_layout.setAlignment(Qt.AlignmentFlag.AlignVCenter)
        
        def make_display_lbl(txt):
            l = QLabel(txt)
            l.setAlignment(Qt.AlignmentFlag.AlignCenter)
            l.setStyleSheet("font-size: 9px; font-weight: bold; color: #9ba5b2; background: white; border: 1px solid #cbd5e0; border-radius: 2px;")
            # Increased height to 22px to match buttons and other labels
            l.setFixedHeight(22)
            return l

        scale_row = QWidget(); sr_layout = QHBoxLayout(scale_row); sr_layout.setContentsMargins(0,0,0,0); sr_layout.setSpacing(1)
        self.btn_scale_prev = ArrowButton('left'); self.btn_scale_prev.clicked.connect(lambda: self.cycle_scale(-1))
        self.lbl_scale = make_display_lbl(self.scale_keys[0])
        self.btn_scale_next = ArrowButton('right'); self.btn_scale_next.clicked.connect(lambda: self.cycle_scale(1))
        sr_layout.addWidget(self.btn_scale_prev); sr_layout.addWidget(self.lbl_scale, 1); sr_layout.addWidget(self.btn_scale_next)
        
        root_row = QWidget(); rr_layout = QHBoxLayout(root_row); rr_layout.setContentsMargins(0,0,0,0); rr_layout.setSpacing(1)
        self.btn_root_prev = ArrowButton('left'); self.btn_root_prev.clicked.connect(lambda: self.cycle_root(-1))
        self.lbl_root = make_display_lbl(self.roots[0])
        self.btn_root_next = ArrowButton('right'); self.btn_root_next.clicked.connect(lambda: self.cycle_root(1))
        rr_layout.addWidget(self.btn_root_prev); rr_layout.addWidget(self.lbl_root, 1); rr_layout.addWidget(self.btn_root_next)

        bc_layout.addWidget(scale_row); bc_layout.addWidget(root_row); c_layout.addWidget(btn_container)

        slider_container = QWidget(); sc_layout = QHBoxLayout(slider_container); sc_layout.setContentsMargins(2, 0, 2, 0); sc_layout.setSpacing(2)
        def make_labeled_slider(val, label_txt, hue_offset, def_val=50, tooltip=""):
            container = QWidget(); container.setFixedWidth(20) 
            vbox = QVBoxLayout(container); vbox.setContentsMargins(0,0,0,0); vbox.setSpacing(1)
            header = HeaderParamLabel(label_txt, "local"); header.setFixedHeight(14) 
            sl = CircleSlider(Qt.Orientation.Vertical, base_hue=(self.base_hue + hue_offset) % 360, default_value=def_val)
            sl.setRange(0, 100); sl.setValue(val); sl.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding); sl.setFixedWidth(20) 
            if tooltip: sl.setToolTip(tooltip)
            header.clicked.connect(lambda: self.randomize_local_slider(sl, label_txt))
            vbox.addWidget(header); vbox.addWidget(sl)
            return container, sl

        w_vol, self.sl_vol = make_labeled_slider(80, "vol", 0, 80, "Master Volume")
        w_rel, self.sl_release = make_labeled_slider(50, "rel", 0, 50, "Release")   
        w_flt, self.sl_tone = make_labeled_slider(60, "flt", 10, 50, "Filter")  
        w_pch, self.sl_root_sl = make_labeled_slider(40, "pch", 20, 40, "Pitch")
        w_dec, self.sl_decay = make_labeled_slider(50, "dec", 30, 50, "Decay")
        w_gld, self.sl_glide = make_labeled_slider(10, "gld", 40, 0, "Glide") 

        for sl in [self.sl_vol, self.sl_release, self.sl_tone, self.sl_root_sl, self.sl_decay, self.sl_glide]:
            sl.valueChanged.connect(self.process_sequence)

        sc_layout.addWidget(w_vol); sc_layout.addWidget(w_rel); sc_layout.addWidget(w_flt)
        sc_layout.addWidget(w_pch); sc_layout.addWidget(w_dec); sc_layout.addWidget(w_gld)
        c_layout.addWidget(slider_container); self.layout.addWidget(ctrl_frame)

        self.piano = BassPianoRoll(steps=STEPS, base_hue=170, parent=self)
        self.piano.setFixedWidth(448); self.piano.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)
        self.piano.pattern_changed.connect(self.process_sequence)
        self.layout.addWidget(self.piano)

        self.layout.addSpacing(6)
        self.btn_clr = ClearButton(self); self.btn_clr.clicked.connect(self.clear)
        self.layout.addWidget(self.btn_clr); self.layout.addStretch(1)   
        self.process_sequence()

    @property
    def pattern(self): return self.piano.pattern
    
    @property
    def velocities(self): return self.piano.values

    def update_sound(self, play=False): self.process_sequence()

    def randomize_local_slider(self, slider, param_name):
        desc_map = {"vol": "volume", "rel": "note release", "flt": "filter cutoff", "pch": "pitch octave", "dec": "amp decay", "gld": "glide amount"}
        full_name = desc_map.get(param_name, param_name)
        slider.setValue(np.random.randint(0, 101)); self.saved_msg.emit(f"bass: {full_name} rnd")

    def cycle_scale(self, delta):
        self.current_scale_idx = (self.current_scale_idx + delta) % len(self.scale_keys)
        name = self.scale_keys[self.current_scale_idx]; self.lbl_scale.setText(name)
        self.process_sequence(); self.saved_msg.emit(f"scale: {name}")

    def cycle_root(self, delta):
        self.current_root_idx = (self.current_root_idx + delta) % len(self.roots)
        root_name = self.roots[self.current_root_idx]; self.lbl_root.setText(root_name)
        self.process_sequence(); self.saved_msg.emit(f"root: {root_name}")

    def process_sequence(self, steps=32, notify=True):      
        octave_offset = 12 if self.sl_root_sl.value() > 60 else -12 if self.sl_root_sl.value() < 40 else 0
        params = {
            'root_note': self.current_root_idx + octave_offset, 'glide': self.sl_glide.value() / 100.0,
            'decay': self.sl_decay.value() / 100.0, 'release': self.sl_release.value() / 100.0,
            'tone': self.sl_tone.value() / 100.0, 'scale_name': self.scale_keys[self.current_scale_idx]
        }
        
        while len(self.piano.pattern) < 32: self.piano.pattern.append(False)
        while len(self.piano.values) < 32: self.piano.values.append(0.5)

        raw = BassEngine.generate_sequence(self.piano.pattern, self.piano.values, self.bpm, params, steps=32)
        vol = (self.sl_vol.value() / 100.0) ** 2

        # This prevents filter blowups from killing the Kick drum in the mixer
        self.current_data = np.nan_to_num(raw * vol, copy=False)
        
        if notify:
            self.pattern_changed.emit()

    def randomize_pattern(self):
        density = np.random.uniform(0.3, 0.6)
        
        # FIX 1: Target the currently visible bank (0 or 16)
        offset = self.piano.view_offset
        
        # Ensure capacity
        while len(self.piano.pattern) < 32: self.piano.pattern.append(False)
        while len(self.piano.values) < 32: self.piano.values.append(0.5)
        while len(self.piano.note_levels) < 32: self.piano.note_levels.append(0.0)

        for i in range(16):
            idx = offset + i
            
            is_active = (np.random.random() < density)
            self.piano.pattern[idx] = is_active
            
            if is_active:
                # FIX 2: Restrict to 0-7 integer grid so they are always clickable
                # (grid_row + 0.5) / 8.0 centers the value in the visual slot
                grid_row = np.random.randint(0, 8)
                val = (grid_row + 0.5) / 8.0
                self.piano.values[idx] = val
                self.piano.note_levels[idx] = 1.0
            else:
                self.piano.note_levels[idx] = 0.0
                
        self.piano.update()
        self.process_sequence(steps=32) # Ensure we process full length
        
        side_num = int(offset/16) + 1
        self.saved_msg.emit(f"bass: random ptn {side_num}")

    def update_bpm(self, new_bpm):
        self.bpm = new_bpm
        # Re-generate the sequence at the new tempo
        self.process_sequence(steps=32)

    def clear(self):
        self.piano.clear(); self.process_sequence()

    def highlight(self, step_float):
        self.piano.set_playing_pos(step_float)

class SlotRow(QFrame):
    pattern_changed = pyqtSignal()
    preview_req = pyqtSignal(object)
    saved_msg = pyqtSignal(str) 

    def __init__(self, label_text, drum_type, base_hue=0, parent=None):
        super().__init__(parent)
        self.label_text = label_text
        self.drum_type = drum_type
        self.view_offset = 0
        self.original_data = None; self.current_data = None; self.raw_sample = None       
        self.is_sample_mode = False  
        self.pattern = [False] * 32
        self.velocities = [0.8] * 32
        self.pads = []; self.paint_state = True
        self.synth_params = {'pitch': 0.5, 'decay': 0.5, 'tone': 0.3}

        self.layout = QHBoxLayout(self); self.layout.setContentsMargins(0, 0, 0, 0); self.layout.setSpacing(0)

        lbl_frame = QWidget(); lbl_frame.setFixedWidth(55)
        lf_layout = QHBoxLayout(lbl_frame); lf_layout.setContentsMargins(0,0,0,0); lf_layout.setSpacing(0)
        self.lbl = DraggableLabel(label_text.lower()); self.lbl.file_dropped.connect(self.load_sample)
        self.lbl.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        lf_layout.addWidget(self.lbl, 1) 
        self.btn_reset = ClearButton(self); self.btn_reset.setFixedSize(14, 14); self.btn_reset.hide()
        self.btn_reset.clicked.connect(self.reset_to_synth); lf_layout.addWidget(self.btn_reset, 0)
        self.layout.addWidget(lbl_frame)

        ctrl_frame = QWidget(); ctrl_frame.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Preferred); ctrl_frame.setFixedWidth(205) 
        ctrl_main_layout = QHBoxLayout(ctrl_frame); ctrl_main_layout.setContentsMargins(0, 0, 0, 0); ctrl_main_layout.setSpacing(0)

        btn_container = QWidget(); btn_container.setFixedWidth(65)
        btn_grid = QGridLayout(btn_container); btn_grid.setContentsMargins(0, 0, 4, 0)
        btn_grid.setSpacing(1); btn_grid.setVerticalSpacing(1)
        self.btn_wav = self.create_btn("wav"); self.btn_wav.clicked.connect(self.export_one)
        self.btn_vel = self.create_btn("vel"); self.btn_vel.clicked.connect(self.randomize_velocity)
        self.btn_rnd = self.create_btn("rnd"); self.btn_rnd.clicked.connect(self.syncopate_gentle)
        self.btn_low = self.create_btn("low"); self.btn_low.clicked.connect(self.lower_velocity)
        btn_grid.addWidget(self.btn_wav, 0, 0); btn_grid.addWidget(self.btn_vel, 0, 1)
        btn_grid.addWidget(self.btn_rnd, 1, 0); btn_grid.addWidget(self.btn_low, 1, 1); btn_grid.setAlignment(Qt.AlignmentFlag.AlignVCenter)
        ctrl_main_layout.addWidget(btn_container)

        crush_container = QWidget(); cc_layout = QHBoxLayout(crush_container); cc_layout.setContentsMargins(2, 0, 2, 0); cc_layout.setSpacing(2)
        def make_v_slider(val, tip, hue_offset, def_val=50):
            sl = CircleSlider(Qt.Orientation.Vertical, base_hue=(base_hue + hue_offset) % 360, default_value=def_val)
            sl.setRange(0, 100); sl.setValue(val); sl.setFixedWidth(20); sl.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)
            sl.setToolTip(tip); return sl

        self.sl_vol = make_v_slider(85, "Volume", 0, def_val=85); self.sl_vol.valueChanged.connect(self.schedule_update)
        self.sl_crush = make_v_slider(0, "Bitcrush", 0, def_val=0); self.sl_crush.valueChanged.connect(self.schedule_update)
        self.sl_filt = make_v_slider(50, "Filter", 10, def_val=50); self.sl_filt.valueChanged.connect(self.schedule_update)
        self.sl_pitch = make_v_slider(50, "Pitch", 20, def_val=50); self.sl_pitch.valueChanged.connect(self.on_synth_change)
        self.sl_decay = make_v_slider(50, "Decay", 30, def_val=50); self.sl_decay.valueChanged.connect(self.on_synth_change)
        self.sl_tone = make_v_slider(30, "Tone", 40, def_val=30); self.sl_tone.valueChanged.connect(self.on_synth_change)
        
        cc_layout.addWidget(self.sl_vol); cc_layout.addWidget(self.sl_crush); cc_layout.addWidget(self.sl_filt)
        cc_layout.addWidget(self.sl_pitch); cc_layout.addWidget(self.sl_decay); cc_layout.addWidget(self.sl_tone)
        ctrl_main_layout.addWidget(crush_container); self.layout.addWidget(ctrl_frame)

        for i in range(STEPS):
            p = StepPad(i, base_hue, self)
            
            # CONNECT DIRECTLY to handlers. 
            # The pad now passes its own index (0-31), so we don't need lambdas or offsets.
            p.toggled.connect(self.on_pad_toggled)
            p.velocity_changed.connect(self.on_pad_velocity)
            
            self.pads.append(p)
            self.layout.addWidget(p)

        self.btn_clr = ClearButton(self); self.btn_clr.clicked.connect(self.clear)
        self.layout.addSpacing(6); self.layout.addWidget(self.btn_clr); self.layout.addStretch(1)
        self.update_timer = QTimer(self); self.update_timer.setSingleShot(True); self.update_timer.setInterval(100)
        self.update_timer.timeout.connect(lambda: self.update_sound(play=False))
        self.update_sound()
    
    def schedule_update(self): self.update_timer.start()
    def start_painting(self, state): self.paint_state = state
    def create_btn(self, text, checkable=False): return FadeButton(text, is_small=True)
    
    def on_pad_toggled(self, real_idx, act, vel):
        # We trust the pad's index implicitly
        while len(self.pattern) <= real_idx: self.pattern.append(False)
        while len(self.velocities) <= real_idx: self.velocities.append(0.8)
        
        self.pattern[real_idx] = act
        self.velocities[real_idx] = vel
        self.pattern_changed.emit()
    
    def on_pad_velocity(self, real_idx, vel):
        # We trust the pad's index implicitly
        while len(self.velocities) <= real_idx: self.velocities.append(0.8)
        
        self.velocities[real_idx] = vel
        
        # Ensure pattern is active if velocity changed
        if not self.pattern[real_idx]:
            self.pattern[real_idx] = True
            # Update UI state just in case
            # Find the pad that triggered this (it might be visible)
            for p in self.pads:
                if p.index == real_idx:
                    p.active = True
                    p.update()
                    break
                    
        self.pattern_changed.emit()

    def lower_velocity(self):
        offset = self.view_offset
        target_len = offset + 16
        while len(self.velocities) < target_len: self.velocities.append(0.8)
        
        changed = False
        for i in range(16):
            real_idx = offset + i
            if self.pattern[real_idx]:
                new_vel = max(0.1, self.velocities[real_idx] - 0.15)
                if abs(new_vel - self.velocities[real_idx]) > 0.001:
                    self.velocities[real_idx] = new_vel
                    self.pads[i].velocity = new_vel
                    self.pads[i].update()
                    changed = True
        if changed: self.pattern_changed.emit()
    
    def load_sample(self, file_path):
        try:
            data, file_sr = sf.read(file_path)
            if len(data.shape) > 1:
                data = np.mean(data, axis=1)
            max_samples = 3 * file_sr 
            if len(data) > max_samples:
                data = data[:max_samples]
            if file_sr != SR:
                num_samples = int(len(data) * SR / file_sr)
                data = signal.resample(data, num_samples)
            peak = np.max(np.abs(data))
            if peak > 0: data = data / peak
            data = SynthEngine.ensure_zero_crossing(data.astype(np.float32))

            self.raw_sample = data
            self.is_sample_mode = True
            
            self.lbl.setStyleSheet("""
                QLabel { color: #38a169; font-weight: bold; font-size: 12px; margin-left: 2px;} 
                QLabel:hover { background-color: #f0fff4; }
            """)
            self.btn_reset.show()
            self.saved_msg.emit(f"loaded: {file_path.split('/')[-1]}")
            
            self.update_sound(play=True)
            
            self.current_data = None  # Force reprocessing
            self.process_audio()      # Process immediately
        
        except Exception as e:
            self.saved_msg.emit("load err")
            print(f"Error loading: {e}")

    def reset_to_synth(self):
        self.is_sample_mode = False
        self.raw_sample = None
        self.btn_reset.hide()
        self.lbl.setStyleSheet("""
            QLabel { 
                font-size: 12px; color: #4a5568; font-weight: bold; 
                margin-left: 2px; border: 1px dashed transparent;
                border-radius: 4px; padding: 2px;
            }
            QLabel:hover { background-color: #ebf8ff; border-color: #90cdf4; color: #2b6cb0; }
        """)
        self.saved_msg.emit(f"reverted: {self.label_text}")
        self.update_sound(play=True)

    def update_synth_params(self, pitch, decay, tone):
        self.synth_params = {'pitch': pitch, 'decay': decay, 'tone': tone}
        self.update_sound(play=False)

    def on_synth_change(self):
        p = self.sl_pitch.value() / 100.0
        d = self.sl_decay.value() / 100.0
        t = self.sl_tone.value() / 100.0
        self.update_synth_params(p, d, t)

    def update_sound(self, play=False):
        # Explicitly read slider values to ensure initial state isn't stale
        p = float(self.sl_pitch.value()) / 100.0
        d = float(self.sl_decay.value()) / 100.0
        t = float(self.sl_tone.value()) / 100.0
        
        self.synth_params = {'pitch': p, 'decay': d, 'tone': t}

        if self.is_sample_mode and self.raw_sample is not None:
            self.original_data = SynthEngine.process_sample(self.raw_sample, self.synth_params)
        else:
            self.original_data = SynthEngine.generate_drum(self.drum_type, self.synth_params)
            
        self.process_audio()
        if play and self.current_data is not None:
            self.preview_req.emit(self.current_data)
        self.pattern_changed.emit()

    def on_crush_change(self):
        self.process_audio()
        self.pattern_changed.emit()

    def process_audio(self):
        if self.original_data is None:
            self.update_sound(play=False)
            return
            
        if len(self.original_data) == 0:
            self.original_data = SynthEngine.generate_fallback_silence()
        
        # Apply Filter
        filt_val = self.sl_filt.value() / 100.0
        filtered = SynthEngine.apply_filter(self.original_data, filt_val)
        
        # Apply Bitcrush
        crush = self.sl_crush.value() / 100.0
        self.current_data = SynthEngine.resample_lofi(filtered, crush)
        
        # Apply Global Volume
        vol = (self.sl_vol.value() / 100.0) ** 2  
        self.current_data *= vol
        
        # Drum Click Fade
        if len(self.current_data) > 100:
            fade = 50
            self.current_data[-fade:] *= np.linspace(1, 0, fade)

        # --- FIX: Sanitize Drum Audio ---
        self.current_data = np.nan_to_num(self.current_data, copy=False)

        if self.current_data is None:
            self.current_data = SynthEngine.generate_fallback_silence()

    def update_step(self, idx, act, vel):
        self.pattern[idx] = act
        self.velocities[idx] = vel
        self.pattern_changed.emit()

    def randomize_velocity(self):
        offset = self.view_offset # Use explicit offset
        target_len = offset + 16
        while len(self.velocities) < target_len: self.velocities.append(0.8)
        while len(self.pattern) < target_len: self.pattern.append(False)
        
        changed = False
        for i in range(16):
            real_idx = offset + i
            if self.pattern[real_idx]:
                new_vel = np.random.uniform(0.3, 1.0)
                self.velocities[real_idx] = new_vel
                self.pads[i].velocity = new_vel
                self.pads[i].update()
                changed = True
        if changed: self.pattern_changed.emit()

    def syncopate_gentle(self):
        offset = self.view_offset
        target_len = offset + 16
        while len(self.pattern) < target_len: self.pattern.append(False)
        while len(self.velocities) < target_len: self.velocities.append(0.8)
        
        density = 0.35
        if "kick" in self.label_text: density = 0.25
        elif "hat" in self.label_text: density = 0.5
        elif "snare" in self.label_text: density = 0.2
        
        changed = False
        for i in range(16):
            real_idx = offset + i
            is_active = np.random.random() < density
            self.pattern[real_idx] = is_active
            self.pads[i].active = is_active
            if is_active:
                new_vel = np.random.uniform(0.4, 1.0)
                self.velocities[real_idx] = new_vel
                self.pads[i].velocity = new_vel
            self.pads[i].update()
            changed = True
        if changed: self.pattern_changed.emit()

    def clear(self):
        offset = self.view_offset
        target_len = offset + 16
        while len(self.pattern) < target_len: self.pattern.append(False)
        
        changed = False
        for i in range(16):
            real_idx = offset + i
            if self.pattern[real_idx]:
                self.pattern[real_idx] = False
                changed = True
            self.pads[i].active = False
            self.pads[i].update()
        if changed: self.pattern_changed.emit()

    def highlight(self, step_float):
        # Pass the raw 0-16 local time directly.
        # StepPad.set_playing_pos handles the (index % 16) logic internally.
        for p in self.pads: 
            p.set_playing_pos(step_float)

    def export_one(self):
        if self.current_data is None: return
        
        home_dir = os.path.expanduser("~")
        save_dir = os.path.join(home_dir, "Music", "sequa")
        
        if not os.path.exists(save_dir):
            try: os.makedirs(save_dir)
            except: self.saved_msg.emit("err: cannot create folder"); return

        timestamp = int(time.time())
        safe_name = self.label_text.replace(" ", "_")
        filename = f"{safe_name}_{timestamp}.wav"
        full_path = os.path.join(save_dir, filename)
        
        try:
            # 1. Get Data
            data_out = self.current_data.copy()
            
            # 2. Normalize safely to -0.5dB
            peak = np.max(np.abs(data_out))
            if peak > 0:
                data_out = data_out * (0.944 / peak)
            
            # 3. Apply Limiter (Soft Clip) at end to catch stray peaks after norm
            data_out = np.tanh(data_out)
            
            # 4. Make Stereo
            data_stereo = np.column_stack((data_out, data_out))

            sf.write(full_path, data_stereo, SR)
            self.saved_msg.emit(f"saved: {filename}")
        except Exception as e:
            self.saved_msg.emit(f"err: {str(e)}")

    def drift_params(self, amount):
        if amount <= 0.01: return
        
        # --- 1. Slider Physics (Optimized) ---
        targets = [self.sl_crush, self.sl_filt, self.sl_decay, self.sl_tone]
        for sl in targets:
            if not hasattr(sl, '_f_val'): sl._f_val = float(sl.value())
            if not hasattr(sl, '_vel'): sl._vel = 0.0
            
            # Add random force
            force = (np.random.random() - 0.5) * 0.04 * amount
            
            # Spring force to keep near center (50)
            dist = sl._f_val - 50.0
            force -= dist * 0.001 * amount

            # Apply force
            sl._vel += force
            
            # FIX: Clamp Velocity (Prevents "Buffer Accum" / runaway physics)
            sl._vel = np.clip(sl._vel, -2.0, 2.0)
            sl._vel *= 0.95 # Damping
            
            new_val = sl._f_val + sl._vel
            sl._f_val = np.clip(new_val, 0.0, 100.0)
            
            # Only update UI/Audio if value changed by integer amount
            if int(sl._f_val) != sl.value():
                sl.blockSignals(True) # Block to prevent 60FPS audio regen spam
                sl.setValue(int(sl._f_val))
                sl.blockSignals(False)
                # We defer audio update to the occasional pattern change or timer

        # --- 2. Pattern Evolution (Updated for 32 Steps) ---
        # Probability check to prevent chaos (runs every frame)
        if np.random.random() < 0.02: 
            
            # Mutation chance based on slider amount
            mutate_prob = amount * 0.002 
            changed = False
            
            # FIX: Iterate full 32 steps (Bank A & B)
            for i in range(32):
                # Capacity Check
                if i >= len(self.pattern): self.pattern.append(False)
                if i >= len(self.velocities): self.velocities.append(0.8)

                if np.random.random() < mutate_prob:
                    # Flip State
                    self.pattern[i] = not self.pattern[i]
                    changed = True
                    
                    # Update Velocity if turning on
                    if self.pattern[i]:
                        self.velocities[i] = np.random.uniform(0.7, 1.0)
                    
                    # Update Visual Pad if currently visible
                    for pad in self.pads:
                        if pad.index == i:
                            pad.active = self.pattern[i]
                            pad.velocity = self.velocities[i]
                            pad.update()

            if changed:
                self.pattern_changed.emit()

class CircleSlider(QSlider):
    def __init__(self, orientation, base_hue=210, default_value=50, parent=None):
        super().__init__(orientation, parent)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.default_value = default_value
        
        self.base_hue = base_hue
        self.current_hue = base_hue 
        self.color_1 = QColor.fromHsl(int(self.base_hue), 150, 160)
        self.color_2 = QColor.fromHsl(int((self.base_hue + 20) % 360), 180, 130)
        
        self.phase = np.random.rand() * 10

    def _update_from_mouse(self, pos):
        # Match margin used in paintEvent
        margin = 8 
        
        if self.orientation() == Qt.Orientation.Horizontal:
            w = self.width() - (margin * 2)
            if w <= 0: return
            
            # Clamp x to visual track area
            rel_x = max(0, min(pos.x() - margin, w))
            norm = rel_x / w
            
            val = self.minimum() + (norm * (self.maximum() - self.minimum()))
        else:
            h = self.height() - (margin * 2)
            if h <= 0: return
            
            # Clamp y to visual track area
            rel_y = max(0, min(pos.y() - margin, h))
            
            # Invert Y (Bottom is min, Top is max for vertical slider logic usually, 
            # but visual Y=0 is top. We want Top=Max, Bottom=Min)
            norm = (h - rel_y) / h
            
            val = self.minimum() + (norm * (self.maximum() - self.minimum()))
        
        self.setValue(int(val))

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.RightButton:
            self.setValue(self.default_value)
            event.accept()
        elif event.button() == Qt.MouseButton.LeftButton:
            self._update_from_mouse(event.pos())
            event.accept()
            self.sliderPressed.emit()
            self.valueChanged.emit(self.value())
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.MouseButton.LeftButton:
            self._update_from_mouse(event.pos())
            event.accept()
            self.valueChanged.emit(self.value())
        else:
            super().mouseMoveEvent(event)

    def tick_color(self):
        self.phase += 0.03
        hue_offset = np.sin(self.phase) * 15
        h = (self.base_hue + hue_offset) % 360
        
        self.color_1 = QColor.fromHsl(int(h), 160, 180)
        self.color_2 = QColor.fromHsl(int((h + 30) % 360), 180, 140)
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        margin = 8
        handle_r = 6.0
        
        painter.setPen(Qt.PenStyle.NoPen)
        
        if self.orientation() == Qt.Orientation.Horizontal:
            cy = self.height() / 2
            w = self.width() - (margin * 2)
            norm = (self.value() - self.minimum()) / (self.maximum() - self.minimum())
            x = margin + (w * norm)
            
            # Groove
            painter.setBrush(QColor("#e2e8f0"))
            painter.drawRoundedRect(QRectF(margin, cy - 2, w, 4), 2, 2)
            
            # Active Groove (Gradient)
            if x > margin:
                rect = QRectF(margin, cy - 2, x - margin, 4)
                grad = QLinearGradient(rect.topLeft(), rect.topRight())
                grad.setColorAt(0, self.color_1)
                grad.setColorAt(1, self.color_2)
                painter.setBrush(grad)
                painter.drawRoundedRect(rect, 2, 2)
            
            # Handle
            painter.setBrush(QColor("white"))
            painter.setPen(QPen(self.color_1, 1)) 
            painter.drawEllipse(QPointF(x, cy), handle_r, handle_r)
            
        else:
            cx = self.width() / 2
            h = self.height() - (margin * 2)
            norm = (self.value() - self.minimum()) / (self.maximum() - self.minimum())
            y = (self.height() - margin) - (h * norm)
            
            # Groove
            painter.setBrush(QColor("#e2e8f0"))
            painter.drawRoundedRect(QRectF(cx - 2, margin, 4, h), 2, 2)
            
            # Animated Gradient Fill
            fill_h = (self.height() - margin) - y
            if fill_h > 0:
                rect = QRectF(cx - 2, y, 4, fill_h)
                grad = QLinearGradient(rect.topLeft(), rect.bottomLeft())
                grad.setColorAt(0, self.color_1)
                grad.setColorAt(1, self.color_2)
                painter.setBrush(grad)
                painter.drawRoundedRect(rect, 2, 2)
                
                painter.setPen(QPen(self.color_1, 1))
            else:
                painter.setPen(QPen(QColor("#4299e1"), 1))

            # Handle
            painter.setBrush(QColor("white"))
            painter.drawEllipse(QPointF(cx, y), handle_r, handle_r)

class StatusWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(26)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        
        self.text = ""
        
        # Internal timer for independent animation
        self.anim_timer = QTimer(self)
        self.anim_timer.timeout.connect(self.animate)
        self.anim_timer.setInterval(30) 
        
        self.font = QFont("Segoe UI", 9, QFont.Weight.DemiBold)
        
        # Fade Variables
        self.opacity = 0.0
        self.target_opacity = 0.0
        self.display_active = False

    def set_text(self, text):
        self.text = text
        self.target_opacity = 1.0
        self.display_active = True
        
        if not self.anim_timer.isActive():
            self.anim_timer.start()
            
        QTimer.singleShot(3000, lambda: self.trigger_fade_out(text))
        self.update()

    def trigger_fade_out(self, match_text):
        if self.text == match_text:
            self.target_opacity = 0.0

    def animate(self):
        # Linear Fade In/Out (More responsive/snappy)
        step = 0.15 
        
        if self.opacity < self.target_opacity:
            self.opacity = min(self.target_opacity, self.opacity + step)
        elif self.opacity > self.target_opacity:
            self.opacity = max(self.target_opacity, self.opacity - step)

        if self.opacity <= 0.0 and self.target_opacity == 0.0:
            self.display_active = False
            self.anim_timer.stop()
            self.update()
            return

        if self.display_active:
            self.update()

    def paintEvent(self, event):
        if self.opacity <= 0.01: return
        
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Calculate alpha (0-255) based on float opacity
        alpha = int(255 * self.opacity)
        
        # Set color with explicit alpha
        col = QColor("#747db8")
        col.setAlpha(alpha)
        
        painter.setPen(col) 
        painter.setFont(self.font)
        
        rect = self.rect()
        text_rect = rect.adjusted(10, 0, -5, 0) 
        painter.drawText(text_rect, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, self.text)

class HeaderParamLabel(QLabel):
    clicked = pyqtSignal(str)

    def __init__(self, text, param_key, parent=None):
        super().__init__(text, parent)
        self.param_key = param_key
        self.setFixedWidth(20) # Match slider width
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        # Default: Gray
        self.base_style = "font-size: 9px; font-weight: bold; color: #718096;"
        self.setStyleSheet(self.base_style)

    def enterEvent(self, event):
        # Hover: Lighter saturation (Soft Blue)
        self.setStyleSheet("font-size: 9px; font-weight: bold; color: #63b3ed;")
        super().enterEvent(event)

    def leaveEvent(self, event):
        self.setStyleSheet(self.base_style)
        super().leaveEvent(event)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(self.param_key)

class SequaWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("sequa")
        # Enlarge window
        self.setFixedSize(780, 630) 
        
        self.slots = []
        self.bpm = BPM_DEFAULT
        self.swing = 0.0
        self.clip_amount = 0.0
        self.evolve_amount = 0.0
        self.reverse_prob = 0.0 
        self.step = -1
        self.last_processed_step = -1
        self.last_preview_time = 0
        self.anim_tick_counter = 0
        self.last_pause_timestamp = 0
        self.total_bars = 1
        self.view_offset = 0  # 0 for Bar 1, 16 for Bar 2

        self.fmt = QAudioFormat()
        self.fmt.setSampleRate(SR)
        self.fmt.setChannelCount(1)
        self.fmt.setSampleFormat(QAudioFormat.SampleFormat.Int16)
        
        self.sink = QAudioSink(QMediaDevices.defaultAudioOutput(), self.fmt)
        self.sink.setBufferSize(int(SR * 2 * (BUFFER_MS / 1000.0)) * 2)
        
        preferred_size = self.sink.bufferSize()
        if preferred_size < 4096:
            self.sink.setBufferSize(4096)
        
        self.gen = LoopGenerator(self.fmt, self)
        self.preview = SoundPreview(self)
        
        self.timer = QTimer()
        self.timer.setTimerType(Qt.TimerType.PreciseTimer)
        self.timer.setInterval(5) 
        self.timer.timeout.connect(self.update_playhead)
        self.timer.start() 
        
        self.setup_ui()
        self.update_mix()
        self.sink.start(self.gen)
    
    def change_kit(self, kit_idx):
        kit_map = [SynthEngine.KIT_SIMPLE, SynthEngine.KIT_PCM, SynthEngine.KIT_EIGHT, SynthEngine.KIT_NINE]
        SynthEngine.set_kit(kit_map[kit_idx])
        for s in self.slots: s.update_sound(play=False)
        now = time.time()
        if self.slots and (now - self.last_preview_time > 0.15):
            self.last_preview_time = now
            drums = [s for s in self.slots if not isinstance(s, ReseqRow)]
            if drums:
                r_slot = drums[np.random.randint(len(drums))]
                # CHANGED: Lower volume for preview sample (0.5)
                self.preview.play(r_slot.current_data * 0.5)
        names = ["simple", "pcm", "eight", "nine"]
        self.show_notification(f"kit loaded: {names[kit_idx]}")
    
    def randomize_top_param(self, name, slider):
         slider.setValue(np.random.randint(0, 101))
         self.show_notification(f"randomized: {name}")
    
    def setup_ui(self):
        cw = QWidget(); self.setCentralWidget(cw)
        cw.setStyleSheet("QWidget { font-family: 'Segoe UI', sans-serif; background-color: #f7fafc; }")
        
        # Reduced Height: 630 -> 610
        self.setFixedSize(780, 610) 
        
        main = QVBoxLayout(cw)
        
        # Reduced Margins: Top 4->2, Bottom 20->10
        main.setContentsMargins(20, 2, 20, 10) 
        main.setSpacing(1) 
        
        # Initialize SplitPill here
        self.btn_split = SplitPill()
        self.btn_split.view_changed.connect(self.switch_view_page)
        
        header = QHBoxLayout()
        header.setSpacing(5) 
        header.setContentsMargins(0, 0, 0, 5)
        header.setAlignment(Qt.AlignmentFlag.AlignVCenter)
        
        self.logo = LogoWidget()
        self.logo.kit_changed.connect(self.change_kit)
        
        logo_cont = QWidget()
        logo_layout = QVBoxLayout(logo_cont)
        logo_layout.setContentsMargins(0, -4, 0, 0) 
        logo_layout.addWidget(self.logo)
        header.addWidget(logo_cont)
        
        class ClickLabel(QLabel):
            clicked = pyqtSignal()
            def __init__(self, text):
                super().__init__(text)
                self.setFixedWidth(48)
                self.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                self.setStyleSheet("color: #718096; font-weight: bold; font-size: 12px; margin-right: 4px;")
                self.setCursor(Qt.CursorShape.PointingHandCursor)
            def mousePressEvent(self, e):
                if e.button() == Qt.MouseButton.LeftButton: self.clicked.emit()
            def enterEvent(self, e):
                self.setStyleSheet("color: #3182ce; font-weight: bold; font-size: 12px; margin-right: 4px;")
            def leaveEvent(self, e):
                self.setStyleSheet("color: #718096; font-weight: bold; font-size: 12px; margin-right: 4px;")

        def setup_slider(val, callback, hue, def_val=0):
            sl = CircleSlider(Qt.Orientation.Horizontal, base_hue=hue, default_value=def_val)
            sl.setRange(0, 100) 
            sl.setValue(val)
            sl.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            sl.valueChanged.connect(callback)
            return sl
        
        # --- Top Controls ---
        self.lbl_bpm = ClickLabel(f"bpm\n{self.bpm}")
        self.lbl_bpm.clicked.connect(lambda: self.set_bpm(np.random.randint(80, 150)))
        
        self.sl_bpm = CircleSlider(Qt.Orientation.Horizontal, base_hue=180, default_value=BPM_DEFAULT)
        self.sl_bpm.setRange(60, 200) 
        self.sl_bpm.setValue(self.bpm) 
        self.sl_bpm.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.sl_bpm.valueChanged.connect(self.set_bpm)
        header.addWidget(self.lbl_bpm)
        header.addWidget(self.sl_bpm)

        # CHANGED: Updated add_top_control to use randomize_top_param
        def add_top_control(txt, slider_obj, hue):
            lbl = ClickLabel(txt)
            lbl.clicked.connect(lambda: self.randomize_top_param(txt, slider_obj))
            header.addWidget(lbl)
            header.addWidget(slider_obj)

        self.sl_swg = setup_slider(0, self.set_swing, 210, def_val=0)
        self.sl_swg.sliderReleased.connect(self.update_mix) 
        add_top_control("swing", self.sl_swg, 210)

        self.sl_clip = setup_slider(0, self.set_clip, 240)
        add_top_control("cut", self.sl_clip, 240)

        self.sl_evolve = setup_slider(0, self.set_evolve, 270)
        add_top_control("evo", self.sl_evolve, 270)

        self.sl_rev = setup_slider(0, self.set_rev, 300)
        add_top_control("rev", self.sl_rev, 300)

        main.addLayout(header)

        # --- Sequencer Headers ---
        head_row = QWidget()
        head_layout = QHBoxLayout(head_row)
        head_layout.setContentsMargins(0, 0, 0, 2)
        head_layout.setSpacing(0)
        
        lbl_space = QWidget()
        lbl_space.setFixedWidth(55) 
        head_layout.addWidget(lbl_space)

        ctrl_head = QWidget()
        ctrl_head.setFixedWidth(205) 
        ch_layout = QHBoxLayout(ctrl_head)
        ch_layout.setContentsMargins(0, 0, 0, 0)
        ch_layout.setSpacing(0)
        
        btn_space = QWidget()
        btn_space.setFixedWidth(65) 
        ch_layout.addWidget(btn_space)
        
        lbl_cont = QWidget()
        lc_layout = QHBoxLayout(lbl_cont)
        lc_layout.setContentsMargins(2, 0, 2, 0) 
        lc_layout.setSpacing(2)
        
        headers = [("vol", "sl_vol"), ("bit", "sl_crush"), ("flt", "sl_filt"), 
                   ("pch", "sl_pitch"), ("dec", "sl_decay"), ("ton", "sl_tone")]
        for txt, key in headers:
            l = HeaderParamLabel(txt, key)
            l.clicked.connect(self.randomize_column)
            lc_layout.addWidget(l)

        ch_layout.addWidget(lbl_cont)
        head_layout.addWidget(ctrl_head)
        head_layout.addStretch()
        main.addWidget(head_row)
        
        # --- Rows ---
        drums = [("kick", "kick"), ("snare", "snare"), ("hat c", "closed hat"), 
                 ("hat o", "open hat"), ("clap", "clap"), ("wood", "perc a"), ("tom", "perc b")]
        
        for i, (l, t) in enumerate(drums):
            row_hue = int((i / len(drums)) * 280) 
            r = SlotRow(l, t, base_hue=row_hue)
            r.pattern_changed.connect(self.update_mix)
            r.preview_req.connect(self.play_preview)
            r.saved_msg.connect(self.show_notification)
            main.addWidget(r) 
            self.slots.append(r)
            r.update_sound(play=False)
            
            if l == "tom":
                self.reseq = ReseqRow(" reseq", self.bpm)
                self.reseq.pattern_changed.connect(self.update_mix)
                self.reseq.saved_msg.connect(self.show_notification)
                main.addWidget(self.reseq)
                self.slots.append(self.reseq)
                self.reseq.process_audio()

                # Bass Row - Passing Base Hue 300 (Pink)
                self.bass = BassRow(self.bpm, base_hue=300)
                self.bass.pattern_changed.connect(self.update_mix)
                self.bass.saved_msg.connect(self.show_notification)
                main.addWidget(self.bass)
                self.slots.append(self.bass)
        
        self.update_mix()

        # --- Footer ---
        # Reduced Top Margin: 10 -> 4
        bot = QHBoxLayout(); bot.setContentsMargins(0, 4, 0, 0); bot.setSpacing(0)
        
        self.btn_play = PlayButton(); self.btn_play.clicked.connect(self.toggle_play)
        btn_exp = FadeButton("export"); btn_exp.clicked.connect(self.export_beat)
        btn_clr = RedFadeButton("clear"); btn_clr.clicked.connect(self.clear_all)
        self.status_widget = StatusWidget()
        
        bot.addWidget(self.btn_play); bot.addSpacing(5)
        bot.addWidget(btn_exp); bot.addSpacing(15) 
        bot.addWidget(self.status_widget, 1)
        
        bot.addWidget(self.btn_split); bot.addSpacing(5)
        bot.addWidget(btn_clr)
        
        main.addLayout(bot)
    
    def switch_view_page(self, view_idx):
        # Update Visual Offset (0 or 16)
        self.view_offset = view_idx * 16
        
        # Refresh all slots to show the new data (Bank A or Bank B)
        for s in self.slots:
            self.ensure_slot_capacity(s)
            self.refresh_slot_view(s)
            
        # Update Mix to play the new bank
        self.update_mix()
        
        # Trigger an immediate visual update to snap the playhead to the new view
        self.update_playhead()

    def refresh_slot_view(self, s):
        off = self.view_offset
        
        # 1. Standard Pads (SlotRow)
        if isinstance(s, SlotRow):
            # Update SlotRow internal offset for randomizers
            s.view_offset = off
            
            for i in range(16):
                data_idx = off + i
                
                # Fetch Data
                is_active = False
                velocity = 0.8
                
                if data_idx < len(s.pattern):
                    is_active = s.pattern[data_idx]
                
                if data_idx < len(s.velocities):
                    velocity = s.velocities[data_idx]
                
                # 1. Update Visual State (Immediate)
                s.pads[i].force_visual_state(is_active, velocity)
                
                # 2. Update Logic Index (CRITICAL)
                # This ensures when you click this pad later, it sends 'data_idx' (e.g. 16)
                s.pads[i].index = data_idx 
                
                # 3. Update Visual properties
                s.pads[i].is_downbeat = (data_idx % 4 == 0)

        # 2. Bass Piano Roll
        if hasattr(s, 'piano'):
            s.piano.view_offset = off
            s.piano.update()
            
        # 3. Reseq Waveform
        if isinstance(s, ReseqRow):
            s.wave_viz.view_offset = off
            s.update_view_state() 
            s.wave_viz.update()

    def ensure_slot_capacity(self, s):
            # 1. Standard Pattern (Drums + Reseq)
            if hasattr(s, 'pattern') and len(s.pattern) < 32:
                s.pattern.extend([False] * 16)
                s.velocities.extend([0.8] * 16)
            
            # 2. Bass Pattern (Piano Roll)
            if hasattr(s, 'piano') and len(s.piano.pattern) < 32:
                s.piano.pattern.extend([False] * 16)
                s.piano.values.extend([0.5] * 16)
                s.piano.note_levels.extend([0.0] * 16)

    def check_active_length(self):
        """Returns 2 if any note exists beyond step 15, else 1."""
        for s in self.slots:
            # Check Standard Drums / Reseq
            if hasattr(s, 'pattern'):
                if len(s.pattern) > 16:
                    for i in range(16, min(32, len(s.pattern))):
                        if s.pattern[i]: return 2
            
            # Check Bass
            if hasattr(s, 'piano'):
                if len(s.piano.pattern) > 16:
                    for i in range(16, min(32, len(s.piano.pattern))):
                        if s.piano.pattern[i]: return 2
        return 1

    def update_mix(self):
        # 1. Expand Capacity
        for s in self.slots: self.ensure_slot_capacity(s)

        # 2. Setup Loop Info
        self.total_bars = 1
        total_steps = 16 
        
        samples_per_bar = int((60.0 / self.bpm) * 4.0 * SR)
        p_start = self.view_offset
        p_end = p_start + 16
        
        slot_data = []
        for s in self.slots:
            is_sliced = isinstance(s, ReseqRow)
            is_bass = isinstance(s, BassRow)
            
            # --- NEW: Get Label for Sidechain Identification ---
            # BassRow uses 'bass', SlotRow uses label_text
            label_id = "bass" if is_bass else getattr(s, 'label_text', '').lower()
            
            # Ensure Capacity
            if is_bass or is_sliced:
                expected_full = samples_per_bar * 2
                current_len = len(s.current_data) if s.current_data is not None else 0
                if abs(current_len - expected_full) > 4000:
                    if is_bass: s.process_sequence(steps=32, notify=False)
                    elif is_sliced: s.process_audio(steps=32, notify=False)

            # Get Pattern Slice
            if is_bass:
                pat = s.piano.pattern[p_start:p_end]
                vel = s.piano.values[p_start:p_end]
            else:
                pat = s.pattern[p_start:p_end]
                vel = s.velocities[p_start:p_end]

            # Get Audio Slice (Bass/Reseq)
            data_to_mix = s.current_data
            if (is_bass or is_sliced):
                if data_to_mix is None or len(data_to_mix) == 0:
                    data_to_mix = np.zeros(samples_per_bar, dtype=np.float32)
                else:
                    # Robust Slicing
                    if self.view_offset >= 16:
                        if len(data_to_mix) > samples_per_bar: data_to_mix = data_to_mix[samples_per_bar:]
                    else:
                        if len(data_to_mix) > samples_per_bar: data_to_mix = data_to_mix[:samples_per_bar]
                    
                    if len(data_to_mix) > samples_per_bar: data_to_mix = data_to_mix[:samples_per_bar]

            slot_data.append({
                'data': data_to_mix, 
                'pattern': pat, 
                'velocities': vel,
                'is_sliced': is_sliced, 
                'is_bass': is_bass,
                'label': label_id  # <--- NEW FIELD
            })
            
        self.gen.set_data(AudioMixer.mix_sequence(slot_data, self.bpm, self.swing, 
                                                  self.clip_amount, self.reverse_prob, 
                                                  steps=total_steps))

    def set_bpm(self, v):
        if self.bpm == v: return
        self.sl_bpm.setToolTip(f"{v} BPM")
        self.lbl_bpm.setText(f"bpm\n{v}")
        if hasattr(self, 'reseq'): self.reseq.update_bpm(v)
        if hasattr(self, 'bass'): self.bass.update_bpm(v)

        if self.gen.playing:
            loop_duration_old = (60.0 / self.bpm) * 4.0
            elapsed = time.perf_counter() - self.start_time
            phase = (elapsed % loop_duration_old) / loop_duration_old
            self.bpm = v
            loop_duration_new = (60.0 / self.bpm) * 4.0
            self.start_time = time.perf_counter() - (phase * loop_duration_new)
        else:
            self.bpm = v
        self.update_mix()

    def set_swing(self, v):
        self.swing = v / 100.0
        self.sl_swg.setToolTip(f"{v}%")

    def set_clip(self, v):
        self.clip_amount = v / 100.0
        self.sl_clip.setToolTip(f"{v}%")
        self.update_mix()
    
    def set_rev(self, v):
        self.reverse_prob = v / 100.0
        self.sl_rev.setToolTip(f"{v}%")
        self.update_mix()

    def set_evolve(self, v):
        self.evolve_amount = v / 100.0
        self.sl_evolve.setToolTip(f"{v}%")

    def sync_all(self):
        for s in self.slots: s.syncopate_gentle()

    def clear_all(self):
        for s in self.slots: s.clear()

    def play_preview(self, data):
        self.preview.play(data)

    def toggle_play(self):
        if self.gen.playing:
            # PAUSE
            self.gen.set_playback_state(False)
            self.btn_play.set_playing(False)
            
            # Record exactly when we paused
            self.last_pause_timestamp = time.perf_counter()
            
            # Note: We do NOT call reset_vis(), so the highlight stays 
            # exactly where it was when we paused.
        else:
            # RESUME / START
            self.gen.set_playback_state(True)
            self.btn_play.set_playing(True)
            
            if self.last_pause_timestamp == 0:
                # Fresh Start
                self.start_time = time.perf_counter()
            else:
                # Resume Logic:
                # We calculate how long we were paused and shift the start_time 
                # forward by that amount. This makes the "elapsed" calculation 
                # continue seamlessly from where it left off.
                pause_duration = time.perf_counter() - self.last_pause_timestamp
                self.start_time += pause_duration
                
                # Reset pause tracker
                self.last_pause_timestamp = 0

    def update_playhead(self):
        if self.anim_tick_counter % 3 == 0:
            self.logo.animate(); self.btn_play.animate()
            for s in self.slots:
                if hasattr(s, 'pads'):
                    for p in s.pads: p.update_anim()
                if isinstance(s, ReseqRow): s.wave_viz.animate_visuals()
                if isinstance(s, BassRow): s.piano.update_anim()

        self.anim_tick_counter += 1
        if self.anim_tick_counter > 20: 
            self.anim_tick_counter = 0
            top_sliders = [self.sl_bpm, self.sl_swg, self.sl_clip, self.sl_evolve, self.sl_rev]
            for sl in top_sliders: sl.tick_color()
            for s in self.slots:
                if hasattr(s, 'sl_crush'): s.sl_crush.tick_color()
                if hasattr(s, 'sl_filt'): s.sl_filt.tick_color()
                if hasattr(s, 'sl_pitch'): s.sl_pitch.tick_color()
                if hasattr(s, 'sl_decay'): s.sl_decay.tick_color()
                if hasattr(s, 'sl_tone'): s.sl_tone.tick_color()
                if isinstance(s, BassRow): 
                    s.sl_vol.tick_color(); s.sl_release.tick_color()
                    s.sl_tone.tick_color(); s.sl_root_sl.tick_color()
                    s.sl_decay.tick_color(); s.sl_glide.tick_color()

        if not self.gen.playing: return
        
        if self.evolve_amount > 0.01:
            for s in self.slots: 
                if hasattr(s, 'drift_params'): s.drift_params(self.evolve_amount)

        # Time Calculation: Always 0.0 to 16.0 (1 Bar Loop)
        loop_duration = (60.0 / self.bpm) * 4.0 
        elapsed = time.perf_counter() - self.start_time
        local_pos = (elapsed % loop_duration) / loop_duration * 16.0
        
        self.step = int(local_pos)
        if self.step != self.last_processed_step:
            if self.step % 4 == 0: self.logo.on_beat()
            if self.step == 0:
                 for s in self.slots:
                    if hasattr(s, 'evolve_fx_params'): s.evolve_fx_params()
            self.last_processed_step = self.step

        # Send 0.0-16.0 to all slots
        for s in self.slots: s.highlight(local_pos)
    
    def randomize_column(self, param_key):
        self.logo.trigger_flash()
        
        # Map keys to readable names
        names = {
            "sl_vol": "volume", "sl_crush": "bitcrush", "sl_filt": "filter",
            "sl_pitch": "pitch", "sl_decay": "decay", "sl_tone": "tone"
        }
        name = names.get(param_key, param_key.replace("sl_", ""))
        
        for s in self.slots:
            if hasattr(s, param_key):
                slider = getattr(s, param_key)
                new_val = np.random.randint(0, 101)
                slider.setValue(new_val)
                
        self.show_notification(f"randomized: all {name}")

    def reset_vis(self):
        self.step = -1
        # Reset to negative float so gradients disappear
        for s in self.slots: s.highlight(-10.0)

    def show_notification(self, text):
        self.status_widget.set_text(text)

    def export_beat(self):
        self.logo.trigger_flash()
        
        # 1. Gather Data (Using current View Offset logic to export CURRENT pattern)
        # To export the FULL 2-bar sequence (A then B), we would need different logic,
        # but "Pattern Mode" usually implies exporting the active loop.
        # Below exports exactly what is hearing (Active Pattern).
        
        slot_data = []
        samples_per_bar = int((60.0 / self.bpm) * 4.0 * SR)
        p_start = self.view_offset
        p_end = p_start + 16

        for s in self.slots:
            is_sliced = isinstance(s, ReseqRow)
            is_bass = isinstance(s, BassRow)
            
            if is_bass:
                pat = s.piano.pattern[p_start:p_end]
                vel = s.piano.values[p_start:p_end]
            else:
                pat = s.pattern[p_start:p_end]
                vel = s.velocities[p_start:p_end]

            data_to_mix = s.current_data
            if (is_bass or is_sliced) and data_to_mix is not None:
                if self.view_offset >= 16:
                    if len(data_to_mix) > samples_per_bar: data_to_mix = data_to_mix[samples_per_bar:]
                else:
                    if len(data_to_mix) > samples_per_bar: data_to_mix = data_to_mix[:samples_per_bar]

            slot_data.append({
                'data': data_to_mix, 'pattern': pat, 'velocities': vel,
                'is_sliced': is_sliced, 'is_bass': is_bass
            })

        # 2. Generate Mix
        mix = AudioMixer.mix_sequence(slot_data, self.bpm, self.swing, 
                                      self.clip_amount, self.reverse_prob, steps=16)
        
        # 3. Loop it (Create 2 loops for the export file)
        final_mono = np.tile(mix, 2)
        
        # 4. Normalize (Target -0.5 dB => ~0.944)
        peak = np.max(np.abs(final_mono))
        if peak > 0:
            final_mono = final_mono * (0.944 / peak)
            
        # 5. Make Stereo (Stack columns)
        final_stereo = np.column_stack((final_mono, final_mono))
        
        # Save
        home_dir = os.path.expanduser("~")
        save_dir = os.path.join(home_dir, "Music", "sequa")
        if not os.path.exists(save_dir):
            try: os.makedirs(save_dir)
            except: self.show_notification("err: cannot create folder"); return

        timestamp = int(time.time())
        filename = f"sequa_ptn{int(self.view_offset/16)+1}_{self.bpm}bpm_{timestamp}.wav"
        full_path = os.path.join(save_dir, filename)

        try:
            sf.write(full_path, final_stereo, SR)
            self.show_notification(f"saved to: Music/sequa")
        except Exception as e:
             QMessageBox.critical(self, "error", f"could not save: {e}")

if __name__ == '__main__':
    try:
        myappid = 'sequa.audio.tool.v1'
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
    except Exception:
        pass 

    app = QApplication(sys.argv)
    
    icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sequa.ico")
    
    if os.path.exists(icon_path):
        app.setWindowIcon(QIcon(icon_path))
    elif os.path.exists("sequa.ico"):
        app.setWindowIcon(QIcon("sequa.ico"))

    if hasattr(Qt.ApplicationAttribute, 'AA_EnableHighDpiScaling'):
        app.setAttribute(Qt.ApplicationAttribute.AA_EnableHighDpiScaling, True)
    if hasattr(Qt.ApplicationAttribute, 'AA_UseHighDpiPixmaps'):
        app.setAttribute(Qt.ApplicationAttribute.AA_UseHighDpiPixmaps, True)

    app.setStyleSheet("""
        QSlider::groove:horizontal { 
            border: 1px solid #cbd5e0; 
            background: #e2e8f0; 
            height: 4px; 
            border-radius: 2px; 
        }
        QSlider::sub-page:horizontal { 
            background: #4299e1; 
            border-radius: 2px; 
        }
        QSlider::handle:horizontal { 
            background: white; 
            border: 1px solid #4299e1; 
            width: 12px; 
            height: 12px; 
            margin: -5px 0; 
            border-radius: 6px; 
        }
        
        QSlider::groove:vertical { 
            border: 1px solid #cbd5e0; 
            background: #e2e8f0; 
            width: 4px; 
            border-radius: 2px; 
        }
        QSlider::sub-page:vertical { 
            background: #e2e8f0; 
            border-radius: 2px; 
        }
        QSlider::add-page:vertical { 
            background: #4299e1; 
            border-radius: 2px; 
        } 
        QSlider::handle:vertical { 
            background: white; 
            border: 1px solid #4299e1; 
            height: 12px; 
            width: 12px; 
            margin: 0 -5px; 
            border-radius: 6px; 
        }
    """)
    win = SequaWindow()
    win.show()
    sys.exit(app.exec())