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
        self.mutex = QMutex() # Lock
        self.open(QIODevice.OpenModeFlag.ReadOnly)

    def set_playback_state(self, is_playing):
        with QMutexLocker(self.mutex):
            self.playing = is_playing
            if is_playing: self.pos = 0

    def set_data(self, float_data):
        if float_data is None or len(float_data) == 0:
            silent_audio = np.zeros(SR // 2, dtype=np.float32)
            float_data = silent_audio
        
        # 1. Clean & Soft Clip
        clean_data = np.nan_to_num(float_data, copy=False, nan=0.0, posinf=1.0, neginf=-1.0)
        audio = np.tanh(clean_data)
        
        # 2. Convert to Int16
        audio = (audio * 32767).astype(np.int16)
        new_bytes = audio.tobytes()
        
        with QMutexLocker(self.mutex):
            # 3. Seamless Playhead Transfer
            current_len = len(self.data)
            new_len = len(new_bytes)
            
            if current_len > 0 and new_len > 0 and self.pos > 0:
                progress = self.pos / current_len
                self.pos = int(progress * new_len)
            
            # STRICT Even Alignment to prevent byte-swap crunch
            self.pos &= ~1 
            
            self.data = new_bytes
            # Safety clamp
            if self.pos >= len(self.data): self.pos = 0

    def readData(self, maxlen):
        with QMutexLocker(self.mutex):
            if not self.playing or not self.data:
                return b'\x00' * maxlen

            # Align request to 2-byte boundary
            if maxlen % 2 != 0: maxlen -= 1
            
            chunk = b''
            data_len = len(self.data)
            
            if data_len == 0: return b'\x00' * maxlen
            
            if self.pos >= data_len: self.pos = 0
            
            remaining = data_len - self.pos
            
            if maxlen <= remaining:
                chunk = self.data[self.pos:self.pos + maxlen]
                self.pos += maxlen
            else:
                # Wrap around logic
                part1 = self.data[self.pos:]
                needed = maxlen - len(part1)
                
                if needed > data_len:
                    repeats = needed // data_len
                    remainder = needed % data_len
                    part2 = self.data * repeats + self.data[:remainder]
                else:
                    part2 = self.data[:needed]
                
                chunk = part1 + part2
                self.pos = needed
            
            # Final alignment check
            if len(chunk) < maxlen:
                chunk += b'\x00' * (maxlen - len(chunk))
            elif len(chunk) > maxlen:
                chunk = chunk[:maxlen]
            
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
            f_end = 45
            # Slower pitch drop for a "boom" rather than a "click"
            freq_env = (f_start - f_end) * np.exp(-t * 12) + f_end
            phase = np.cumsum(freq_env) * 2 * np.pi / SR
            
            # Use sin directly (less saturation than before)
            osc = np.sin(phase)
            
            # Gentle amp envelope
            d_amp = 6 + ((1.0 - p_decay) * 40)
            
            # Lowpass to ensure "Sine"
            sos_lp = signal.butter(2, 5000, 'lp', fs=SR, output='sos')
            y = signal.sosfilt(sos_lp, osc) * np.exp(-t * d_amp)

        elif drum_type == "snare":
            # "Boxy" Analog Snare - Sine heavy
            f_body = 170 + (p_pitch * 40)
            # Pure sine body
            body = np.sin(2 * np.pi * f_body * t) * np.exp(-t * 20)
            
            noise = np.random.uniform(-0.5, 0.5, len(t))
            # Soft Bandpass for the noise (brush-like)
            f_center = 2500 + (p_tone * 1500)
            sos_bp = signal.butter(2, [f_center - 1000, f_center + 1000], 'bp', fs=SR, output='sos')
            noise = signal.sosfilt(sos_bp, noise)
            
            d_noise = 15 + ((1.0 - p_decay) * 40)
            # Mix: Body is dominant for the "soft" feel
            y = (body * 0.8) + (noise * np.exp(-t * d_noise) * 0.4)

        elif "hat" in drum_type or "cymbal" in drum_type:
            # Replaced harsh square waves with "Filtered Noise + High Sine"
            # This creates a soft "Shaker/CR-78" hat vibe.
            
            # 1. The "Hiss" (Bandpassed Noise)
            noise = np.random.uniform(-1, 1, len(t))
            bp_center = 7000 + (p_tone * 3000)
            sos_bp = signal.butter(2, [bp_center - 2000, bp_center + 2000], 'bp', fs=SR, output='sos')
            hiss = signal.sosfilt(sos_bp, noise)
            
            # 2. The "Ring" (High Sine for metallic color)
            f_metal = 800 + (p_pitch * 300)
            # FM modulation for texture
            mod = np.sin(2 * np.pi * (f_metal * 3.5) * t) * 50
            metal = np.sin(2 * np.pi * (f_metal + mod) * t) * 0.2
            
            sig = hiss + metal
            
            if "closed" in drum_type:
                decay = 60 + ((1.0 - p_decay) * 200)
                # Very fast attack softening
                attack = np.minimum(t * 2000, 1.0)
                y = sig * attack * np.exp(-t * decay)
            else:
                decay = 10 + ((1.0 - p_decay) * 30)
                y = sig * np.exp(-t * decay) * 0.8

        elif drum_type == "clap":
            # Analog Clap: Soft filtered noise
            noise = np.random.uniform(-1, 1, len(t))
            # Lower bandpass for a "warmer" clap
            bp_low = 700 + (p_pitch * 200)
            bp_high = bp_low + 600
            sos = signal.butter(2, [bp_low, bp_high], 'bp', fs=SR, output='sos')
            filt = signal.sosfilt(sos, noise)
            
            # Simple envelope, soft attack
            env = np.exp(-t * (10 + (1.0 - p_decay) * 30))
            attack = min(len(t), int(SR * 0.015)) # 15ms attack
            env[:attack] *= np.linspace(0, 1, attack)
            y = filt * env

        elif drum_type == "wood" or "perc" in drum_type and "a" in drum_type:
            # Woodblock: "Bloop" sound
            # Added slight pitch chirp for that Rhythm Box character
            f_base = 700 + (p_pitch * 400)
            f_env = f_base * (1.0 + 0.1 * np.exp(-t * 50))
            
            phase = np.cumsum(f_env) * 2 * np.pi / SR
            osc = np.sin(phase)
            
            # Short, percussive envelope
            decay = 30 + ((1.0 - p_decay) * 100)
            y = osc * np.exp(-t * decay)

        else:
            # Tom: Pure Sine Sweep
            f = 140 + (p_pitch * 100)
            # Gentle sweep
            f_env = f * (1.0 - 0.2 * np.exp(-t * 10))
            phase = np.cumsum(f_env) * 2 * np.pi / SR
            y = np.sin(phase) * np.exp(-t * (8 + (1.0 - p_decay) * 25))
            
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
            y *= np.exp(-t * (30 + (1.0-p_decay)*50))
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
            # Deep Sine Kick
            # 1. Pitch Sweep: Start HIGHER (200Hz) for more "Punch/Click"
            f_start = 200 + (p_pitch * 50) 
            f_end = 40
            
            # Envelope: Faster transient drop (0.008) for tighter impact
            decay_transient = 0.008
            
            freq_env = (f_start - f_end) * np.exp(-t / decay_transient) + f_end
            phase = np.cumsum(freq_env) * 2 * np.pi / SR
            
            # 2. Waveform: Pure Sine
            y = np.sin(phase)
            
            # 3. Transient: Thud
            thud_noise = np.random.uniform(-1, 1, len(t))
            sos_thud = signal.butter(2, 800, 'lp', fs=SR, output='sos')
            thud = signal.sosfilt(sos_thud, thud_noise) * np.exp(-t * 200)
            
            # Amplitude Envelope
            # CHANGED: Base decay increased from 2 to 6 for a much tighter, shorter tail
            # Range is now much wider, allowing short kicks by default
            amp_decay = 6 + ((1.0 - p_decay) * 15)
            y = (y + thud * 0.5) * np.exp(-t * amp_decay)

        elif drum_type == "snare":
            f_body = 140 + (p_pitch * 120)
            tone_osc = np.sin(2 * np.pi * f_body * t) * np.exp(-t * (30 + (1.0-p_decay)*60))
            filt_center = 1000 + (p_tone * 5000)
            noise = rng.uniform(-1, 1, len(t))
            noise = signal.sosfilt(signal.butter(2, [filt_center, filt_center+2000], 'bp', fs=SR, output='sos'), noise)
            noise_env = np.exp(-t * (30 + ((1.0-p_decay) * 80)))
            y = (tone_osc * 0.5) + (noise * noise_env * 0.8)

        elif drum_type == "closed hat" or drum_type == "open hat":

            base_f = 350 + (p_pitch * 200) 
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
                # Made decay slightly faster to sound "crisper"
                decay_coef = 90 + ((0.75 - p_decay) * 300)
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
            pulse_spacing = 0.011
            
            # Decay affects the tail length
            tail_decay = 30 + ((1.0 - p_decay) * 60)
            
            for i in range(4):
                start_idx = int(i * pulse_spacing * SR)
                if start_idx >= len(env): break
                
                # Last hit is the loudest (1.0), pre-hits are transients (0.7)
                amp = 0.7 if i < 3 else 1.0
                
                remaining = len(env) - start_idx
                local_t = np.linspace(0, remaining/SR, remaining)
                
                # Extremely sharp decay (350) for the flam hits
                decay = 300 if i < 3 else tail_decay
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
        
        duration = 0.8
        t = np.linspace(0, duration, int(SR * duration), endpoint=False)
        y = np.zeros_like(t)

        if drum_type == "kick":
            f_start = 150 + (p_pitch * 250)
            f_end = 45 + (p_pitch * 30)
            # Increased decay speed slightly for tighter punch
            f_decay = 30 + (p_decay * 50) 
            freq_env = (f_start - f_end) * np.exp(-t * f_decay) + f_end
            phase = np.cumsum(freq_env) * 2 * np.pi / SR
            
            drive = 1.0 + (p_tone * 4.0)
            osc = np.tanh(np.sin(phase) * drive)
            
            amp_decay = 5 + ((1.0 - p_decay) * 45)
            # Louder click, sharper envelope for punch
            click = np.random.normal(0, 0.5, len(t)) * np.exp(-t * 300)
            y = (osc + click * 0.4) * np.exp(-t * amp_decay)

        elif drum_type == "snare":
            f_root = 170 + (p_pitch * 40)
            f_env = f_root * (1.0 - 0.05 * t)
            tone = np.sin(np.cumsum(f_env) * 2 * np.pi / SR)
            tone_env = np.exp(-t * 25)
            
            noise = np.random.uniform(-1, 1, len(t))
            hp_freq = 1500 + (p_tone * 1000)
            sos_hp = signal.butter(4, hp_freq, 'hp', fs=SR, output='sos')
            noise = signal.sosfilt(sos_hp, noise)
            
            noise_decay = 20 + ((1.0 - p_decay) * 40)
            noise_env = np.exp(-t * noise_decay)
            
            y = (tone * tone_env * 0.3) + (noise * noise_env * 0.9)
            y = np.tanh(y * 1.6)

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
                # Reduced decay rate significantly so it's not just a click
                # (Lower number in exp = longer sound)
                decay_rate = 50 + ((1.0 - p_decay) * 150)
                y = sig * attack * np.exp(-t * decay_rate) * 0.6
            else:
                hp_freq = 6000 + (p_tone * 2000)
                sos_hp = signal.butter(4, hp_freq, 'hp', fs=SR, output='sos')
                sig = signal.sosfilt(sos_hp, sig)
                
                decay = 10 + ((1.0 - p_decay) * 35)
                y = sig * np.exp(-t * decay)
                y = np.tanh(y * 1.0) * 0.7
            
            # Subtle Lowpass to soften digital harshness
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
            f1 = 100 + (p_pitch * 60)
            f2 = 300 + (p_pitch * 50)
            
            osc1 = np.sin(2 * np.pi * f1 * t)
            osc2 = np.sin(2 * np.pi * f2 * t)
            
            noise = np.random.uniform(-1, 1, len(t))
            # Shorter, sharper transient for "snap"
            transient = noise * np.exp(-t * 600) * 0.3

            raw = (osc1 * osc2) + transient

            low_cut = 300
            high_cut = 2500
            sos_bp = signal.butter(2, [low_cut, high_cut], 'bp', fs=SR, output='sos')
            y = signal.sosfilt(sos_bp, raw)
            
            decay = 60 + ((1.0-p_decay) * 200)
            y *= np.exp(-t * decay)
            y = np.tanh(y * 3.0) * 0.8

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
        # (Paste your existing process_sample logic here)
        # It remains the same as before.
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
    def resample_lofi(data, crush_val):
        if crush_val <= 0.01: return data

        reduction = 1.0 + (crush_val * 2.5)
        
        orig_len = len(data)
        target_len = max(1, int(orig_len / reduction))
        lo = signal.resample(data, target_len)

        bits = 16 - (crush_val * 5) 
        
        steps = 2 ** bits
        lo = np.round(lo * steps) / steps
        restored = signal.resample(lo, orig_len).astype(np.float32)
        return SynthEngine.ensure_zero_crossing(np.clip(restored, -1.0, 1.0))

    @staticmethod
    def apply_filter(data, val):
        # (Paste existing apply_filter)
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
    def apply_filter(data, val):
        # 0.0 (Bottom) = Lowpass (Muffled) -> 0.5 (Center) = Open -> 1.0 (Top) = Highpass (Thin)
        if 0.45 < val < 0.55: return data # Deadzone at center
        
        if val <= 0.45:
            # Lowpass: 0.0 is ~150Hz, 0.45 is ~18kHz
            norm = val / 0.45
            # Logarithmic-ish feel
            cutoff = 150 + (norm**2 * 18000)
            sos = signal.butter(2, cutoff, 'lp', fs=SR, output='sos')
        else:
            # Highpass: 0.55 is ~20Hz, 1.0 is ~8000Hz
            norm = (val - 0.55) / 0.45
            cutoff = 20 + (norm**2 * 8000)
            sos = signal.butter(2, cutoff, 'hp', fs=SR, output='sos')
            
        return signal.sosfilt(sos, data).astype(np.float32)

class AudioMixer:
    @staticmethod
    def mix_sequence(slots, bpm, swing, clip_val, rev_prob, steps=STEPS):
        sec_beat = 60.0 / bpm
        sec_step = sec_beat / 4.0
        total_samples = int(sec_step * steps * SR)
        
        # FORCE EVEN LENGTH
        if total_samples % 2 != 0: total_samples += 1

        swing_offset = int(sec_step * swing * 0.33 * SR)
        
        out = np.zeros(total_samples + int(SR * 0.5), dtype=np.float32)

        for s in slots:
            raw_data = s['data']
            if raw_data is None or len(raw_data) == 0: continue
                
            is_sliced = s.get('is_sliced', False)
            
            # --- SHARED PARAMETERS ---
            s_pattern = s['pattern']
            s_vels = s['velocities']

            if not is_sliced:
                # === ONE SHOT (DRUMS) LOGIC ===
                # (Existing logic for drums)
                max_seq_len = total_samples + int(SR * 0.5)
                
                if clip_val > 0.0:
                    keep_ratio = 1.0 / (1.0 + (clip_val * 20.0))
                    actual_len = max(150, int(len(raw_data) * keep_ratio))
                    data_fwd = raw_data[:actual_len].copy()
                    fade_samples = min(200, int(actual_len * 0.4))
                    if fade_samples > 0:
                        data_fwd[-fade_samples:] *= np.linspace(1, 0, fade_samples)
                else:
                    limit = min(len(raw_data), max_seq_len)
                    data_fwd = raw_data[:limit].copy()

                fade_in = min(100, len(data_fwd) // 10)
                if fade_in > 0: data_fwd[:fade_in] *= np.linspace(0, 1, fade_in)

                if rev_prob > 0.0: data_rev = np.ascontiguousarray(data_fwd[::-1])
                else: data_rev = data_fwd

                s_len = len(data_fwd)
                
                for i, (active, vel) in enumerate(zip(s_pattern, s_vels)):
                    if active:
                        start_pos = int(i * sec_step * SR)
                        if i % 2 != 0: start_pos += swing_offset
                        
                        if start_pos < len(out):
                            current_sample = data_rev if (rev_prob > 0.0 and np.random.random() < rev_prob) else data_fwd
                            write_len = min(s_len, len(out) - start_pos)
                            if write_len > 0:
                                gain = (vel ** 1.5) * 0.8
                                out[start_pos:start_pos + write_len] += current_sample[:write_len] * gain

            else:
                # === SLICER (RESEQ) LOGIC ===
                # Fixed: Now handles Swing and Gating here, ensuring sync with drums
                
                # Length of one logical step in the source audio
                src_step_len = len(raw_data) // steps
                if src_step_len < 100: continue # Safety check

                fade_samples = min(300, src_step_len // 4)

                for i, (active, vel) in enumerate(zip(s_pattern, s_vels)):
                    if active:
                        # 1. Source Coordinates (Strict Grid)
                        src_start = i * src_step_len
                        src_end = src_start + src_step_len
                        
                        # 2. Destination Coordinates (Swung Grid)
                        dst_start = int(i * sec_step * SR)
                        if i % 2 != 0: dst_start += swing_offset
                        
                        # Bounds checks
                        if src_end > len(raw_data): src_end = len(raw_data)
                        if dst_start >= len(out): continue

                        # 3. Extract Slice
                        chunk = raw_data[src_start:src_end].copy()
                        
                        # 4. Apply Velocity
                        chunk *= (vel ** 1.5)

                        # 5. Micro-fades (De-clicking)
                        if fade_samples > 0:
                            chunk[:fade_samples] *= np.linspace(0, 1, fade_samples)
                            chunk[-fade_samples:] *= np.linspace(1, 0, fade_samples)

                        # 6. Write to Mix
                        write_len = min(len(chunk), len(out) - dst_start)
                        if write_len > 0:
                            # Gain reduction to prevent clipping when summing full loops
                            out[dst_start:dst_start+write_len] += chunk[:write_len] * 0.85

        # Wrap Tail
        tail = out[total_samples:]
        wrap_len = min(len(tail), total_samples)
        out[:wrap_len] += tail[:wrap_len]
        
        final = out[:total_samples]
        
        peak = np.max(np.abs(final))
        if peak > 1.0:
            final = np.tanh(final * 0.8) * 0.95
        
        return final

# --- UI Components ---

class LogoWidget(QWidget):
    kit_changed = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(60, 60)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.grid_size = 4
        self.cell_size = 10
        self.current = np.random.uniform(180, 255, (4, 4, 3))
        self.targets = np.random.uniform(180, 255, (4, 4, 3))
        self.flash_val = 0.0 
        
        self.kit_index = 2 # Starts at eight (Indices: 0=simple, 1=pcm, 2=eight, 3=nine)
        self.kit_names = ["simple", "pcm", "eight", "nine"]
        self.text_alpha = 0.0

    def trigger_flash(self):
        self.flash_val = 1.0

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            # Cycle 6 -> 7 -> 8 -> 9
            self.kit_index = (self.kit_index + 1) % 4
            self.trigger_flash()
            self.text_alpha = 1.0
            self.kit_changed.emit(self.kit_index)

    def animate(self):
        if self.flash_val > 0.001: self.flash_val *= 0.94
        else: self.flash_val = 0.0
        
        if self.text_alpha > 0.01: self.text_alpha *= 0.95
        
        speed = 0.08 + (self.flash_val * 0.25)
        self.current += (self.targets - self.current) * speed
        
        prob = 0.3 + (self.flash_val * 0.4)
        if np.random.random() < prob:
            r, c = np.random.randint(0, self.grid_size, 2)
            self.targets[r,c] = [np.random.randint(120, 190), 
                                 np.random.randint(190, 230), 
                                 np.random.randint(230, 255)]
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # 1. Draw Grid (Existing Code)
        grid_w = self.grid_size * self.cell_size
        off_x = (self.width() - grid_w) / 2
        off_y = (self.height() - grid_w) / 2 
        painter.setPen(Qt.PenStyle.NoPen)
        center_idx = 1.5 
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                rgb = self.current[r,c].astype(int)
                col = QColor(*rgb)
                if self.flash_val > 0.1: col = col.lighter(int(100 + (self.flash_val * 60)))
                dist = np.sqrt((r - center_idx)**2 + (c - center_idx)**2)
                alpha = int(np.clip(240 - (dist * 80), 0, 255))
                col.setAlpha(alpha)
                x = off_x + c * self.cell_size
                y = off_y + r * self.cell_size
                rect = QRectF(x + 1, y + 1, self.cell_size - 2, self.cell_size - 2)
                grad = QLinearGradient(rect.topLeft(), rect.bottomRight())
                grad.setColorAt(0, col)
                col_dark = col.darker(105)
                col_dark.setAlpha(alpha)
                grad.setColorAt(1, col_dark)
                painter.setBrush(grad)
                painter.drawRoundedRect(rect, 3.0, 3.0)

        # 2. Draw "Swap" text on hover (subtle)
        if self.underMouse() and self.text_alpha < 0.2:
            painter.setPen(QColor(100, 120, 140, 150))
            f = QFont("Segoe UI", 8)
            f.setBold(True)
            painter.setFont(f)
            painter.drawText(self.rect().adjusted(0,0,0,-4), Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignHCenter, "swap")

        # 3. Draw Kit Number (fading in/out)
        if self.text_alpha > 0.01:
            painter.setOpacity(self.text_alpha)
            f = QFont("Segoe UI", 12, QFont.Weight.Bold)
            painter.setFont(f)
            painter.setPen(QColor(40, 60, 80))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self.kit_names[self.kit_index])

class PlayButton(QPushButton):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(80, 26)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.playing = False
        
        # Animation State (2 Gradient Stops)
        # Initialize with the original blueish tone range
        self.current = np.array([[63., 108., 155.], [80., 130., 170.]])
        self.targets = np.array([[63., 108., 155.], [80., 130., 170.]])

    def set_playing(self, state):
        self.playing = state
        self.update()

    def animate(self):
        # 1. Smooth Interpolation
        self.current += (self.targets - self.current) * 0.05
        
        # 2. Pick new targets occasionally
        if np.random.random() < 0.02:
            # Constrain to "Pastel Blue/Teal" theme to match UI
            # R: 50-90, G: 100-150, B: 160-210
            t1 = [np.random.randint(50, 90), np.random.randint(100, 150), np.random.randint(160, 210)]
            t2 = [np.random.randint(50, 90), np.random.randint(100, 150), np.random.randint(160, 210)]
            self.targets = np.array([t1, t2])
            
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Gradient Background
        grad = QLinearGradient(0, 0, self.width(), 0)
        
        # Convert floats to integers for QColor
        c1 = QColor(*self.current[0].astype(int))
        c2 = QColor(*self.current[1].astype(int))
        
        # Interaction Tint
        if self.underMouse():
            c1, c2 = c1.lighter(110), c2.lighter(110)
        if self.isDown():
            c1, c2 = c1.darker(110), c2.darker(110)
            
        grad.setColorAt(0, c1)
        grad.setColorAt(1, c2)
        
        painter.setBrush(grad)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(self.rect(), 13, 13)
        
        # Icon
        painter.setBrush(QColor("white"))
        cx, cy = self.width() / 2, self.height() / 2
        if self.playing:
            painter.drawRoundedRect(QRectF(cx - 3, cy - 3, 6, 6), 1, 1)
        else:
            path = QPainterPath()
            path.moveTo(cx - 2, cy - 5)
            path.lineTo(cx - 2, cy + 5)
            path.lineTo(cx + 6, cy)
            path.closeSubpath()
            painter.drawPath(path)

class ClearButton(QPushButton):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(20, 20)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Dynamic sizing based on widget dimensions
        s = min(self.width(), self.height())
        margin = 1
        rect = QRectF(margin, margin, s - (margin * 2), s - (margin * 2))
        
        bg = QColor("white")
        border = QColor("#feb2b2")
        dot_color = QColor("#e53e3e")
        
        if self.isDown():
            bg = QColor("#fee2e2")
        elif self.underMouse():
            bg = QColor("#fff5f5")
            border = QColor("#e53e3e")
            
        painter.setBrush(bg)
        painter.setPen(QPen(border, 1))
        painter.drawEllipse(rect)
        
        # Center dot
        painter.setBrush(dot_color)
        painter.setPen(Qt.PenStyle.NoPen)
        cx, cy = s / 2.0, s / 2.0
        
        # Changed from s/5.0 to s/7.0 for a smaller dot
        r = s / 7.0 
        painter.drawEllipse(QPointF(cx, cy), r, r)

class StepPad(QWidget):
    toggled = pyqtSignal(bool, float)
    velocity_changed = pyqtSignal(float)

    def __init__(self, index, base_hue, parent=None):
        super().__init__(parent)
        self.index = index
        self.base_hue = base_hue 
        self.active = False
        self.velocity = 0.8
        self.is_playing_head = False
        self.flash_val = 0.0
        self.is_downbeat = (index % 4 == 0)

        self.setFixedWidth(28)
        # Change Policy to FIXED horizontal, EXPANDING vertical.
        # This prevents the pads from stretching wider than 28px.
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)

        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setMouseTracking(True)

    def set_playing(self, playing):
        # Trigger the flash when the playhead lands on this pad
        if self.is_playing_head != playing:
            self.is_playing_head = playing
            
            # If the pad is active and just got hit by playhead, trigger max flash
            if playing and self.active:
                self.flash_val = 1.0
            
            self.update()

    def process_mouse_input(self, local_pos, force_state=None):
        h = self.height()
        # Ensure y is within bounds for calculation
        y = max(0.0, min(float(h), local_pos.y()))
        new_vel = max(0.1, min(1.0, 1.0 - (y / h)))

        if force_state is not None:
            # Force specific state (Used for drag/paint)
            if self.active != force_state:
                self.active = force_state
                if self.active: self.velocity = new_vel
                self.toggled.emit(self.active, self.velocity)
            elif self.active:
                # Update velocity if already active
                if abs(new_vel - self.velocity) > 0.01:
                    self.velocity = new_vel
                    self.velocity_changed.emit(self.velocity)
        else:
            # Toggle logic (Standard click)
            self.active = not self.active
            if self.active: self.velocity = new_vel
            self.toggled.emit(self.active, self.velocity)
        self.update()

    def mousePressEvent(self, event):
        if event.button() in (Qt.MouseButton.LeftButton, Qt.MouseButton.RightButton):
            self.drag_start_pos = event.position() if hasattr(event, 'position') else event.localPos()
            self.was_active_at_press = self.active
            self.is_drag_active = False
            
            # If inactive, activate immediately (Standard step-sequencer behavior)
            if not self.active:
                self.process_mouse_input(event.pos(), force_state=True)
            
            # Initialize painting for dragging across neighbors
            if self.parent() and hasattr(self.parent(), 'start_painting'):
                # Paint target is opposite of what it was, or True if we just activated
                target = not self.was_active_at_press if self.was_active_at_press else True
                self.parent().start_painting(target)

    def mouseMoveEvent(self, event):
        if event.buttons() & (Qt.MouseButton.LeftButton | Qt.MouseButton.RightButton):
            # 1. Detect Drag Threshold (for velocity adjustment on single pad)
            if not self.is_drag_active:
                cur_pos = event.position() if hasattr(event, 'position') else event.localPos()
                if (cur_pos - self.drag_start_pos).manhattanLength() > 5:
                    self.is_drag_active = True
            
            if self.rect().contains(event.pos()):
                # If dragging inside this pad, update velocity
                if self.is_drag_active:
                    self.process_mouse_input(event.pos(), force_state=True)
            else:
                # 2. Neighbor Painting (Drag across pads)
                parent = self.parent()
                if parent:
                    pos_in_parent = self.mapTo(parent, event.pos())
                    child = parent.childAt(pos_in_parent)
                    if isinstance(child, StepPad) and child != self:
                         if hasattr(parent, 'paint_state'):
                            child.process_mouse_input(child.mapFrom(parent, pos_in_parent), force_state=parent.paint_state)

    def mouseReleaseEvent(self, event):
        # If it was a fast click (no drag detected) and pad was originally active, toggle it OFF
        if not self.is_drag_active and self.was_active_at_press and self.active:
             if self.rect().contains(event.pos()):
                self.process_mouse_input(event.pos(), force_state=False)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        r = self.rect().adjusted(1, 1, -1, -1)
        
        # 1. Base Pad Rendering
        if self.active:
            # Dynamic Hue (Wave Effect)
            t = time.time() * 2.5
            hue_offset = np.sin(t + (self.index * 0.2)) * 12
            current_hue = int((self.base_hue + hue_offset) % 360)
            
            # Base color
            c = QColor.fromHsl(current_hue, 160, 140)
            base_alpha = int(180 + (self.velocity * 75))
            c.setAlpha(base_alpha)
            
            painter.setBrush(c)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawRoundedRect(r, 3, 3)

            if self.flash_val > 0.01:
                cx, cy = r.center().x(), r.center().y()
                rad = max(r.width(), r.height()) * 0.8
                
                grad = QRadialGradient(cx, cy, rad)

                glow_alpha = int(self.flash_val * 160)
                grad.setColorAt(0, QColor(255, 255, 255, glow_alpha))
                grad.setColorAt(1, QColor(255, 255, 255, 0))
                
                painter.setBrush(grad)
                painter.setPen(Qt.PenStyle.NoPen)
                painter.drawRoundedRect(r, 3, 3)

        else:
            # Inactive Pad
            painter.setBrush(QColor("#e2e8f0") if self.is_downbeat else QColor("#edf2f7"))
            painter.setPen(QPen(QColor("#cbd5e0"), 1))
            painter.drawRoundedRect(r, 3, 3)

        # 3. Playhead (Dark Blue) - Only on Inactive Pads
        if self.is_playing_head and not self.active:
             painter.setBrush(QColor(40, 60, 90, 50))
             painter.setPen(Qt.PenStyle.NoPen)
             painter.drawRoundedRect(r, 3, 3)

        # 4. Velocity Line (Active Only)
        if self.active:
            ly = max(r.y()+2, min(r.bottom()-2, int(r.y() + r.height() * (1.0 - self.velocity))))
            painter.setPen(QPen(QColor(255, 255, 255, 200), 1))
            painter.drawLine(r.x()+4, ly, r.right()-4, ly)

        # 5. Animation Loop
        if self.flash_val > 0.01:
            self.flash_val *= 0.96 
            self.update()
        else:
            self.flash_val = 0.0

class DraggableLabel(QLabel):
    file_dropped = pyqtSignal(str)

    def __init__(self, text, parent=None):
        super().__init__(text, parent)
        self.setAcceptDrops(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setToolTip("Drag & Drop or Click to load audio file")
        # Make it look slightly interactive
        self.setStyleSheet("""
            QLabel { 
                font-size: 12px; color: #4a5568; font-weight: bold; 
                margin-left: 2px; border: 1px dashed transparent;
                border-radius: 4px; padding: 2px;
            }
            QLabel:hover { background-color: #ebf8ff; border-color: #90cdf4; color: #2b6cb0; }
        """)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        if urls:
            path = urls[0].toLocalFile()
            self.file_dropped.emit(path)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            fname, _ = QFileDialog.getOpenFileName(
                self, "open", "", "Audio Files (*.wav *.mp3 *.aif *.flac)"
            )
            if fname:
                self.file_dropped.emit(fname)

class GradientLabel(QWidget):
    clicked = pyqtSignal()

    def __init__(self, text, parent=None):
        super().__init__(parent)
        self.text = text
        self.phase = 0.0
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.animate)
        self.timer.start(50)
        # Font size 12 to match DraggableLabel
        self.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Preferred)
        self.setFixedWidth(55)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def mousePressEvent(self, event):
        self.clicked.emit()

    def animate(self):
        self.phase = (self.phase + 0.02) % 1.0
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
        # Adjust rect to match DraggableLabel margin
        r = self.rect().adjusted(2, 0, 0, 0)
        painter.drawText(r, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, self.text)

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

    def __init__(self, steps=STEPS, parent=None):
        super().__init__(parent)
        self.steps = steps
        self.pattern = [False] * steps
        self.velocities = [0.8] * steps
        self.waveform_data = None
        
        self.setAcceptDrops(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setMouseTracking(True)

        self.setFixedWidth(448)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)
        
        self.play_step = -1
        self.hover_step = -1
        self.cached_poly = None
        
        # Interaction state
        self.is_dragging = False
        self.drag_start_y = 0
        self.drag_state = True # What we are painting (True/False)
        self.last_step_idx = -1
        self.initial_vel_at_click = 0.8

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

    def set_playing_step(self, step):
        if self.play_step != step:
            self.play_step = step
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
        idx = int(event.pos().x() / step_w)
        
        if 0 <= idx < self.steps:
            # Toggle logic
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
        idx = int(event.pos().x() / step_w)
        
        # 1. Handle Hover Visuals
        if idx != self.hover_step:
            self.hover_step = idx
            self.update()

        # 2. Handle Interaction
        if self.is_dragging and 0 <= idx < self.steps:
            
            # A. Horizontal Paint (New Step)
            if idx != self.last_step_idx:
                self.pattern[idx] = self.drag_state
                # Reset vertical reference for the new step to current Y
                self.drag_start_y = event.pos().y() 
                self.initial_vel_at_click = self.velocities[idx]
                self.last_step_idx = idx
                self.pattern_changed.emit()
            
            # B. Vertical Velocity Drag (Current Step)
            # Only adjust velocity if the step is Active
            if self.pattern[idx]:
                dy = self.drag_start_y - event.pos().y()
                # Sensitivity: Full height = full range roughly
                vel_delta = dy / 200.0 
                new_vel = np.clip(self.initial_vel_at_click + vel_delta, 0.1, 1.0)
                
                if abs(new_vel - self.velocities[idx]) > 0.01:
                    self.velocities[idx] = new_vel
                    # No pattern_changed emit here to avoid re-generating audio constantly
                    # But we need visual update
                    self.update()

    def mouseReleaseEvent(self, event):
        self.is_dragging = False
        # Emit one final change for velocity updates
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
        
        # Background
        painter.setBrush(QColor("#edf2f7"))
        painter.setPen(QPen(QColor("#cbd5e0"), 1))
        painter.drawRoundedRect(r, 3, 3)

        if self.waveform_data is None:
            painter.setPen(QColor("#a0aec0"))
            painter.setFont(QFont("Segoe UI", 9))
            painter.drawText(r, Qt.AlignmentFlag.AlignCenter, "drag audio here")
            return

        # Generate Poly
        if self.cached_poly is None and self.waveform_data is not None:
            pts = []
            cy = h / 2
            amp = (h / 2) * 0.95  # Increased amplitude slightly
            x_step = w / len(self.waveform_data)
            
            pts.append(QPointF(0, cy))
            for i, val in enumerate(self.waveform_data):
                pts.append(QPointF(i * x_step, cy + abs(val) * amp))
            pts.append(QPointF(w, cy))
            
            for i in range(len(self.waveform_data)-1, -1, -1):
                val = self.waveform_data[i]
                pts.append(QPointF(i * x_step, cy - abs(val) * amp))
            
            self.cached_poly = QPolygonF(pts)

        painter.save()
        painter.translate(r.topLeft())

        # Draw Full Gradient Waveform (Clipped to box)
        if self.cached_poly:
            path = QPainterPath()
            path.addRoundedRect(0, 0, w, h, 3, 3)
            painter.setClipPath(path)
            
            # UPDATED: More saturated gradient for better visibility
            grad = QLinearGradient(0, 0, w, 0)
            grad.setColorAt(0, QColor("#3182ce")) # Stronger Blue
            grad.setColorAt(1, QColor("#805ad5")) # Stronger Purple
            painter.setBrush(QBrush(grad))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawPolygon(self.cached_poly)

        step_w = w / self.steps
        
        # Draw Overlays
        for i in range(self.steps):
            x = i * step_w
            is_active = self.pattern[i]
            is_play = (i == self.play_step)
            is_hover = (i == self.hover_step)
            
            rect = QRectF(x, 0, step_w, h)

            if is_play:
                # Playhead highlight
                painter.setBrush(QColor(255, 255, 255, 120))
                painter.setPen(Qt.PenStyle.NoPen)
                painter.drawRect(rect)
            elif not is_active:
                # Inactive: Dim out the waveform significantly (higher alpha white)
                # This creates high contrast between Active (Color) and Inactive (Pale)
                painter.setBrush(QColor(240, 244, 248, 210))
                painter.setPen(Qt.PenStyle.NoPen)
                painter.drawRect(rect)
                
                if is_hover:
                    painter.setBrush(QColor(255, 255, 255, 100))
                    painter.drawRect(rect)
            else:
                # Active: Apply Saturation based on Velocity
                # UPDATED: Adjusted range. High velocity = True Color. Low velocity = Washed out.
                vel = self.velocities[i]
                # Map 0.1->1.0 to Alpha 200->0
                alpha = int((1.0 - vel) * 180) 
                if alpha > 0:
                    painter.setBrush(QColor(255, 255, 255, alpha))
                    painter.setPen(Qt.PenStyle.NoPen)
                    painter.drawRect(rect)
            
            # Dividers
            if i > 0:
                painter.setPen(QPen(QColor(255, 255, 255, 120), 1))
                painter.drawLine(int(x), 0, int(x), int(h))

        painter.restore()

class AnimToggle(QPushButton):
    def __init__(self, text, anim_type='pulse', parent=None):
        super().__init__(text, parent)
        self.setCheckable(True)
        self.setFixedSize(30, 18)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.anim_type = anim_type
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update)
        self.timer.start(40)
        self.phase = 0.0
        
        # Font settings EXACTLY match SlotRow (10px)
        self.setStyleSheet("""
            QPushButton { 
                background: white; border: 1px solid #cbd5e0; border-radius: 3px;
                margin: 0px; padding: 0px; color: #4a5568; 
                font-size: 10px; font-weight: bold;
            }
            QPushButton:hover { background: #ebf8ff; color: #3182ce; border-color: #90cdf4; }
            QPushButton:checked { border-color: #a0aec0; } 
        """)

    def paintEvent(self, event):
        super().paintEvent(event)
        if not self.isChecked(): return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = self.rect()
        
        self.phase += 0.15
        if(self.phase > 100): self.phase = 0
        
        cx, cy = rect.center().x(), rect.center().y()

        if self.anim_type == 'pulse':
            # Purple Oscillation (Filter)
            # Alpha oscillates between 40 and 120
            alpha = int(80 + (np.sin(self.phase) * 40))
            color = QColor(159, 122, 234, alpha) 
            
            grad = QRadialGradient(cx, cy, rect.width() * 0.6)
            grad.setColorAt(0, color)
            grad.setColorAt(1, QColor(255, 255, 255, 0))
            
            painter.setBrush(grad)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawRoundedRect(rect, 3, 3)
            
        elif self.anim_type == 'ripple':
            # Blue Ripple (Spatial)
            # Sawtooth wave 0.0 -> 1.0
            prog = (self.phase % 4.0) / 4.0
            radius = prog * rect.width()
            alpha = int((1.0 - prog) * 180)
            
            color = QColor(66, 153, 225, alpha)
            painter.setPen(QPen(color, 2))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            # Draw ellipse centered
            painter.drawEllipse(QPointF(cx, cy), radius, radius * 0.6)

class AnimRevButton(QPushButton):
    def __init__(self, text, parent=None):
        super().__init__(text, parent)
        self.setCheckable(True)
        self.setFixedSize(30, 18)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update)
        self.timer.start(25)
        self.scan_pos = 1.0 # Start at Right (1.0)
        
        self.setStyleSheet("""
            QPushButton { 
                background: white; border: 1px solid #cbd5e0; border-radius: 3px;
                margin: 0px; padding: 0px; color: #4a5568; 
                font-size: 10px; font-weight: bold;
            }
            QPushButton:hover { background: #ebf8ff; color: #3182ce; border-color: #90cdf4; }
        """)

    def paintEvent(self, event):
        super().paintEvent(event)
        if not self.isChecked(): return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = self.rect()
        w, h = rect.width(), rect.height()

        # Move Left (Right -> Left)
        self.scan_pos -= 0.08
        if self.scan_pos < -0.3: self.scan_pos = 1.0
        
        # Calculate visual X position
        x_base = self.scan_pos * w

        # Draw Trail (Behind the movement, so to the Right)
        for i in range(4):
            # i=0 is main line, i=1..3 are trails
            lag_px = i * 2.5
            x = x_base + lag_px
            
            # Clip
            if x > w or x < 0: continue

            alpha = 255 if i == 0 else int(160 - (i * 50))
            if alpha < 0: alpha = 0
            
            pen = QPen(QColor(159, 122, 234, alpha)) # Purple
            pen.setWidthF(1.5 if i == 0 else 1.0)
            painter.setPen(pen)
            
            painter.drawLine(QPointF(x, 2), QPointF(x, h-2))

class ReseqRow(QFrame):
    pattern_changed = pyqtSignal()
    preview_req = pyqtSignal(object)
    saved_msg = pyqtSignal(str) 

    def __init__(self, label_text, bpm_ref, parent=None):
        super().__init__(parent)
        self.label_text = label_text
        self.bpm = bpm_ref 
        
        self.full_source_data = None 
        self.raw_sample = None       
        self.current_data = None     
        
        # FX States
        self.is_reversed = False
        self.mod_active = False
        self.mod_envelopes = [] 
        self.spatial_active = False
        self.spatial_params = {}

        self.layout = QHBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0) 
        self.layout.setSpacing(0)

        # Label Area
        lbl_frame = QWidget()
        lbl_frame.setFixedWidth(55)
        lf_layout = QHBoxLayout(lbl_frame)
        lf_layout.setContentsMargins(0,0,0,0)
        self.lbl = GradientLabel(label_text)
        self.lbl.clicked.connect(self.clear_sample)
        self.lbl.setToolTip("Click to clear sample")
        lf_layout.addWidget(self.lbl)
        self.layout.addWidget(lbl_frame)

        # Controls
        ctrl_frame = QWidget()
        ctrl_frame.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Preferred)
        ctrl_main_layout = QHBoxLayout(ctrl_frame)
        ctrl_main_layout.setContentsMargins(0, 0, 0, 0)
        ctrl_main_layout.setSpacing(0)

        # Button Container
        btn_container = QWidget()
        btn_container.setFixedWidth(65)
        btn_grid = QGridLayout(btn_container)
        btn_grid.setContentsMargins(0, 0, 4, 0)
        btn_grid.setSpacing(1)
        btn_grid.setVerticalSpacing(1)
        
        self.btn_snap = self.create_btn("snp", checkable=False)
        self.btn_snap.setToolTip("Random Snap to Transient")
        self.btn_snap.clicked.connect(self.snap_to_transient)
        
        self.btn_rev = AnimRevButton("rev", parent=self)
        self.btn_rev.setToolTip("Reverse Loop")
        self.btn_rev.toggled.connect(self.toggle_reverse)
        
        # Animated Toggles
        self.btn_mod = AnimToggle("flt", anim_type='pulse', parent=self)
        self.btn_mod.setToolTip("Filter Mod")
        self.btn_mod.toggled.connect(self.toggle_mod)
        
        self.btn_space = AnimToggle("del", anim_type='ripple', parent=self)
        self.btn_space.setToolTip("Reverb/Delay")
        self.btn_space.toggled.connect(self.toggle_spatial)

        btn_grid.addWidget(self.btn_snap, 0, 0)
        btn_grid.addWidget(self.btn_rev, 0, 1)
        btn_grid.addWidget(self.btn_mod, 1, 0)
        btn_grid.addWidget(self.btn_space, 1, 1)
        
        btn_grid.setAlignment(Qt.AlignmentFlag.AlignVCenter)
        ctrl_main_layout.addWidget(btn_container)

        # Sliders
        crush_container = QWidget()
        cc_layout = QHBoxLayout(crush_container)
        cc_layout.setContentsMargins(2, 0, 2, 0)
        cc_layout.setSpacing(2)
        
        def make_v_slider(val, tip, handler, hue_offset, def_val=50):
            sl = CircleSlider(Qt.Orientation.Vertical, base_hue=(260 + hue_offset) % 360, default_value=def_val)
            sl.setRange(0, 100)
            sl.setValue(val)
            sl.setFixedWidth(20)
            sl.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)
            sl.setToolTip(tip)
            sl.valueChanged.connect(handler)
            return sl

        self.sl_crush = make_v_slider(0, "Bitcrush", self.on_fx_change, 0, def_val=0)
        self.sl_filt = make_v_slider(50, "Filter", self.on_fx_change, 8, def_val=50)
        self.sl_pitch = make_v_slider(50, "Pitch", self.on_fx_change, 16, def_val=50)
        self.sl_decay = make_v_slider(50, "Gate/Decay", self.on_fx_change, 24, def_val=50)
        self.sl_tone = make_v_slider(50, "Tone", self.on_fx_change, 32, def_val=50)

        cc_layout.addWidget(self.sl_crush)
        cc_layout.addWidget(self.sl_filt)
        cc_layout.addWidget(self.sl_pitch)
        cc_layout.addWidget(self.sl_decay)
        cc_layout.addWidget(self.sl_tone)
        ctrl_main_layout.addWidget(crush_container)
        self.layout.addWidget(ctrl_frame)

        # Waveform
        self.wave_viz = ReseqWaveform(STEPS, self)
        self.wave_viz.file_dropped.connect(self.load_sample)
        self.wave_viz.pattern_changed.connect(lambda: self.pattern_changed.emit())
        self.layout.addWidget(self.wave_viz)

        self.btn_clr = ClearButton(self)
        self.btn_clr.clicked.connect(self.clear)
        self.layout.addSpacing(6)
        self.layout.addWidget(self.btn_clr)
        self.layout.addStretch(1)

        # Process audio after all UI elements are created
        self.process_audio()  # Force initial processing

    def create_btn(self, text, checkable=False):
        b = QPushButton(text)
        b.setFixedSize(30, 18)
        b.setCheckable(checkable)
        b.setCursor(Qt.CursorShape.PointingHandCursor)
        b.setStyleSheet("""
            QPushButton { 
                background: white; border: 1px solid #cbd5e0; border-radius: 3px;
                margin: 0px; padding: 0px; color: #4a5568; 
                font-size: 10px; font-weight: bold;
            }
            QPushButton:hover { background: #ebf8ff; color: #3182ce; border-color: #90cdf4; }
            QPushButton:checked { background: #e9d8fd; color: #6b46c1; border-color: #b794f4; }
        """)
        return b
    
    def clear(self):
        self.wave_viz.pattern = [False] * STEPS
        self.pattern_changed.emit()
        self.wave_viz.update()
    
    def highlight(self, idx):
        self.wave_viz.set_playing_step(idx)
    
    def update_sound(self, play=False):
        self.process_audio()
        if play and self.current_data is not None:
            self.preview_req.emit(self.current_data)
    
    @property
    def pattern(self): return self.wave_viz.pattern
    
    @property
    def velocities(self): return self.wave_viz.velocities

    def load_sample(self, path):
        try:
            data, f_sr = sf.read(path)
            if len(data.shape) > 1: data = np.mean(data, axis=1)
            
            if f_sr != SR:
                ratio = SR / f_sr
                pad_amt = 4096 
                padded = np.pad(data, pad_amt, mode='reflect')
                new_len = int(len(padded) * ratio)
                resampled = signal.resample(padded, new_len)
                crop_idx = int(pad_amt * ratio)
                data = resampled[crop_idx : -crop_idx]

            if len(data) > 300:
                fade_len = 150
                curve = 0.5 * (1.0 - np.cos(np.linspace(0, np.pi, fade_len)))
                data[:fade_len] *= curve
                data[-fade_len:] *= curve[::-1]

            peak = np.max(np.abs(data))
            if peak > 0: data /= peak
            
            self.full_source_data = data.astype(np.float32)
            self.full_source_data = np.nan_to_num(self.full_source_data) # Safety

            self.saved_msg.emit(f"reseq: {path.split('/')[-1]}")
            self.snap_to_transient()
            
        except Exception as e:
            print(e)
            self.saved_msg.emit("err loading sample")

    def apply_mod_filter(self, audio_data):
        if not self.mod_active or not self.mod_envelopes or len(audio_data) == 0:
            return audio_data

        step_len = len(audio_data) // STEPS
        if step_len < 10: return audio_data

        processed_chunks = []
        for i in range(STEPS):
            start = i * step_len
            end = (i + 1) * step_len if i < STEPS - 1 else len(audio_data)
            chunk = audio_data[start:end]
            
            if len(chunk) == 0: continue

            env = self.mod_envelopes[i]
            freq = (env['start'] + env['end']) * 0.5
            freq = np.clip(freq, 50, 16000)
            
            try:
                if env['type'] == 'lp':
                    sos = signal.butter(2, freq, 'lp', fs=SR, output='sos')
                elif env['type'] == 'hp':
                    sos = signal.butter(2, freq, 'hp', fs=SR, output='sos')
                else: 
                    width = freq * 0.5
                    sos = signal.butter(2, [max(20, freq - width), min(SR/2-1, freq + width)], 'bp', fs=SR, output='sos')
                chunk = signal.sosfilt(sos, chunk)
            except:
                pass 
            processed_chunks.append(chunk)

        return np.concatenate(processed_chunks)

    def snap_to_transient(self):
        if self.full_source_data is None: return

        default_style = """
            QPushButton { 
                background: white; border: 1px solid #cbd5e0; border-radius: 3px;
                margin: 0px; padding: 0px; color: #4a5568; 
                font-size: 10px; font-weight: bold;
            }
            QPushButton:hover { background: #ebf8ff; color: #3182ce; border-color: #90cdf4; }
        """
        
        try:
            sec_beat = 60.0 / self.bpm
            sec_step = sec_beat / 4.0
            target_length = int(sec_step * STEPS * SR)
            
            source_len = len(self.full_source_data)
            
            # 1. Transient Search
            best_start = 0
            min_usable = int(SR * 0.1) 
            limit = source_len - min_usable
            
            if limit > 0:
                max_energy = -1.0
                attempts = 8
                for _ in range(attempts):
                    cand = np.random.randint(0, limit)
                    window = self.full_source_data[cand : cand + 2048]
                    energy = np.sum(np.abs(window))
                    if energy > max_energy:
                        max_energy = energy
                        best_start = cand
                
                z_win = self.full_source_data[best_start : best_start + 1000]
                zcs = np.where(np.diff(np.sign(z_win)))[0]
                if len(zcs) > 0: best_start += zcs[0]
            
            # 2. Extract & Tile
            extracted = self.full_source_data[best_start:]
            if len(extracted) < 1024: extracted = self.full_source_data
            
            needed = target_length
            if len(extracted) >= needed:
                out_buffer = extracted[:needed].copy()
            else:
                repeats = (needed // len(extracted)) + 2
                tiled = np.tile(extracted, repeats)
                out_buffer = tiled[:needed].copy()

            # 3. Clean Edges
            out_buffer = np.nan_to_num(out_buffer, copy=False)
            
            fade_len = min(500, len(out_buffer) // 10)
            if fade_len > 0:
                out_buffer[:fade_len] *= np.linspace(0, 1, fade_len)
                out_buffer[-fade_len:] *= np.linspace(1, 0, fade_len)

            peak = np.max(np.abs(out_buffer))
            if peak > 0: out_buffer *= (0.95 / peak)

            self.raw_sample = out_buffer.astype(np.float32)
            self.current_data = None
            
            # Update Viz immediately
            self.wave_viz.set_data(self.raw_sample)

            self.btn_snap.setStyleSheet("""
                QPushButton { background: #E9D8FD; color: #553C9A; border: 1px solid #9F7AEA; border-radius: 3px; }
            """)
            QTimer.singleShot(150, lambda: self.btn_snap.setStyleSheet(default_style))
            
            # TRIGGER AUDIO PROCESSING
            # We do NOT emit pattern_changed here. process_audio will do it when it's done.
            QTimer.singleShot(10, self.process_audio)

        except Exception as e:
            print(f"Snap error: {e}")
            self.btn_snap.setStyleSheet(default_style)

    def process_audio(self):
        """Generates the continuous processed texture."""
        sec_beat = 60.0 / self.bpm
        sec_step = sec_beat / 4.0
        target_len = int(sec_step * STEPS * SR)

        if self.raw_sample is None:
            self.current_data = np.zeros(target_len, dtype=np.float32)
            self.wave_viz.set_data(None)
            return

        # --- FX CHAIN START ---
        
        working_audio = np.array(self.raw_sample, dtype=np.float32)
        working_audio = np.nan_to_num(working_audio)
        
        if self.is_reversed: working_audio = working_audio[::-1]
        
        if len(working_audio) < 100: 
            working_audio = np.zeros(target_len, dtype=np.float32)

        # 1. Pitch (High Quality / Anti-Aliased)
        p_pitch = self.sl_pitch.value() / 100.0
        speed = 0.5 + (p_pitch * 1.5)
        
        orig_len = len(working_audio)
        target_resample_len = int(orig_len / speed)
        
        # FIX: Use signal.resample (FFT) instead of np.interp (Linear)
        # This removes the "crunchy" aliasing artifacts.
        if abs(speed - 1.0) > 0.01 and target_resample_len > 10:
            working_audio = signal.resample(working_audio, target_resample_len).astype(np.float32)

        # 2. Tone Filters
        p_tone = self.sl_tone.value() / 100.0
        if p_tone < 0.45:
            cutoff = 400 + (p_tone * 8000)
            sos = signal.butter(2, cutoff, 'lp', fs=SR, output='sos')
            working_audio = signal.sosfilt(sos, working_audio)
        elif p_tone > 0.55:
            cutoff = 100 + ((p_tone - 0.5) * 4000)
            sos = signal.butter(2, cutoff, 'hp', fs=SR, output='sos')
            working_audio = signal.sosfilt(sos, working_audio)

        if self.mod_active: working_audio = self.apply_mod_filter(working_audio)

        filt_val = self.sl_filt.value() / 100.0
        working_audio = SynthEngine.apply_filter(working_audio, filt_val)
        
        # 3. Decay
        p_decay = self.sl_decay.value() / 100.0
        # Only apply gentle decay if not full sustain, to avoid choking the loop
        if p_decay < 0.95:
            t = np.linspace(0, 1, len(working_audio))
            coef = 0.1 + ((1.0 - p_decay) * 5.0) 
            working_audio *= np.exp(-t * coef)

        # 4. Crush (Strict Check)
        crush = self.sl_crush.value() / 100.0
        if crush > 0.01:
            working_audio = SynthEngine.resample_lofi(working_audio, crush)
        
        # 5. Tiling & De-clicking (Clean Loop Points)
        # FIX: Fade edges BEFORE tiling to prevent "clicks/pops" at loop points
        fade_edge = min(200, len(working_audio) // 10)
        if fade_edge > 0:
            working_audio[:fade_edge] *= np.linspace(0, 1, fade_edge)
            working_audio[-fade_edge:] *= np.linspace(1, 0, fade_edge)

        curr_len = len(working_audio)
        if curr_len < target_len and curr_len > 0:
            repeats = (target_len // curr_len) + 2
            working_audio = np.tile(working_audio, repeats)
            
        if len(working_audio) > target_len:
            working_audio = working_audio[:target_len]
        elif len(working_audio) < target_len:
            padded = np.zeros(target_len, dtype=np.float32)
            padded[:len(working_audio)] = working_audio
            working_audio = padded

        # --- FX CHAIN END ---

        # Final Polish: Gentle highpass to remove mud, gentle lowpass for silkiness
        sos_dc = signal.butter(1, 30, 'hp', fs=SR, output='sos')
        working_audio = signal.sosfilt(sos_dc, working_audio)
        
        sos_aa = signal.butter(1, 19000, 'lp', fs=SR, output='sos')
        working_audio = signal.sosfilt(sos_aa, working_audio)

        working_audio = self.apply_background_reverb(working_audio)
        if self.spatial_active:
            working_audio = self.apply_spatial_effects(working_audio)
        
        # FIX: Gentle Normalization instead of Hard Limiting
        peak = np.max(np.abs(working_audio))
        if peak > 0.95: 
            working_audio *= (0.90 / peak) # normalize to -1dB
        
        self.current_data = working_audio.astype(np.float32)
        self.pattern_changed.emit()

    def ensure_zero_crossing_fades(self, data):
        """Apply fades to ensure zero-crossing at both ends"""
        if len(data) < 100:
            return data
        
        fade_len = min(256, len(data) // 10)  # Adaptive fade length
        
        # Apply fade-in
        fade_in = np.linspace(0, 1, fade_len)
        data[:fade_len] *= fade_in
        
        # Apply fade-out  
        fade_out = np.linspace(1, 0, fade_len)
        data[-fade_len:] *= fade_out
        
        return data

    def toggle_reverse(self, state):
        self.is_reversed = state
        self.schedule_update()

    def toggle_mod(self, state):
        self.mod_active = state
        if state:
            self.mod_envelopes = []
            for i in range(STEPS):
                f_type = np.random.choice(['lp', 'bp', 'hp'], p=[0.5, 0.4, 0.1])
                if f_type == 'hp':
                    f_start = np.random.uniform(200, 3000)
                    f_end = np.random.uniform(200, 3000)
                else:
                    f_start = np.random.uniform(200, 7000)
                    f_end = np.random.uniform(200, 7000)

                self.mod_envelopes.append({
                    'type': f_type, 'start': f_start, 'end': f_end,
                    'q': 0.7 + (np.random.random() * 0.5)
                })
        self.schedule_update()

    def toggle_spatial(self, state):
        self.spatial_active = state
        if state:
            sync_opts = [0.25, 0.5, 0.75, 1.0]
            mult = sync_opts[np.random.randint(len(sync_opts))]
            self.spatial_params = {
                'delay_mult': mult,
                'feedback': np.random.uniform(0.4, 0.7),
                'reverb_mix': np.random.uniform(0.25, 0.5)
            }
        self.schedule_update()
  
    def clear_sample(self):
        self.full_source_data = None
        self.raw_sample = None
        self.current_data = None
        self.wave_viz.set_data(None)
        self.pattern_changed.emit()
        self.saved_msg.emit("reseq: sample cleared")
    
    def schedule_update(self):
        """Debounce timer start"""
        if hasattr(self, 'update_timer'):
            self.update_timer.start()
    
    def on_fx_change(self):
        self.current_data = None
        self.process_audio()
        self.pattern_changed.emit()
    
    def drift_params(self, amount):
        if amount <= 0.01: return
        targets = [self.sl_crush, self.sl_filt, self.sl_decay, self.sl_tone]
        for sl in targets:
            if not hasattr(sl, '_f_val'): sl._f_val = float(sl.value())
            if not hasattr(sl, '_vel'): sl._vel = np.random.uniform(-0.05, 0.05)
            force = (np.random.random() - 0.5) * 0.04 * amount
            if sl._f_val < 10: force += 0.03 * amount
            elif sl._f_val > 90: force -= 0.03 * amount
            sl._vel += force
            sl._vel *= 0.985
            new_val = sl._f_val + sl._vel
            sl._f_val = np.clip(new_val, 0.0, 100.0)
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
            fb_drift = np.random.uniform(-0.04, 0.04)
            self.spatial_params['feedback'] = np.clip(self.spatial_params['feedback'] + fb_drift, 0.2, 0.75)
            mix_drift = np.random.uniform(-0.03, 0.03)
            self.spatial_params['reverb_mix'] = np.clip(self.spatial_params['reverb_mix'] + mix_drift, 0.05, 0.45)
            changed = True
            
        if changed:
            self.on_fx_change()
    
    def update_bpm(self, new_bpm):
        self.bpm = new_bpm
        if self.full_source_data is not None:
            self.snap_to_transient()
        else:
            self.process_audio()
    
    def apply_background_reverb(self, audio_data):
        if len(audio_data) == 0: return audio_data
        
        wet = np.zeros_like(audio_data)
        delays = [int(SR * 0.015), int(SR * 0.03), int(SR * 0.06)]
        
        for d in delays:
            if d < len(audio_data):
                rolled = np.roll(audio_data, d)
                rolled[:d] = 0
                wet += rolled
        
        sos_hp = signal.butter(1, 600, 'hp', fs=SR, output='sos')
        wet = signal.sosfilt(sos_hp, wet)
        
        return (audio_data * 0.75) + (wet * 0.35)

    def apply_spatial_effects(self, audio_data):
        """Fixed: Circular Delay + BOOSTED Mix (x3.5)"""
        mix = self.spatial_params.get('reverb_mix', 0.4) 
        feedback = self.spatial_params.get('feedback', 0.5)
        delay_seconds = (60.0 / self.bpm) * self.spatial_params.get('delay_mult', 0.5)
        
        d_samples = int(delay_seconds * SR)
        wet_delay = np.zeros_like(audio_data)
        
        if d_samples > 0 and len(audio_data) > 0:
            d1 = np.roll(audio_data, d_samples)
            d2_samp = d_samples * 2
            d2 = np.roll(audio_data, d2_samp)
            
            # Simple Feedback Delay
            raw_delay = (d1 * feedback) + (d2 * (feedback * 0.8))
            
            # Filter the repeats
            sos_bp = signal.butter(1, [400, 3000], 'bp', fs=SR, output='sos')
            wet_delay = signal.sosfilt(sos_bp, raw_delay)

        # Multiplier increased from 2.5 to 3.5 for very prominent effect
        return audio_data + (wet_delay * mix * 3.5)

class SlotRow(QFrame):
    pattern_changed = pyqtSignal()
    preview_req = pyqtSignal(object)
    saved_msg = pyqtSignal(str) 

    def __init__(self, label_text, drum_type, base_hue=0, parent=None):
        super().__init__(parent)
        self.label_text = label_text
        self.drum_type = drum_type
        
        self.original_data = None    
        self.current_data = None     
        self.raw_sample = None       
        self.is_sample_mode = False  
        
        self.pattern = [False] * STEPS
        self.velocities = [0.8] * STEPS
        self.pads = []
        self.paint_state = True
        
        self.synth_params = {'pitch': 0.5, 'decay': 0.5, 'tone': 0.3}

        self.layout = QHBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0) 
        self.layout.setSpacing(0)

        # --- Label Area (Drag Drop + Reset) ---
        lbl_frame = QWidget()
        lbl_frame.setFixedWidth(55)
        lf_layout = QHBoxLayout(lbl_frame)
        lf_layout.setContentsMargins(0,0,0,0)
        lf_layout.setSpacing(0)
        self.lbl = DraggableLabel(label_text.lower())
        self.lbl.file_dropped.connect(self.load_sample)
        self.lbl.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        lf_layout.addWidget(self.lbl, 1) 
        self.btn_reset = ClearButton(self)
        self.btn_reset.setFixedSize(14, 14)
        self.btn_reset.hide()
        self.btn_reset.clicked.connect(self.reset_to_synth)
        lf_layout.addWidget(self.btn_reset, 0)
        self.layout.addWidget(lbl_frame)

        # --- Controls ---
        ctrl_frame = QWidget()
        ctrl_frame.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Preferred)
        
        # Main layout for the control block (Buttons | Sliders)
        ctrl_main_layout = QHBoxLayout(ctrl_frame)
        ctrl_main_layout.setContentsMargins(0, 0, 0, 0)
        ctrl_main_layout.setSpacing(0)

        # 1. Button Grid Container (Fixed Width: 65px)
        btn_container = QWidget()
        btn_container.setFixedWidth(65)
        
        # Grid for the 4 buttons
        btn_grid = QGridLayout(btn_container)
        btn_grid.setContentsMargins(0, 0, 4, 0) # 4px right padding
        btn_grid.setSpacing(1)                   # 1px gap between buttons
        btn_grid.setVerticalSpacing(1)           # Tight vertical spacing
        
        # Create Buttons
        self.btn_wav = self.create_btn("wav")
        self.btn_wav.clicked.connect(self.export_one)
        
        self.btn_vel = self.create_btn("vel")
        self.btn_vel.clicked.connect(self.randomize_velocity)
        
        self.btn_rnd = self.create_btn("rnd")
        self.btn_rnd.clicked.connect(self.syncopate_gentle)
        
        self.btn_low = self.create_btn("low")
        self.btn_low.clicked.connect(self.lower_velocity)
        
        # Add to Grid (Compact 2x2)
        btn_grid.addWidget(self.btn_wav, 0, 0)
        btn_grid.addWidget(self.btn_vel, 0, 1)
        btn_grid.addWidget(self.btn_rnd, 1, 0)
        btn_grid.addWidget(self.btn_low, 1, 1)
        
        # Center the grid vertically
        btn_grid.setAlignment(Qt.AlignmentFlag.AlignVCenter)

        ctrl_main_layout.addWidget(btn_container)

        # 2. Slider Container
        crush_container = QWidget()
        cc_layout = QHBoxLayout(crush_container)
        cc_layout.setContentsMargins(2, 0, 2, 0)
        cc_layout.setSpacing(2)
        
        # Define make_v_slider as a proper method with self parameter
        def make_v_slider(val, tip, handler, hue_offset, def_val=50):
            sl = CircleSlider(Qt.Orientation.Vertical, base_hue=(base_hue + hue_offset) % 360, default_value=def_val)
            sl.setRange(0, 100)
            sl.setValue(val)
            sl.setFixedWidth(20) 
            sl.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)
            sl.setToolTip(tip)
            
            # Connect to the scheduler method
            sl.valueChanged.connect(self.schedule_update) 
            return sl

        self.sl_crush = make_v_slider(0, "Bitcrush", None, 0, def_val=0)
        self.sl_filt = make_v_slider(50, "Filter", None, 8, def_val=50)
        self.sl_pitch = make_v_slider(50, "Pitch", None, 16, def_val=50)
        self.sl_decay = make_v_slider(50, "Decay", None, 24, def_val=50)
        self.sl_tone = make_v_slider(30, "Tone", None, 32, def_val=30)
        
        cc_layout.addWidget(self.sl_crush)
        cc_layout.addWidget(self.sl_filt)
        cc_layout.addWidget(self.sl_pitch)
        cc_layout.addWidget(self.sl_decay)
        cc_layout.addWidget(self.sl_tone)
        
        ctrl_main_layout.addWidget(crush_container)
        self.layout.addWidget(ctrl_frame)

        # --- Pads ---
        for i in range(STEPS):
            p = StepPad(i, base_hue, self)
            p.toggled.connect(lambda a, v, idx=i: self.update_step(idx, a, v))
            p.velocity_changed.connect(lambda v, idx=i: self.update_vel(idx, v))
            self.pads.append(p)
            self.layout.addWidget(p)

        self.btn_clr = ClearButton(self)
        self.btn_clr.clicked.connect(self.clear)
        self.layout.addSpacing(6)
        self.layout.addWidget(self.btn_clr)
        self.layout.addStretch(1)
        
        # Timer setup
        self.update_timer = QTimer(self)
        self.update_timer.setSingleShot(True)
        self.update_timer.setInterval(100)
        self.update_timer.timeout.connect(lambda: self.update_sound(play=False))
        
        # MOVE THIS TO THE END: Initialize audio after all UI elements are created
        self.update_sound()

    def schedule_update(self):
        """Debounce timer start"""
        if hasattr(self, 'update_timer'):
            self.update_timer.start()

    def lower_velocity(self):
        changed = False
        for i in range(STEPS):
            if self.pattern[i]:
                # Lower by 0.15, clamp at 0.1
                new_vel = max(0.1, self.velocities[i] - 0.15)
                if abs(new_vel - self.velocities[i]) > 0.001:
                    self.velocities[i] = new_vel
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

    def start_painting(self, state):
        self.paint_state = state

    def create_btn(self, text):
        b = QPushButton(text)
        b.setFixedSize(30, 18)
        b.setCursor(Qt.CursorShape.PointingHandCursor)
        b.setStyleSheet("""
            QPushButton { 
                background: white; border: 1px solid #cbd5e0; margin: 0px; 
                color: #4a5568; font-size: 10px; font-weight: bold; border-radius: 3px;
                padding: 0px;
            }
            QPushButton:hover { background: #ebf8ff; color: #3182ce; border-color: #90cdf4; }
            QPushButton:checked { background: #3f6c9b; color: white; border-color: #3f6c9b; }
        """)
        return b

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
        
        # CRITICAL FIX: Drum Click
        # Ensure the generated drum sound fades out at the very end of its buffer
        # This prevents DC offsets when the audio mixer sums it
        if len(self.current_data) > 100:
            fade = 50
            self.current_data[-fade:] *= np.linspace(1, 0, fade)

        if self.current_data is None:
            self.current_data = SynthEngine.generate_fallback_silence()

    def update_step(self, idx, act, vel):
        self.pattern[idx] = act
        self.velocities[idx] = vel
        self.pattern_changed.emit()

    def update_vel(self, idx, vel):
        self.velocities[idx] = vel
        if not self.pattern[idx]:
            self.pattern[idx] = True
            self.pads[idx].active = True
            self.pads[idx].update()
        self.pattern_changed.emit()

    def randomize_velocity(self):
        changed = False
        for i in range(STEPS):
            if self.pattern[i]:
                new_vel = np.random.uniform(0.3, 1.0)
                self.velocities[i] = new_vel
                self.pads[i].velocity = new_vel
                self.pads[i].update()
                changed = True
        if changed: self.pattern_changed.emit()

    def syncopate_gentle(self):
        density = 0.35
        if "kick" in self.label_text: density = 0.25
        elif "hat" in self.label_text: density = 0.5
        elif "snare" in self.label_text: density = 0.2
        
        for i in range(STEPS):
            is_active = np.random.random() < density
            self.pattern[i] = is_active
            self.pads[i].active = is_active
            if is_active:
                new_vel = np.random.uniform(0.4, 1.0)
                self.velocities[i] = new_vel
                self.pads[i].velocity = new_vel
            self.pads[i].update()
        self.pattern_changed.emit()

    def clear(self):
        for i in range(STEPS):
            self.pattern[i] = False
            self.pads[i].active = False
            self.pads[i].update()
        self.pattern_changed.emit()

    def highlight(self, idx):
        for i, p in enumerate(self.pads): p.set_playing(i == idx)

    def export_one(self):
        if self.current_data is None: return
        
        home_dir = os.path.expanduser("~")
        save_dir = os.path.join(home_dir, "Music", "sequa")
        
        if not os.path.exists(save_dir):
            try: os.makedirs(save_dir)
            except:
                self.saved_msg.emit("err: cannot create folder")
                return

        timestamp = int(time.time())
        safe_name = self.label_text.replace(" ", "_")
        filename = f"{safe_name}_{timestamp}.wav"
        full_path = os.path.join(save_dir, filename)
        
        try:
            sf.write(full_path, self.current_data, SR)
            self.saved_msg.emit(f"saved: Music/sequa/{filename}")
        except Exception as e:
            self.saved_msg.emit(f"err: {str(e)}")

    def drift_params(self, amount):
        if amount <= 0.01: return
        targets = [self.sl_crush, self.sl_filt, self.sl_decay, self.sl_tone]
        for sl in targets:
            if not hasattr(sl, '_f_val'): sl._f_val = float(sl.value())
            if not hasattr(sl, '_vel'): sl._vel = np.random.uniform(-0.05, 0.05)
            force = (np.random.random() - 0.5) * 0.04 * amount
            if sl._f_val < 10: force += 0.03 * amount
            elif sl._f_val > 90: force -= 0.03 * amount
            sl._vel += force
            sl._vel *= 0.985
            new_val = sl._f_val + sl._vel
            sl._f_val = np.clip(new_val, 0.0, 100.0)
            if int(sl._f_val) != sl.value(): sl.setValue(int(sl._f_val))

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

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.RightButton:
            self.setValue(self.default_value)
            event.accept()
        else:
            super().mousePressEvent(event)

    def tick_color(self):
        # Slower, discrete-ish oscillation around the base hue (+/- 15 degrees)
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
        # 1. Faster Fade Logic
        if abs(self.opacity - self.target_opacity) > 0.001:
            # Increased speed factor from 0.05 to 0.15
            self.opacity += (self.target_opacity - self.opacity) * 0.15
        else:
            self.opacity = self.target_opacity

        # Stop timer if fully invisible
        if self.opacity < 0.01 and self.target_opacity == 0.0:
            self.opacity = 0.0
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
        painter.setOpacity(self.opacity) 
        
        # Draw Text (Aligned Left) - No background
        painter.setPen(QColor("#5a67d8")) # Soft Indigo
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
        self.setFixedSize(760, 560)
        
        self.slots = []
        self.bpm = BPM_DEFAULT
        self.swing = 0.0
        self.clip_amount = 0.0
        self.evolve_amount = 0.0
        self.reverse_prob = 0.0 # Reverse probability
        self.step = -1
        self.last_processed_step = -1
        self.last_preview_time = 0
        
        # Color animation ticker
        self.anim_tick_counter = 0

        self.fmt = QAudioFormat()
        self.fmt.setSampleRate(SR)
        self.fmt.setChannelCount(1)
        self.fmt.setSampleFormat(QAudioFormat.SampleFormat.Int16)
        
        self.sink = QAudioSink(QMediaDevices.defaultAudioOutput(), self.fmt)
        self.sink.setBufferSize(int(SR * 2 * (BUFFER_MS / 1000.0)) * 2)
        
        # Set larger buffer size for stability
        preferred_size = self.sink.bufferSize()
        if preferred_size < 4096:
            self.sink.setBufferSize(4096)
        
        self.gen = LoopGenerator(self.fmt, self)
        self.preview = SoundPreview(self)
        
        self.timer = QTimer()
        self.timer.setTimerType(Qt.TimerType.PreciseTimer)
        self.timer.setInterval(5) 
        self.timer.timeout.connect(self.update_playhead)
        self.timer.start() # <--- Start immediately, never stop
        
        self.setup_ui()
        self.update_mix()
        self.sink.start(self.gen)
    
    def change_kit(self, kit_idx):
        # 1. Update Engine
        kit_map = [SynthEngine.KIT_SIMPLE, SynthEngine.KIT_PCM, SynthEngine.KIT_EIGHT, SynthEngine.KIT_NINE]
        SynthEngine.set_kit(kit_map[kit_idx])
        
        # 2. Regenerate all sounds (CPU calculation is safe to do rapidly)
        for s in self.slots:
            s.update_sound(play=False)
            
        # 3. Handle Audio Preview with Debounce
        # Only touch the QAudioSink if enough time has passed (150ms)
        now = time.time()
        if self.slots and (now - self.last_preview_time > 0.15):
            self.last_preview_time = now
            
            # Filter out ReseqRow to play only drums
            drums = [s for s in self.slots if not isinstance(s, ReseqRow)]
            if drums:
                r_slot = drums[np.random.randint(len(drums))]
                self.preview.play(r_slot.current_data)
            
        # 4. Show Notification
        names = ["simple", "pcm", "eight", "nine"]
        self.show_notification(f"kit loaded: {names[kit_idx]}")
    
    def setup_ui(self):
        cw = QWidget()
        self.setCentralWidget(cw)
        cw.setStyleSheet("QWidget { font-family: 'Segoe UI', sans-serif; background-color: #f7fafc; }")
        main = QVBoxLayout(cw)
        main.setContentsMargins(15, 2, 15, 15) 
        main.setSpacing(2)

        # --- Top Control Header ---
        header = QHBoxLayout()
        header.setSpacing(10)
        header.setContentsMargins(0, 0, 0, 5)
        header.setAlignment(Qt.AlignmentFlag.AlignVCenter)
        
        # Logo - Added directly to header to prevent margin clipping
        self.logo = LogoWidget()
        self.logo.kit_changed.connect(self.change_kit)
        # header.addWidget(self.logo) # We wrap it slightly to ensure left spacing if needed, but direct is safest:
        logo_cont = QWidget()
        logo_layout = QVBoxLayout(logo_cont)
        logo_layout.setContentsMargins(0, 0, 0, 0) # removed the 8px top margin
        logo_layout.addWidget(self.logo)
        header.addWidget(logo_cont)
        
        def setup_lbl(text):
            l = QLabel(text)
            # Increased from 40 to 48 to fit "swing" comfortably
            l.setFixedWidth(48) 
            l.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            l.setStyleSheet("color: #4a5568; font-weight: bold; font-size: 12px; margin-right: 6px;")
            return l

        def setup_slider(val, callback, hue, def_val=0):
            sl = CircleSlider(Qt.Orientation.Horizontal, base_hue=hue, default_value=def_val)
            sl.setRange(0, 100)
            sl.setValue(val)
            sl.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            sl.valueChanged.connect(callback)
            return sl
        
        # --- Controls ---
        
        self.lbl_bpm = QLabel(f"bpm\n{self.bpm}")
        self.lbl_bpm.setFixedWidth(35)
        self.lbl_bpm.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.lbl_bpm.setStyleSheet("color: #676f7c; font-weight: bold; font-size: 11px; margin-right: 4px;")
        
        self.sl_bpm = setup_slider(self.bpm, self.set_bpm, 180, def_val=BPM_DEFAULT)
        self.sl_bpm.setRange(60, 200)
        self.sl_bpm.setToolTip(f"{self.bpm} BPM") 
        
        header.addWidget(self.lbl_bpm)
        header.addWidget(self.sl_bpm)

        self.lbl_swg = setup_lbl("swing")
        self.sl_swg = setup_slider(0, self.set_swing, 210, def_val=0)
        self.sl_swg.sliderReleased.connect(self.update_mix) 
        header.addWidget(self.lbl_swg)
        header.addWidget(self.sl_swg)

        self.lbl_clip = setup_lbl("cut")
        self.sl_clip = setup_slider(0, self.set_clip, 240)
        header.addWidget(self.lbl_clip)
        header.addWidget(self.sl_clip)

        self.lbl_evolve = setup_lbl("evo")
        self.sl_evolve = setup_slider(0, self.set_evolve, 270)
        header.addWidget(self.lbl_evolve)
        header.addWidget(self.sl_evolve)

        self.lbl_rev = setup_lbl("rev")
        self.sl_rev = setup_slider(0, self.set_rev, 300)
        header.addWidget(self.lbl_rev)
        header.addWidget(self.sl_rev)

        main.addLayout(header)

        # --- Sequencer Area ---
        
        # Column Labels
        head_row = QWidget()
        head_layout = QHBoxLayout(head_row)
        head_layout.setContentsMargins(0, 0, 0, 2)
        head_layout.setSpacing(0)
        
        # 1. Match SlotRow Label Column
        lbl_space = QWidget()
        lbl_space.setFixedWidth(55) 
        head_layout.addWidget(lbl_space)

        # 2. Match SlotRow Control Column
        ctrl_head = QWidget()
        # Fixed Width = 65 (Buttons) + 112 (Sliders: 2 margin + 5*20 width + 4*2 spacing + 2 margin)
        ctrl_head.setFixedWidth(177) 
        
        ch_layout = QHBoxLayout(ctrl_head)
        ch_layout.setContentsMargins(0, 0, 0, 0)
        ch_layout.setSpacing(0)
        
        # 2a. Button Space
        btn_space = QWidget()
        btn_space.setFixedWidth(65) 
        ch_layout.addWidget(btn_space)
        
        # 2b. Label Container (Matches Slider Container)
        lbl_cont = QWidget()
        lc_layout = QHBoxLayout(lbl_cont)
        lc_layout.setContentsMargins(2, 0, 2, 0) 
        lc_layout.setSpacing(2)
        
        # Use Clickable HeaderParamLabels
        # Key maps to the attribute name of the slider in the rows
        headers = [("bit", "sl_crush"), ("flt", "sl_filt"), ("pch", "sl_pitch"), 
                   ("dec", "sl_decay"), ("ton", "sl_tone")]
                   
        for txt, key in headers:
            l = HeaderParamLabel(txt, key)
            l.clicked.connect(self.randomize_column)
            lc_layout.addWidget(l)

        ch_layout.addWidget(lbl_cont)
        head_layout.addWidget(ctrl_head)
        head_layout.addStretch()
        main.addWidget(head_row)
        
        # Instrument Rows
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
        
        # Generate initial mix
        self.update_mix()

        # Footer
        bot = QHBoxLayout()
        bot.setContentsMargins(0, 10, 0, 0)
        
        self.btn_play = PlayButton()
        self.btn_play.clicked.connect(self.toggle_play)
        
        btn_exp = QPushButton("export loop")
        btn_exp.setFixedSize(100, 26)
        btn_exp.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_exp.clicked.connect(self.export_beat)
        btn_exp.setStyleSheet("""
            QPushButton { 
                background: white; border: 1px solid #a0aec0; 
                border-radius: 13px; color: #2d3748; font-weight: bold; font-size: 12px;
            }
            QPushButton:hover { border-color: #3f6c9b; color: #3f6c9b; }
        """)
        
        btn_clr = QPushButton("clear all")
        btn_clr.setFixedSize(80, 26)
        btn_clr.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_clr.clicked.connect(self.clear_all)
        btn_clr.setStyleSheet("""
            QPushButton { 
                background: white; border: 1px solid #feb2b2; 
                border-radius: 13px; color: #c53030; font-weight: bold; font-size: 12px;
            }
            QPushButton:hover { border-color: #c53030; background: #fff5f5; }
        """)

        self.status_widget = StatusWidget()
        
        bot.addWidget(self.btn_play)
        bot.addSpacing(5)
        bot.addWidget(btn_exp)
        
        # Reduced spacing here so the status text feels connected to the export action
        bot.addSpacing(8) 
        
        # Status widget fills the gap between Export and Clear
        bot.addWidget(self.status_widget, 1)
        
        bot.addWidget(btn_clr)
        main.addLayout(bot)
    
    def set_bpm(self, v):
        if self.bpm == v: return
        self.sl_bpm.setToolTip(f"{v} BPM")
        self.lbl_bpm.setText(f"bpm\n{v}")
        
        # Update Reseq row so it restretches audio
        if hasattr(self, 'reseq'):
            self.reseq.update_bpm(v)

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

    def update_mix(self):
        slot_data = []
        for s in self.slots:
            # FIX: Pass the is_sliced flag for live playback too
            is_sliced_track = isinstance(s, ReseqRow)
            
            slot_data.append({
                'data': s.current_data, 
                'pattern': s.pattern, 
                'velocities': s.velocities,
                'is_sliced': is_sliced_track # <--- CRITICAL FLAG
            })
        
        self.gen.set_data(AudioMixer.mix_sequence(slot_data, self.bpm, self.swing, 
                                                  self.clip_amount, self.reverse_prob))

    def play_preview(self, data):
        self.preview.play(data)

    def toggle_play(self):
        if self.gen.playing:
            self.gen.set_playback_state(False)
            # REMOVED: self.timer.stop() 
            # The timer must keep running for UI animations (logo, buttons) to work while paused.
            self.btn_play.set_playing(False)
            self.reset_vis()
        else:
            self.gen.pos = 0 
            self.gen.set_playback_state(True)
            self.start_time = time.perf_counter()
            self.btn_play.set_playing(True)
            self.step = -1
            self.last_processed_step = -1

    def update_playhead(self):
        self.logo.animate()
        self.btn_play.animate()
        
        self.anim_tick_counter += 1
        if self.anim_tick_counter > 15: 
            self.anim_tick_counter = 0
            
        if not self.gen.playing:
            return
            
            top_sliders = [self.sl_bpm, self.sl_swg, self.sl_clip, self.sl_evolve, self.sl_rev]
            for sl in top_sliders:
                sl.tick_color()

            for s in self.slots:
                # Ensure we only tick sliders that actually exist on the object
                if hasattr(s, 'sl_crush'): s.sl_crush.tick_color()
                if hasattr(s, 'sl_filt'): s.sl_filt.tick_color()
                if hasattr(s, 'sl_pitch'): s.sl_pitch.tick_color()
                if hasattr(s, 'sl_decay'): s.sl_decay.tick_color()
                if hasattr(s, 'sl_tone'): s.sl_tone.tick_color()

        # Update pads visually every frame
        for s in self.slots:
            if hasattr(s, 'pads'):
                for p in s.pads:
                    if p.active: p.update()

        if not self.gen.playing: return

        # --- EVOLVE DRIFT (SLIDERS) ---
        if self.evolve_amount > 0.01:
            for s in self.slots: 
                # Check for method existence to prevent crash on ReseqRow vs SlotRow differences
                if hasattr(s, 'drift_params'):
                    s.drift_params(self.evolve_amount)

        # --- PLAYHEAD LOGIC ---
        loop_duration = (60.0 / self.bpm) * 4.0
        elapsed = time.perf_counter() - self.start_time
        current_step = int((elapsed % loop_duration) / loop_duration * STEPS)
        
        if current_step != self.step:
            self.step = current_step if current_step < STEPS else 0
            
            # --- RESEQ EVOLUTION TRIGGER ---
            # Triggered exactly when the loop wraps back to step 0
            if self.step == 0:
                for s in self.slots:
                    if hasattr(s, 'evolve_fx_params'):
                        s.evolve_fx_params()

            # Trigger the dark playhead on the pads
            for s in self.slots: s.highlight(self.step)

            if self.step != self.last_processed_step:
                if self.evolve_amount > 0.05:
                    if np.random.random() < (self.evolve_amount * 0.3): 
                        # Filter for slots that have pads (Standard Slots)
                        valid_slots = [s for s in self.slots if hasattr(s, 'pads')]
                        if valid_slots:
                            s_idx = np.random.randint(len(valid_slots))
                            slot = valid_slots[s_idx]
                            p_idx = np.random.randint(STEPS)
                            new_state = not slot.pattern[p_idx]
                            vel = slot.velocities[p_idx]
                            slot.pads[p_idx].active = new_state
                            slot.pads[p_idx].update()
                            slot.update_step(p_idx, new_state, vel)
            
            self.last_processed_step = self.step
    
    def randomize_column(self, param_key):
        """Randomizes the specific parameter for ALL slots"""
        self.logo.trigger_flash() # Visual feedback
        
        for s in self.slots:
            if hasattr(s, param_key):
                slider = getattr(s, param_key)
                # Generate unique random value per slot
                new_val = np.random.randint(0, 101)
                slider.setValue(new_val)
                
        self.show_notification(f"randomized: {param_key.replace('sl_', '')}")

    def reset_vis(self):
        self.step = -1
        for s in self.slots: s.highlight(-1)

    def show_notification(self, text):
        self.status_widget.set_text(text)

    def export_beat(self):
        self.logo.trigger_flash()
        
        # --- Mix Logic ---
        slot_data = []
        for s in self.slots:
            # FIX: Explicitly check if this slot is the Reseq/Slicer
            # This tells AudioMixer to chop it up instead of playing it like a drum
            is_sliced_track = isinstance(s, ReseqRow)
            
            slot_data.append({
                'data': s.current_data, 
                'pattern': s.pattern, 
                'velocities': s.velocities,
                'is_sliced': is_sliced_track  # <--- CRITICAL FLAG
            })
            
        mix = AudioMixer.mix_sequence(slot_data, self.bpm, self.swing, 
                                      self.clip_amount, self.reverse_prob)
        # Tile for 2 loops so the tail wraps nicely for listeners
        final = np.tile(mix, 2)
        
        # --- Directory Setup ---
        home_dir = os.path.expanduser("~")
        save_dir = os.path.join(home_dir, "Music", "sequa")
        
        if not os.path.exists(save_dir):
            try: os.makedirs(save_dir)
            except:
                self.show_notification("err: cannot create folder")
                return

        # --- Save File ---
        timestamp = int(time.time())
        filename = f"sequa_loop_{self.bpm}bpm_{timestamp}.wav"
        full_path = os.path.join(save_dir, filename)

        try:
            sf.write(full_path, final, SR)
            self.show_notification(f"saved to Music/sequa")
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