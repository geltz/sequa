import sys
import time
import numpy as np
import soundfile as sf
from scipy import signal

from PyQt6.QtCore import (Qt, pyqtSignal, QTimer, QRectF, QIODevice, 
                          QByteArray, QPropertyAnimation, QEasingCurve, 
                          QPointF, QUrl)
from PyQt6.QtGui import (QColor, QPainter, QPen, QFont, QPainterPath, 
                         QLinearGradient, QBrush)
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

class LoopGenerator(QIODevice):
    def __init__(self, fmt, parent=None):
        super().__init__(parent)
        self.data = bytes()
        self.pos = 0
        self.fmt = fmt
        self.playing = False
        self.open(QIODevice.OpenModeFlag.ReadOnly)

    def set_playback_state(self, is_playing):
        self.playing = is_playing
        if is_playing: self.pos = 0

    def set_data(self, float_data):
        audio = np.clip(float_data, -1.0, 1.0)
        audio = (audio * 32767).astype(np.int16)
        new_bytes = audio.tobytes()
        
        current_len = len(self.data)
        new_len = len(new_bytes)
        
        if current_len > 0 and new_len > 0 and self.pos > 0:
            progress = self.pos / current_len
            self.pos = int(progress * new_len)
            if self.pos % 2 != 0: self.pos -= 1
        
        self.data = new_bytes
        if self.pos >= len(self.data): self.pos = 0

    def readData(self, maxlen):
        if not self.playing or not self.data:
            return b'\x00' * maxlen

        chunk = b''
        data_len = len(self.data)
        while len(chunk) < maxlen:
            if self.pos >= data_len: self.pos = 0
            remaining = data_len - self.pos
            to_read = min(maxlen - len(chunk), remaining)
            chunk += self.data[self.pos : self.pos + to_read]
            self.pos += to_read
        return chunk

    def writeData(self, data): return 0
    def bytesAvailable(self): return len(self.data) + super().bytesAvailable() + 4096
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
        audio = np.clip(float_data, -1.0, 1.0)
        audio = (audio * 32767).astype(np.int16)
        self.data_bytes = QByteArray(audio.tobytes())
        self.pos = 0
        self.sink.start(self)

    def readData(self, maxlen):
        if self.pos >= self.data_bytes.size(): return b''
        chunk = self.data_bytes.mid(self.pos, maxlen)
        self.pos += chunk.size()
        return chunk.data()

    def writeData(self, data): return 0
    def bytesAvailable(self): return self.data_bytes.size() - self.pos
    def isSequential(self): return True

# --- Synth Engine ---

class SynthEngine:
    @staticmethod
    def ensure_zero_crossing(data):
        if len(data) < 200: return SynthEngine.declick(data)
        limit = min(len(data) // 4, 2000)
        
        zero_crossings_start = np.where(np.diff(np.sign(data[:limit])))[0]
        start_idx = zero_crossings_start[0] + 1 if len(zero_crossings_start) > 0 else 0

        zero_crossings_end = np.where(np.diff(np.sign(data[-limit:])))[0]
        end_idx = (len(data) - limit) + zero_crossings_end[-1] if len(zero_crossings_end) > 0 else len(data)

        new_data = data[start_idx:end_idx]
        if len(new_data) < 10: return SynthEngine.declick(data)
        
        fade_len = 10
        if len(new_data) > fade_len * 2:
            new_data[:fade_len] *= np.linspace(0, 1, fade_len)
            new_data[-fade_len:] *= np.linspace(1, 0, fade_len)
        return new_data

    @staticmethod
    def declick(data):
        if len(data) < 100: return data
        fade_len = min(int(SR * 0.003), len(data) // 2)
        fade_in = np.sin(np.linspace(0, np.pi/2, fade_len))
        fade_out = np.cos(np.linspace(0, np.pi/2, fade_len))
        data[:fade_len] *= fade_in
        data[-fade_len:] *= fade_out
        return data

    @staticmethod
    def process_sample(raw_data, params):
        """Applies Pitch, Decay, and Tone to an external sample."""
        if raw_data is None or len(raw_data) == 0:
            return np.zeros(100, dtype=np.float32)

        p_pitch = params.get('pitch', 0.5)
        p_decay = params.get('decay', 0.5)
        p_tone = params.get('tone', 0.5)

        # 1. Pitch (Resampling)
        # Range: 0.5x speed (low) to 2.0x speed (high)
        speed = 0.5 + (p_pitch * 1.5)
        new_len = int(len(raw_data) / speed)
        if new_len < 10: new_len = 10
        y = signal.resample(raw_data, new_len)

        t = np.linspace(0, len(y) / SR, len(y))

        # 2. Tone (Simple Tilt Filter)
        # < 0.5 Lowpass, > 0.5 Highpass
        if p_tone < 0.45:
            cutoff = 500 + (p_tone * 8000) # 500Hz to ~4kHz
            sos = signal.butter(1, cutoff, 'lp', fs=SR, output='sos')
            y = signal.sosfilt(sos, y)
        elif p_tone > 0.55:
            cutoff = 100 + ((p_tone - 0.5) * 4000) 
            sos = signal.butter(1, cutoff, 'hp', fs=SR, output='sos')
            y = signal.sosfilt(sos, y)

        # 3. Decay (Exponential Envelope)
        # Map decay slider 0.0-1.0 to a decay coefficient
        decay_coef = 0.5 + ((1.0 - p_decay) * 15) 
        env = np.exp(-t * decay_coef)
        y = y * env

        return SynthEngine.ensure_zero_crossing(y.astype(np.float32))

    @staticmethod
    def generate_drum(drum_type, params):
        p_pitch = params.get('pitch', 0.5)
        p_decay = params.get('decay', 0.5)
        p_tone = params.get('tone', 0.5)
        
        duration = 0.8
        t = np.linspace(0, duration, int(SR * duration), endpoint=False)
        y = np.zeros_like(t)
        
        rng = np.random.default_rng(int(p_pitch * 1000 + p_tone * 100))

        if drum_type == "kick":
            f_start = 50 + (p_pitch * 250)
            f_end = 30 + (p_pitch * 30)
            f_decay = 20 + (p_decay * 60)
            freq_env = (f_start - f_end) * np.exp(-t * f_decay) + f_end
            phase = np.cumsum(freq_env) * 2 * np.pi / SR
            drive = 1.0 + (p_tone * 3.0)
            osc = np.tanh(np.sin(phase) * drive)
            amp_decay = 5 + ((1.0 - p_decay) * 55)
            amp_env = np.exp(-t * amp_decay)
            y = osc * amp_env

        elif drum_type == "snare":
            f_body = 140 + (p_pitch * 150)
            tone_osc = np.sin(2 * np.pi * f_body * t) * np.exp(-t * (30 + (1.0-p_decay)*60))
            filt_center = 1000 + (p_tone * 5000)
            noise = rng.uniform(-1, 1, len(t))
            noise = signal.sosfilt(signal.butter(2, [filt_center, filt_center+2000], 'bp', fs=SR, output='sos'), noise)
            noise_env = np.exp(-t * (30 + ((1.0-p_decay) * 80)))
            y = (tone_osc * 0.5) + (noise * noise_env * 0.8)

        elif drum_type == "closed hat" or drum_type == "open hat":
            # 808-style
            # Uses ratios closer to the original 6 schmitt trigger oscillators
            base_f = 200 + (p_pitch * 150)
            ratios = [2.0, 3.0, 4.16, 5.43, 6.79, 8.21]
            sum_sig = np.zeros_like(t)
            
            for r in ratios:
                phase = 2 * np.pi * base_f * r * t
                sum_sig += np.sign(np.sin(phase)) # Square waves
            
            # Bandpass: 808 hats are distinctively bandpassed
            # Modified: Higher center and wider band for "Air"
            bp_center = 4000 + (p_tone * 5000) 
            sos_bp = signal.butter(2, [bp_center, bp_center + 4000], 'bp', fs=SR, output='sos')
            processed = signal.sosfilt(sos_bp, sum_sig)
            
            # Highpass: Clean up mud - Raised frequency for softer texture
            hp_freq = 7000 + (p_tone * 3000)
            sos_hp = signal.butter(2, hp_freq, 'hp', fs=SR, output='sos')
            processed = signal.sosfilt(sos_hp, processed)

            if drum_type == "closed hat":
                decay_coef = 60 + ((1.0 - p_decay) * 200)
            else:
                decay_coef = 8 + ((1.0 - p_decay) * 90)
            
            # Soften attack slightly to remove digital click
            processed[:50] *= np.linspace(0, 1, 50)
            
            env = np.exp(-t * decay_coef)
            y = processed * env

        elif drum_type == "clap":
            noise = rng.uniform(-1, 1, len(t))

            low = 900 + (p_pitch * 200)
            high = 2400 + (p_pitch * 600)
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
                
                # Extremely sharp decay (350) for the flam hits
                decay = 350 if i < 3 else tail_decay
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

        return SynthEngine.ensure_zero_crossing(y.astype(np.float32))

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
        swing_offset = int(sec_step * swing * 0.33 * SR)
        
        out = np.zeros(total_samples + int(SR * 0.5), dtype=np.float32)

        for s in slots:
            raw_data = s['data']
            if raw_data is None or len(raw_data) == 0: continue
            
            # Optimization: Calculate max possible length
            max_seq_len = total_samples + int(SR * 0.5) 

            # --- Reverse Logic ---
            # If the probability check passes, flip the audio array
            if rev_prob > 0.0 and np.random.random() < rev_prob:
                # Use ascontiguousarray to ensure memory safety after slicing
                raw_data = np.ascontiguousarray(raw_data[::-1])

            if clip_val > 0.0:
                # NEW CURVE: "Analog Gate" Style
                keep_ratio = 1.0 / (1.0 + (clip_val * 20.0))
                actual_len = max(150, int(len(raw_data) * keep_ratio))
                data = raw_data[:actual_len].copy()
                
                fade_samples = min(400, int(actual_len * 0.3))
                if fade_samples > 0:
                    data[-fade_samples:] *= np.linspace(1, 0, fade_samples)
            else:
                if len(raw_data) > max_seq_len:
                     data = raw_data[:max_seq_len]
                else:
                     data = raw_data

            s_len = len(data)
            
            for i, (active, vel) in enumerate(zip(s['pattern'], s['velocities'])):
                if active:
                    start_pos = int(i * sec_step * SR)
                    if i % 2 != 0: start_pos += swing_offset
                    
                    if start_pos < len(out):
                        write_len = min(s_len, len(out) - start_pos)
                        out[start_pos : start_pos + write_len] += data[:write_len] * (vel ** 1.5)
        
        tail = out[total_samples:]
        wrap_len = min(len(tail), total_samples)
        out[:wrap_len] += tail[:wrap_len]
        
        final = out[:total_samples]
        np.clip(final, -1.0, 1.0, out=final)
        
        peak = np.max(np.abs(final))
        if peak > 0.95: final *= (0.95 / peak)
            
        return final

# --- UI Components ---

class LogoWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(54, 54) # Increased from 42 for better visibility
        self.grid_size = 4
        self.cell_size = 10 # Slightly larger cells
        # Initialize colors (R, G, B) - Pale Palette
        self.current = np.random.uniform(180, 255, (4, 4, 3))
        self.targets = np.random.uniform(180, 255, (4, 4, 3))
        
        # Font setup for the text below
        self.text_font = QFont("Segoe UI", 11, QFont.Weight.Bold)
        self.text_font.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, 0.5)

    def animate(self):
        # Smooth interpolation towards target colors
        self.current += (self.targets - self.current) * 0.08
        
        # Randomly pick new targets for a few cells
        if np.random.random() < 0.3:
            r, c = np.random.randint(0, self.grid_size, 2)
            # Target: Cool Pales (Low Red, Med Green, High Blue)
            self.targets[r,c] = [np.random.randint(120, 190), 
                                 np.random.randint(190, 230), 
                                 np.random.randint(230, 255)]
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # 1. Draw Grid (Centered horizontally, top)
        grid_w = self.grid_size * self.cell_size
        off_x = (self.width() - grid_w) / 2
        off_y = 2
        
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                rgb = self.current[r,c].astype(int)
                col = QColor(*rgb)
                x = off_x + c * self.cell_size
                y = off_y + r * self.cell_size
                
                # Subtle gradient per cell
                grad = QLinearGradient(x, y, x + self.cell_size, y + self.cell_size)
                grad.setColorAt(0, col)
                grad.setColorAt(1, col.darker(105))
                painter.fillRect(QRectF(x, y, self.cell_size-1, self.cell_size-1), grad)

class PlayButton(QPushButton):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(80, 26)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.playing = False

    def set_playing(self, state):
        self.playing = state
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        bg = QColor("#3f6c9b")
        if self.underMouse(): bg = bg.lighter(110)
        if self.isDown(): bg = bg.darker(110)
        painter.setBrush(bg)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(self.rect(), 13, 13)
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

    def __init__(self, index, parent=None):
        super().__init__(parent)
        self.index = index
        self.active = False
        self.velocity = 0.8
        self.is_playing_head = False
        self.flash_val = 0.0 # For hit animation
        self.is_downbeat = (index % 4 == 0)
        self.setFixedWidth(28)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setMouseTracking(True)

    def set_playing(self, playing):
        if self.is_playing_head != playing:
            self.is_playing_head = playing
            # Trigger flash animation if turning active
            if playing and self.active:
                self.flash_val = 1.0
            self.update()

    def process_mouse_input(self, local_pos, force_state=None):
        h = self.height()
        y = max(0.0, min(float(h), local_pos.y()))
        new_vel = max(0.1, min(1.0, 1.0 - (y / h)))

        if force_state is not None:
            if self.active != force_state:
                self.active = force_state
                if self.active: self.velocity = new_vel
                self.toggled.emit(self.active, self.velocity)
            elif self.active:
                if abs(new_vel - self.velocity) > 0.01:
                    self.velocity = new_vel
                    self.velocity_changed.emit(self.velocity)
        else:
            self.active = not self.active
            if self.active: self.velocity = new_vel
            self.toggled.emit(self.active, self.velocity)
        self.update()

    def mousePressEvent(self, event):
        if event.button() in (Qt.MouseButton.LeftButton, Qt.MouseButton.RightButton):
            self.process_mouse_input(event.pos())
            if self.parent() and hasattr(self.parent(), 'start_painting'):
                self.parent().start_painting(self.active)

    def mouseMoveEvent(self, event):
        if event.buttons() & (Qt.MouseButton.LeftButton | Qt.MouseButton.RightButton):
            if self.rect().contains(event.pos()):
                if hasattr(self.parent(), 'paint_state'):
                    self.process_mouse_input(event.pos(), force_state=self.parent().paint_state)
            else:
                parent = self.parent()
                if parent:
                    pos_in_parent = self.mapTo(parent, event.pos())
                    child = parent.childAt(pos_in_parent)
                    if isinstance(child, StepPad) and child != self:
                         if hasattr(parent, 'paint_state'):
                            child.process_mouse_input(child.mapFrom(parent, pos_in_parent), force_state=parent.paint_state)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        r = self.rect().adjusted(1, 1, -1, -1)
        
        # Draw Base
        if self.active:
            c = QColor(63, 108, 155)
            # Apply flash effect: reduce opacity (alpha) temporarily when hit
            base_alpha = int(100 + (self.velocity * 155))
            if self.flash_val > 0.01:
                # Dip alpha significantly based on flash_val for "ghost" effect
                mod_alpha = base_alpha * (1.0 - (self.flash_val * 0.6))
                c.setAlpha(int(mod_alpha))
            else:
                c.setAlpha(base_alpha)
                
            painter.setBrush(c)
            painter.setPen(Qt.PenStyle.NoPen)
        else:
            painter.setBrush(QColor("#e2e8f0") if self.is_downbeat else QColor("#edf2f7"))
            painter.setPen(QPen(QColor("#cbd5e0"), 1))
            
        painter.drawRoundedRect(r, 3, 3)

        # Playhead Highlight
        if self.is_playing_head:
            # If active, we handled the visual in the block above, just add subtle overlay
            if not self.active:
                 painter.setBrush(QColor(70, 110, 160, 100))
                 painter.drawRoundedRect(r, 3, 3)

        # Velocity Line
        if self.active:
            ly = max(r.y()+2, min(r.bottom()-2, int(r.y() + r.height() * (1.0 - self.velocity))))
            painter.setPen(QPen(QColor(255, 255, 255, 200), 1))
            painter.drawLine(r.x()+4, ly, r.right()-4, ly)

        # Animation Loop
        if self.flash_val > 0.01:
            self.flash_val *= 0.85 # Decay speed
            self.update() # Request next frame
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

    # --- NEW: Click to Open Dialog ---
    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            fname, _ = QFileDialog.getOpenFileName(
                self, "Open Audio Sample", "", "Audio Files (*.wav *.mp3 *.aif *.flac)"
            )
            if fname:
                self.file_dropped.emit(fname)

class SlotRow(QFrame):
    pattern_changed = pyqtSignal()
    preview_req = pyqtSignal(object)
    saved_msg = pyqtSignal(str) 

    def __init__(self, label_text, drum_type, base_hue=0, parent=None):
        super().__init__(parent)
        self.label_text = label_text
        self.drum_type = drum_type
        
        # Audio Data State
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
        self.btn_reset.setToolTip("Remove sample and revert to synth")
        self.btn_reset.hide()
        self.btn_reset.clicked.connect(self.reset_to_synth)
        lf_layout.addWidget(self.btn_reset, 0)
        
        self.layout.addWidget(lbl_frame)

        # --- Controls ---
        ctrl_frame = QWidget()
        ctrl_layout = QHBoxLayout(ctrl_frame)
        ctrl_layout.setContentsMargins(0, 0, 4, 0)
        ctrl_layout.setSpacing(2) 
        
        self.btn_wav = self.create_btn("wav")
        self.btn_wav.clicked.connect(self.export_one)
        ctrl_layout.addWidget(self.btn_wav)
        
        self.btn_vel = self.create_btn("vel")
        self.btn_vel.clicked.connect(self.randomize_velocity)
        ctrl_layout.addWidget(self.btn_vel)

        self.btn_rnd = self.create_btn("rnd")
        self.btn_rnd.clicked.connect(self.syncopate_gentle)
        ctrl_layout.addWidget(self.btn_rnd)
        
        # FX Container
        crush_container = QWidget()
        cc_layout = QHBoxLayout(crush_container)
        cc_layout.setContentsMargins(2,0,2,0)
        cc_layout.setSpacing(2) 
        
        def make_v_slider(val, tip, handler, hue_offset):
            # Pass the calculated hue to the slider
            # hue_offset creates the gradient effect from left to right sliders
            sl = CircleSlider(Qt.Orientation.Vertical, base_hue=(base_hue + hue_offset) % 360)
            sl.setRange(0, 100)
            sl.setValue(val)
            sl.setFixedSize(20, 60) 
            sl.setToolTip(tip)
            sl.valueChanged.connect(handler)
            return sl

        # Create sliders with increasing hue offsets (0, 8, 16, 24, 32)
        self.sl_crush = make_v_slider(0, "Bitcrush", self.on_crush_change, 0)
        self.sl_filt = make_v_slider(50, "Filter", self.on_crush_change, 8)
        self.sl_pitch = make_v_slider(50, "Pitch", self.on_synth_change, 16)
        self.sl_decay = make_v_slider(50, "Decay", self.on_synth_change, 24)
        self.sl_tone = make_v_slider(30, "Tone", self.on_synth_change, 32)

        cc_layout.addWidget(self.sl_crush)
        cc_layout.addWidget(self.sl_filt)
        cc_layout.addWidget(self.sl_pitch)
        cc_layout.addWidget(self.sl_decay)
        cc_layout.addWidget(self.sl_tone)
        
        ctrl_layout.addWidget(crush_container)
        self.layout.addWidget(ctrl_frame)

        # --- Pads ---
        for i in range(STEPS):
            p = StepPad(i, self)
            p.toggled.connect(lambda a, v, idx=i: self.update_step(idx, a, v))
            p.velocity_changed.connect(lambda v, idx=i: self.update_vel(idx, v))
            self.pads.append(p)
            self.layout.addWidget(p)

        self.btn_clr = ClearButton(self)
        self.btn_clr.clicked.connect(self.clear)
        self.layout.addSpacing(6)
        self.layout.addWidget(self.btn_clr)
        
        self.layout.addStretch() 
        self.update_sound()

    def load_sample(self, file_path):
        try:
            # 1. Read only metadata first to check length (optimization)
            # or just read and slice immediately.
            data, file_sr = sf.read(file_path)

            # Convert to mono if stereo
            if len(data.shape) > 1:
                data = np.mean(data, axis=1)

            # Keep only first 3 seconds
            max_samples = 3 * file_sr 
            if len(data) > max_samples:
                data = data[:max_samples]

            # Resample if mismatch
            if file_sr != SR:
                num_samples = int(len(data) * SR / file_sr)
                # Now resampling is fast because len(data) is small
                data = signal.resample(data, num_samples)
            
            # Normalize
            peak = np.max(np.abs(data))
            if peak > 0: data = data / peak
            
            # Trim silence & declick
            data = SynthEngine.ensure_zero_crossing(data.astype(np.float32))

            self.raw_sample = data
            self.is_sample_mode = True
            
            # Visual indication
            self.lbl.setStyleSheet("""
                QLabel { color: #38a169; font-weight: bold; font-size: 12px; margin-left: 2px;} 
                QLabel:hover { background-color: #f0fff4; }
            """)
            self.btn_reset.show()
            self.saved_msg.emit(f"loaded: {file_path.split('/')[-1]}")
            
            self.update_sound(play=True)
            
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
        if self.is_sample_mode and self.raw_sample is not None:
            # Apply Pitch, Decay, Tone to the raw sample
            self.original_data = SynthEngine.process_sample(self.raw_sample, self.synth_params)
        else:
            # Generate fresh synth sound
            self.original_data = SynthEngine.generate_drum(self.drum_type, self.synth_params)
            
        self.process_audio()
        if play and self.current_data is not None:
            self.preview_req.emit(self.current_data)
        self.pattern_changed.emit()

    def on_crush_change(self):
        self.process_audio()
        self.pattern_changed.emit()

    def process_audio(self):
        if self.original_data is None: return
        
        # Apply Filter first
        filt_val = self.sl_filt.value() / 100.0
        filtered = SynthEngine.apply_filter(self.original_data, filt_val)
        
        # Then Bitcrush
        crush = self.sl_crush.value() / 100.0
        self.current_data = SynthEngine.resample_lofi(filtered, crush)

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
        timestamp = int(time.time())
        safe_name = self.label_text.replace(" ", "_")
        filename = f"{safe_name}_{timestamp}.wav"
        try:
            sf.write(filename, self.current_data, SR)
            self.saved_msg.emit(f"saved: {filename}")
        except Exception as e:
            self.saved_msg.emit(f"err: {str(e)}")

    def drift_params(self, amount):
        if amount <= 0.01: return
        targets = [self.sl_crush, self.sl_filt, self.sl_pitch, self.sl_decay, self.sl_tone]
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
    def __init__(self, orientation, base_hue=210, parent=None):
        super().__init__(orientation, parent)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        
        self.base_hue = base_hue
        # Start at base hue
        self.current_hue = base_hue 
        self.color_1 = QColor.fromHsl(int(self.base_hue), 150, 160)
        self.color_2 = QColor.fromHsl(int((self.base_hue + 20) % 360), 180, 130)
        
        # Random phase so sliders don't pulse in perfect unison
        self.phase = np.random.rand() * 10

    def tick_color(self):
        # Slower, discrete-ish oscillation around the base hue (+/- 15 degrees)
        self.phase += 0.03
        hue_offset = np.sin(self.phase) * 15
        h = (self.base_hue + hue_offset) % 360
        
        # Generate palette based on new hue
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
            x = margin + (w * (self.value() - self.minimum()) / (self.maximum() - self.minimum()))
            
            # Groove
            painter.setBrush(QColor("#e2e8f0"))
            painter.drawRoundedRect(QRectF(margin, cy - 2, w, 4), 2, 2)
            
            # Active Groove
            painter.setBrush(QColor("#4299e1"))
            if x > margin:
                painter.drawRoundedRect(QRectF(margin, cy - 2, x - margin, 4), 2, 2)
            
            # Handle
            painter.setBrush(QColor("white"))
            painter.setPen(QPen(QColor("#4299e1"), 1))
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
        self.reverse_prob = 0.0 # NEW: Reverse probability
        self.step = -1
        self.last_processed_step = -1
        
        # Color animation ticker
        self.anim_tick_counter = 0

        self.fmt = QAudioFormat()
        self.fmt.setSampleRate(SR)
        self.fmt.setChannelCount(1)
        self.fmt.setSampleFormat(QAudioFormat.SampleFormat.Int16)
        
        self.sink = QAudioSink(QMediaDevices.defaultAudioOutput(), self.fmt)
        self.sink.setBufferSize(int(SR * 2 * (BUFFER_MS / 1000.0)) * 2)
        
        self.gen = LoopGenerator(self.fmt, self)
        self.preview = SoundPreview(self)

        self.timer = QTimer()
        self.timer.setTimerType(Qt.TimerType.PreciseTimer)
        self.timer.setInterval(5) 
        self.timer.timeout.connect(self.update_playhead)

        self.msg_timer = QTimer()
        self.msg_timer.setSingleShot(True)
        self.msg_timer.timeout.connect(self.start_fade_out)

        self.setup_ui()
        self.update_mix()
        self.sink.start(self.gen)

    def setup_ui(self):
        cw = QWidget()
        self.setCentralWidget(cw)
        cw.setStyleSheet("QWidget { font-family: 'Segoe UI', sans-serif; background-color: #f7fafc; }")
        
        main = QVBoxLayout(cw)
        main.setContentsMargins(15, 2, 15, 15) 
        main.setSpacing(2)

        # --- Top Control Header ---
        header = QHBoxLayout()
        header.setSpacing(5)
        header.setContentsMargins(0, 0, 0, 5)
        header.setAlignment(Qt.AlignmentFlag.AlignVCenter)

        # Logo
        logo_cont = QWidget()
        logo_layout = QVBoxLayout(logo_cont)
        logo_layout.setContentsMargins(0, 8, 0, 0) 
        self.logo = LogoWidget()
        logo_layout.addWidget(self.logo)
        header.addWidget(logo_cont)
        
        def setup_lbl(text):
            l = QLabel(text)
            # UPDATED: Increased width from 75 to 85 to fit "evolve 100%"
            l.setFixedWidth(85) 
            l.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            l.setStyleSheet("color: #4a5568; font-weight: bold; font-size: 12px; margin-right: 4px;")
            return l

        def setup_slider(val, callback):
            # Use a neutral base hue (Blue) for the top sliders
            sl = CircleSlider(Qt.Orientation.Horizontal, base_hue=210)
            sl.setRange(0, 100)
            sl.setValue(val)
            sl.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            sl.valueChanged.connect(callback)
            return sl

        # Controls
        self.lbl_bpm = setup_lbl(f"{self.bpm} bpm")
        sl_bpm = setup_slider(self.bpm, self.set_bpm)
        sl_bpm.setRange(60, 200)
        header.addWidget(self.lbl_bpm)
        header.addWidget(sl_bpm)

        self.lbl_swg = setup_lbl("swing")
        header.addWidget(self.lbl_swg)
        sl_swg = setup_slider(0, self.set_swing)
        sl_swg.sliderReleased.connect(self.update_mix) 
        header.addWidget(sl_swg)

        self.lbl_clip = setup_lbl("clip")
        header.addWidget(self.lbl_clip)
        header.addWidget(setup_slider(0, self.set_clip))

        self.lbl_evolve = setup_lbl("evolve")
        header.addWidget(self.lbl_evolve)
        header.addWidget(setup_slider(0, self.set_evolve))

        self.lbl_rev = setup_lbl("rev")
        header.addWidget(self.lbl_rev)
        header.addWidget(setup_slider(0, self.set_rev))

        main.addLayout(header)

        # --- Sequencer Area ---
        
        # Column Labels
        head_row = QWidget()
        head_layout = QHBoxLayout(head_row)
        head_layout.setContentsMargins(0, 0, 0, 2)
        head_layout.setSpacing(0)
        
        lbl_space = QWidget()
        lbl_space.setFixedWidth(55)
        head_layout.addWidget(lbl_space)

        ctrl_head = QWidget()
        ch_layout = QHBoxLayout(ctrl_head)
        ch_layout.setContentsMargins(0, 0, 4, 0)
        ch_layout.setSpacing(2)
        
        btn_space = QWidget()
        btn_space.setFixedWidth(94)
        ch_layout.addWidget(btn_space)
        
        lbl_cont = QWidget()
        lc_layout = QHBoxLayout(lbl_cont)
        lc_layout.setContentsMargins(2, 0, 2, 0)
        lc_layout.setSpacing(2)
        
        for txt in ["bit", "flt", "pch", "dec", "ton"]:
            l = QLabel(txt)
            l.setFixedWidth(18)
            l.setAlignment(Qt.AlignmentFlag.AlignCenter)
            l.setStyleSheet("font-size: 9px; font-weight: bold; color: #718096;")
            lc_layout.addWidget(l)

        ch_layout.addWidget(lbl_cont)
        head_layout.addWidget(ctrl_head)
        head_layout.addStretch()
        main.addWidget(head_row)
        
        # Instrument Rows
        drums = [("kick", "kick"), ("snare", "snare"), ("hat c", "closed hat"), 
                 ("hat o", "open hat"), ("clap", "clap"), ("wood", "perc a"), ("tom", "perc b")]
        
        for i, (l, t) in enumerate(drums):
            # UPDATED: Calculate Hue based on index (Red 0 -> Purple 280)
            # 7 instruments -> approx 40 degrees shift per row
            row_hue = int((i / len(drums)) * 280) 
            
            r = SlotRow(l, t, base_hue=row_hue)
            r.pattern_changed.connect(self.update_mix)
            r.preview_req.connect(self.play_preview)
            r.saved_msg.connect(self.show_notification)
            main.addWidget(r) 
            self.slots.append(r)

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

        self.lbl_saved_msg = QLabel("")
        self.lbl_saved_msg.setStyleSheet("color: #4299e1; font-weight: bold; font-size: 12px; margin-left: 10px;")
        self.fade_effect = QGraphicsOpacityEffect(self.lbl_saved_msg)
        self.lbl_saved_msg.setGraphicsEffect(self.fade_effect)
        
        self.fade_anim = QPropertyAnimation(self.fade_effect, b"opacity")
        self.fade_anim.setDuration(2000) 
        self.fade_anim.setEasingCurve(QEasingCurve.Type.InOutQuad)
        self.fade_anim.finished.connect(lambda: self.lbl_saved_msg.setText(""))
        
        bot.addWidget(self.btn_play)
        bot.addSpacing(5)
        bot.addWidget(btn_exp)
        bot.addWidget(self.lbl_saved_msg)
        bot.addStretch()
        bot.addWidget(btn_clr)
        main.addLayout(bot)
    
    def set_bpm(self, v):
        if self.bpm == v: return
        
        if self.gen.playing:
            loop_duration_old = (60.0 / self.bpm) * 4.0
            elapsed = time.perf_counter() - self.start_time
            phase = (elapsed % loop_duration_old) / loop_duration_old
            
            self.bpm = v
            
            loop_duration_new = (60.0 / self.bpm) * 4.0
            self.start_time = time.perf_counter() - (phase * loop_duration_new)
        else:
            self.bpm = v
            
        self.lbl_bpm.setText(f"{v} bpm")
        self.update_mix()

    def set_swing(self, v):
        self.swing = v / 100.0
        self.lbl_swg.setText(f"swing {v}%")

    def set_clip(self, v):
        self.clip_amount = v / 100.0
        self.lbl_clip.setText(f"clip {v}%")
        self.update_mix()
    
    # --- NEW: Set Rev ---
    def set_rev(self, v):
        self.reverse_prob = v / 100.0
        self.lbl_rev.setText(f"rev {v}%")
        self.update_mix()

    def set_evolve(self, v):
        self.evolve_amount = v / 100.0
        self.lbl_evolve.setText(f"evolve {v}%")

    def sync_all(self):
        for s in self.slots: s.syncopate_gentle()

    def clear_all(self):
        for s in self.slots: s.clear()

    def update_mix(self):
        slot_data = []
        for s in self.slots:
            slot_data.append({'data': s.current_data, 'pattern': s.pattern, 'velocities': s.velocities})
        # Pass reverse_prob to mix_sequence
        self.gen.set_data(AudioMixer.mix_sequence(slot_data, self.bpm, self.swing, 
                                                  self.clip_amount, self.reverse_prob))

    def play_preview(self, data):
        self.preview.play(data)

    def toggle_play(self):
        if self.gen.playing:
            self.gen.set_playback_state(False)
            self.timer.stop()
            self.btn_play.set_playing(False)
            self.reset_vis()
        else:
            self.gen.pos = 0 
            self.gen.set_playback_state(True)
            self.start_time = time.perf_counter()
            self.timer.start()
            self.btn_play.set_playing(True)
            self.step = -1
            self.last_processed_step = -1
            self.update_playhead()

    def update_playhead(self):
        # Always animate logo
        self.logo.animate()
        
        # --- NEW: Animate Slider Colors (Throttled) ---
        self.anim_tick_counter += 1
        if self.anim_tick_counter > 15: # Approx every 75ms
            self.anim_tick_counter = 0
            for s in self.slots:
                # Update colors for vertical sliders
                s.sl_crush.tick_color()
                s.sl_filt.tick_color()
                s.sl_pitch.tick_color()
                s.sl_decay.tick_color()
                s.sl_tone.tick_color()

        if not self.gen.playing: return

        # --- EVOLVE DRIFT ---
        if self.evolve_amount > 0.01:
            for s in self.slots: 
                s.drift_params(self.evolve_amount)

        loop_duration = (60.0 / self.bpm) * 4.0
        elapsed = time.perf_counter() - self.start_time
        current_step = int((elapsed % loop_duration) / loop_duration * STEPS)
        
        if current_step != self.step:
            self.step = current_step if current_step < STEPS else 0
            for s in self.slots: s.highlight(self.step)

            if self.step != self.last_processed_step:
                if self.evolve_amount > 0.05:
                    if np.random.random() < (self.evolve_amount * 0.3): 
                        s_idx = np.random.randint(len(self.slots))
                        slot = self.slots[s_idx]
                        p_idx = np.random.randint(STEPS)
                        new_state = not slot.pattern[p_idx]
                        vel = slot.velocities[p_idx]
                        slot.pads[p_idx].active = new_state
                        slot.pads[p_idx].update()
                        slot.update_step(p_idx, new_state, vel)
            
            self.last_processed_step = self.step

    def reset_vis(self):
        self.step = -1
        for s in self.slots: s.highlight(-1)

    def show_notification(self, text):
        self.fade_anim.stop()           
        self.msg_timer.stop()           
        self.lbl_saved_msg.setText(text)
        self.fade_effect.setOpacity(1.0) 
        self.msg_timer.start(2000)   

    def start_fade_out(self):
        self.fade_anim.setStartValue(1.0)
        self.fade_anim.setEndValue(0.0)
        self.fade_anim.start()

    def export_beat(self):
        slot_data = []
        for s in self.slots:
            slot_data.append({'data': s.current_data, 'pattern': s.pattern, 'velocities': s.velocities})
        # Pass reverse_prob to mix_sequence
        mix = AudioMixer.mix_sequence(slot_data, self.bpm, self.swing, 
                                      self.clip_amount, self.reverse_prob)
        final = np.tile(mix, 2)
        timestamp = int(time.time())
        filename = f"sequa_loop_{self.bpm}bpm_{timestamp}.wav"
        try:
            sf.write(filename, final, SR)
            self.show_notification(f"exported: {filename}")
        except Exception as e:
             QMessageBox.critical(self, "error", f"could not save: {e}")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    if hasattr(Qt.ApplicationAttribute, 'AA_EnableHighDpiScaling'):
        app.setAttribute(Qt.ApplicationAttribute.AA_EnableHighDpiScaling, True)
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