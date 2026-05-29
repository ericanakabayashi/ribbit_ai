import io
import os
import warnings
import numpy as np
import librosa
from scipy import signal

warnings.filterwarnings("ignore", category=UserWarning)


class AudioPipeline:
    """Audio preprocessing pipeline for frog call classification.

    Supports loading from local path, raw bytes, or S3.
    """

    def __init__(
        self,
        file_path: str | None = None,
        audio_bytes: bytes | None = None,
        file_key: str | None = None,
        s3_bucket: str | None = None,
        method: str = "STFT",
        spect: str = "mel",
        remove_background: bool = True,
        remove_crop_audio: bool = True,
        plot_spectrogram: bool = False,
        plot_envelope: bool = False,
        normalize_audio: bool = False,
        sr: int = 48000,
        add_augmentation: bool = False,
        bg_file_path: str | None = None,
        bg_file_key: str | None = None,
        bg_bucket: str | None = None,
    ):
        self.file_path = file_path
        self.audio_bytes = audio_bytes
        self.file_key = file_key
        self.s3_bucket = s3_bucket

        self.sampling_rate = sr
        self.spect = spect
        self.method = method
        self.remove_background = remove_background
        self.remove_crop_audio = remove_crop_audio
        self.plot_spectrogram = plot_spectrogram
        self.plot_envelope = plot_envelope
        self.normalize_audio = normalize_audio
        self.add_augmentation = add_augmentation

        self.bg_file_path = bg_file_path
        self.bg_file_key = bg_file_key
        self.bg_bucket = bg_bucket

        self.audio_data: np.ndarray | None = None
        self.cleaned_audio: np.ndarray | None = None
        self.spectrogram: np.ndarray | None = None
        self.envelope: np.ndarray | None = None
        self.bg_noise: np.ndarray | None = None

    # ── loading ────────────────────────────────────────────────────────────

    def load_audio(self):
        duration = 3 if self.remove_crop_audio else None
        if self.audio_bytes is not None:
            self.audio_data, self.sampling_rate = librosa.load(
                io.BytesIO(self.audio_bytes), sr=self.sampling_rate, duration=duration
            )
        elif self.s3_bucket and self.file_key:
            self.audio_data, self.sampling_rate = self._load_from_s3(
                self.s3_bucket, self.file_key, duration
            )
        elif self.file_path:
            if not os.path.exists(self.file_path):
                raise FileNotFoundError(f"File not found: {self.file_path}")
            self.audio_data, self.sampling_rate = librosa.load(
                self.file_path, sr=self.sampling_rate, duration=duration
            )
        else:
            raise ValueError("No audio source provided (file_path, audio_bytes, or s3_bucket+file_key).")

        if not self.audio_data.size:
            raise ValueError("Loaded audio data is empty.")

        if self.normalize_audio:
            self._gaussian_norm()

    def _load_from_s3(self, bucket: str, key: str, duration):
        import boto3

        obj = boto3.client("s3").get_object(Bucket=bucket, Key=key)
        raw = obj["Body"].read()
        return librosa.load(io.BytesIO(raw), sr=self.sampling_rate, duration=duration)

    def _gaussian_norm(self):
        std = np.std(self.audio_data)
        if std > 0:
            self.audio_data = (self.audio_data - np.mean(self.audio_data)) / std

    # ── augmentation ───────────────────────────────────────────────────────

    def load_background_audio(self):
        if self.bg_file_path:
            if not os.path.exists(self.bg_file_path):
                raise FileNotFoundError(f"Background file not found: {self.bg_file_path}")
            self.bg_noise, bg_sr = librosa.load(self.bg_file_path, sr=None)
        elif self.bg_bucket and self.bg_file_key:
            self.bg_noise, bg_sr = self._load_from_s3(self.bg_bucket, self.bg_file_key, None)
        else:
            return

        if bg_sr != self.sampling_rate:
            self.bg_noise = librosa.resample(self.bg_noise, orig_sr=bg_sr, target_sr=self.sampling_rate)

        n = len(self.audio_data)
        if len(self.bg_noise) < n:
            self.bg_noise = np.pad(self.bg_noise, (0, n - len(self.bg_noise)), "constant")
        else:
            self.bg_noise = self.bg_noise[:n]

    def add_background_noise(self):
        if self.bg_noise is None or not np.isfinite(self.bg_noise).all():
            return
        max_bg = np.max(np.abs(self.bg_noise))
        bg = self.bg_noise / max_bg if max_bg > 0 else self.bg_noise
        n = min(len(self.audio_data), len(bg))
        self.audio_data[:n] += bg[:n]

    # ── denoising ─────────────────────────────────────────────────────────

    def remove_background_noise(self):
        if self.method == "STFT":
            self.cleaned_audio = self._denoise_stft()
        elif self.method == "PCEN":
            self.cleaned_audio = self._denoise_pcen()
        elif self.method == "HPF":
            self.cleaned_audio = self._high_pass_filter(self.audio_data)

    def _denoise_stft(self) -> np.ndarray:
        if not np.isfinite(self.audio_data).all():
            raise ValueError("Audio data contains non-finite values.")
        S = librosa.stft(self.audio_data)
        mag, phase = np.abs(S), np.angle(S)
        noise = np.median(mag, axis=1, keepdims=True)
        cleaned = np.where(mag < noise, 0, mag) * np.exp(1j * phase)
        return librosa.istft(cleaned, dtype=np.float64)

    def _denoise_pcen(self) -> np.ndarray:
        S = librosa.stft(self.audio_data)
        mag = np.abs(S)
        pcen = librosa.pcen(
            mag * (2**31),
            sr=self.sampling_rate,
            gain=1.1,
            hop_length=512,
            bias=2,
            power=0.5,
            time_constant=0.8,
            eps=1e-6,
            max_size=2,
        )
        return librosa.istft(pcen * np.exp(1j * np.angle(S)), dtype=np.float64)

    def _high_pass_filter(self, y: np.ndarray) -> np.ndarray:
        b, a = signal.butter(10, 2000 / (self.sampling_rate / 2), btype="highpass")
        return signal.lfilter(b, a, y)

    # ── spectrogram ────────────────────────────────────────────────────────

    def compute_spectrogram(self):
        audio = self.cleaned_audio if self.remove_background and self.cleaned_audio is not None else self.audio_data
        if self.spect == "mel":
            mel = librosa.feature.melspectrogram(y=audio, sr=self.sampling_rate)
            self.spectrogram = librosa.power_to_db(mel, ref=np.max)
        else:
            S = librosa.stft(audio)
            self.spectrogram = librosa.amplitude_to_db(np.abs(S), ref=np.max)
        self.envelope = np.abs(self.spectrogram).max(axis=0)

    # ── plotting (optional, not needed for inference) ──────────────────────

    def plot_results(self):
        import matplotlib.pyplot as plt

        num_plots = int(self.plot_spectrogram) + int(self.plot_envelope)
        if num_plots == 0 or self.spectrogram is None:
            return

        plt.figure(figsize=(5.11, 0.96), dpi=100)
        if self.plot_spectrogram:
            plt.subplot(num_plots, 1, 1)
            plt.imshow(self.spectrogram, aspect="auto", origin="lower", cmap="gray")
            plt.axis("off")
        if self.plot_envelope:
            plt.subplot(num_plots, 1, 2 if self.plot_spectrogram else 1)
            plt.plot(self.envelope, color="gray")
            plt.axis("off")
        plt.tight_layout()
        plt.show()

    # ── main entry point ───────────────────────────────────────────────────

    def run_pipeline(self):
        self.load_audio()
        if self.add_augmentation:
            self.load_background_audio()
            self.add_background_noise()
        if self.remove_background:
            self.remove_background_noise()
        self.compute_spectrogram()
        if self.plot_spectrogram or self.plot_envelope:
            self.plot_results()
