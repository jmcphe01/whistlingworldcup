"""Whistle detection: bandpass filter + dominance ratio + duration gate.

Usage:
    detector = WhistleDetector(sample_rate=44100, center_freq=3000, bandwidth=200)
    ...
    is_whistle = detector.process(chunk)  # chunk = 1-D float array, one audio block
"""

import numpy as np
from scipy.signal import butter, sosfilt, sosfilt_zi


class WhistleDetector:
    """Detects a sustained whistle tone in a stream of audio chunks.

    Each call to process() runs three checks, in order:

    1. Bandpass filter -- the raw chunk is passed through a strict IIR
       bandpass filter centered on `center_freq` with total width
       `bandwidth` (so `center_freq` +/- `bandwidth/2`). The filter keeps
       state (`self._zi`) across calls, so it filters the audio as one
       continuous stream instead of restarting (and glitching) at every
       chunk boundary.

    2. Dominance ratio -- RMS(filtered chunk) / RMS(raw chunk). This is
       "what fraction of this chunk's total loudness lives inside the
       whistle band". A pure tone at center_freq scores near 1.0; broadband
       noise (a hammer strike, a drill, speech) spreads its energy across
       the whole spectrum and scores low even if it's loud.

    3. Duration gate -- the dominance ratio must stay >= dominance_threshold
       *continuously* for at least min_duration seconds before process()
       returns True. This rejects brief broadband transients that might
       momentarily spike the ratio, and matches how an actual whistle
       sounds: a held tone, not an instant.
    """

    def __init__(
        self,
        sample_rate,
        center_freq=3000.0,       # Hz -- tune this to your whistle's pitch
        bandwidth=200.0,          # Hz -- total pass width (+/- bandwidth/2)
        dominance_threshold=0.6,  # filtered/raw RMS ratio required to count as "whistling"
        min_duration=0.4,         # seconds the ratio must hold continuously to trigger
        filter_order=4,
    ):
        self.sample_rate = sample_rate
        self.dominance_threshold = dominance_threshold
        self.min_duration = min_duration

        nyquist = sample_rate / 2
        low = max((center_freq - bandwidth / 2) / nyquist, 1e-6)
        high = min((center_freq + bandwidth / 2) / nyquist, 1 - 1e-6)
        if not low < high:
            raise ValueError("bandwidth/center_freq produce an empty passband")

        self._sos = butter(filter_order, [low, high], btype="bandpass", output="sos")
        self._zi = sosfilt_zi(self._sos) * 0.0  # filter's internal state, starts at rest

        # Public, read-only-in-spirit: last computed values, handy for a UI
        # (see whistle_visualizer.py) that wants to show *why* it did or
        # didn't trigger, without recomputing anything.
        self.dominance = 0.0
        self.seconds_above = 0.0  # how long the dominance ratio has held, continuously

    def process(self, chunk):
        """Feed one chunk of raw audio (1-D array-like). Returns True the
        moment the duration gate is satisfied, False otherwise."""
        chunk = np.asarray(chunk, dtype=np.float64)

        filtered, self._zi = sosfilt(self._sos, chunk, zi=self._zi)

        raw_rms = np.sqrt(np.mean(chunk ** 2)) + 1e-12
        band_rms = np.sqrt(np.mean(filtered ** 2))
        self.dominance = band_rms / raw_rms

        chunk_seconds = len(chunk) / self.sample_rate
        if self.dominance >= self.dominance_threshold:
            self.seconds_above += chunk_seconds
        else:
            self.seconds_above = 0.0

        return self.seconds_above >= self.min_duration

    def reset(self):
        """Clear filter and duration-gate state (e.g. after a trigger, or
        when switching to a different audio source)."""
        self._zi = sosfilt_zi(self._sos) * 0.0
        self.dominance = 0.0
        self.seconds_above = 0.0


class PitchWhistleDetector:
    """Classifies a sustained whistle's pitch into one of six zones, for
    the ME193 WhistleSoccer assignment: a low pitch means STOP, a high
    pitch means FORWARD ("speed up"), and the range in between splits into
    four fixed-angle pivot commands -- one continuous pitch axis, six
    zones, low to high:

        STOP | pivot[90] | pivot[45] | pivot[-45] | pivot[-90] | FORWARD
      freq_min                                                  freq_max

    A pivot command's bracketed number is the angle to turn in degrees:
    positive = turn left that many degrees, negative = turn right that
    many degrees (so pivot[90]/pivot[45] turn left, pivot[-45]/pivot[-90]
    turn right -- two step sizes each way instead of one open-ended
    "keep turning" command per direction).

    Zone names spell out the full command (`pivot[90]`, not `LEFT`)
    because the MQTT channel is shared with other teams -- a bare "LEFT"
    or "RIGHT" is too likely to collide with someone else's message on the
    same broker/topic namespace.

    Noise rejection works the same way as WhistleDetector -- a bandpass
    filter across the *whole* whistle range plus a dominance-ratio/
    duration gate decide *whether* a whistle is happening at all -- and
    only once that's confirmed does it estimate *which* pitch, via FFT
    peak-picking restricted to that same range.
    """

    # low pitch -> high pitch, in order
    ZONES = ("STOP", "pivot[90]", "pivot[45]", "pivot[-45]", "pivot[-90]", "FORWARD")

    def __init__(
        self,
        sample_rate,
        freq_min=1200.0,          # Hz -- bottom of the whistle range (-> STOP)
        freq_max=4500.0,          # Hz -- top of the whistle range (-> FORWARD)
        dominance_threshold=0.5,  # filtered/raw RMS ratio required to count as "whistling"
        min_duration=0.25,        # seconds the ratio must hold continuously to count
        filter_order=4,
    ):
        self.sample_rate = sample_rate
        self.freq_min = freq_min
        self.freq_max = freq_max
        self.dominance_threshold = dominance_threshold
        self.min_duration = min_duration

        nyquist = sample_rate / 2
        low = max(freq_min / nyquist, 1e-6)
        high = min(freq_max / nyquist, 1 - 1e-6)
        if not low < high:
            raise ValueError("freq_min/freq_max produce an empty passband")

        self._sos = butter(filter_order, [low, high], btype="bandpass", output="sos")
        self._zi = sosfilt_zi(self._sos) * 0.0

        # Public: last computed values, for a live UI to show *why* it
        # decided what it decided, without recomputing anything.
        self.dominance = 0.0
        self.seconds_above = 0.0
        self.pitch_hz = None  # last estimated pitch, or None while no whistle is held
        self.zone = "STOP"    # last classified zone -- "STOP" doubles as the no-whistle default

    def process(self, chunk):
        """Feed one chunk of raw audio (1-D array-like). Returns the
        classified zone name (one of ZONES). Returns "STOP" both when a
        low-pitched whistle is heard AND when no whistle is heard at all
        -- silence is the assignment's required no-whistle behavior, and
        it's indistinguishable from "the pitch is at the bottom of the
        range" without a whistle, so the two share the same zone."""
        chunk = np.asarray(chunk, dtype=np.float64)

        filtered, self._zi = sosfilt(self._sos, chunk, zi=self._zi)

        raw_rms = np.sqrt(np.mean(chunk ** 2)) + 1e-12
        band_rms = np.sqrt(np.mean(filtered ** 2))
        self.dominance = band_rms / raw_rms

        chunk_seconds = len(chunk) / self.sample_rate
        if self.dominance >= self.dominance_threshold:
            self.seconds_above += chunk_seconds
        else:
            self.seconds_above = 0.0

        if self.seconds_above >= self.min_duration:
            self.pitch_hz = self._estimate_pitch(filtered)
            self.zone = self._classify(self.pitch_hz)
        else:
            self.pitch_hz = None
            self.zone = "STOP"

        return self.zone

    def _estimate_pitch(self, filtered_chunk):
        """Peak frequency of the filtered chunk's spectrum. The bandpass
        filter already suppresses everything outside [freq_min, freq_max],
        so this just finds where the remaining energy peaked."""
        windowed = filtered_chunk * np.hanning(len(filtered_chunk))
        spectrum = np.abs(np.fft.rfft(windowed))
        freqs = np.fft.rfftfreq(len(filtered_chunk), 1 / self.sample_rate)
        return float(freqs[np.argmax(spectrum)])

    def _classify(self, pitch_hz):
        span = self.freq_max - self.freq_min
        frac = (pitch_hz - self.freq_min) / span  # 0 (low) .. 1 (high)
        if frac < 1 / 6:
            return "STOP"
        if frac < 2 / 6:
            return "pivot[90]"
        if frac < 3 / 6:
            return "pivot[45]"
        if frac < 4 / 6:
            return "pivot[-45]"
        if frac < 5 / 6:
            return "pivot[-90]"
        return "FORWARD"

    def reset(self):
        """Clear filter and duration-gate state (e.g. when switching to a
        different audio source)."""
        self._zi = sosfilt_zi(self._sos) * 0.0
        self.dominance = 0.0
        self.seconds_above = 0.0
        self.pitch_hz = None
        self.zone = "STOP"
