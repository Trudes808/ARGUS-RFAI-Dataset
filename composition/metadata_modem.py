"""OFDM metadata modem for the 245.76 MSps composition pipeline.

Encodes a block's metadata (waveform class / variation / time-frequency
positions / bandwidth / power) into a robust OFDM burst that survives a simple
wireless channel, and decodes it back. The burst is generated at a native rate
(default 30.72 MHz) and is resampled to the 245.76 MSps composite rate by the
composer; this module works purely at the native rate.

Frame on the air (native rate):
    [ reference OFDM symbol ][ n_data_syms data OFDM symbols ]
- reference symbol  : known BPSK on every active subcarrier -> per-subcarrier
                      least-squares channel estimate at the receiver.
- data symbols      : QPSK on data subcarriers, known QPSK pilots on pilot
                      subcarriers (residual common-phase tracking).
- FEC               : rate-1/rep repetition with a fixed pseudo-random
                      interleaver, soft-combined at the receiver (simple, robust,
                      no Viterbi needed); CRC-32 validates the payload.

The burst length is a fixed system constant (ref + n_data_syms symbols), so the
receiver can extract it deterministically at a known offset after the ZC.
"""

from __future__ import annotations
import struct
import zlib
import numpy as np


class MetaModem:
    def __init__(self, fs_native=30.72e6, nfft=256, cp=64, n_active=200,
                 pilot_spacing=8, n_data_syms=20, rep=3, seed=0xC0FFEE, smooth_width=1):
        self.fs_native = float(fs_native)
        self.nfft = int(nfft)
        self.cp = int(cp)
        self.n_active = int(n_active)
        self.pilot_spacing = int(pilot_spacing)
        self.n_data_syms = int(n_data_syms)
        self.rep = int(rep)
        self.bps = 2                      # QPSK data subcarriers
        self.smooth_width = int(smooth_width)   # channel-estimate freq smoothing (1 = off)
        self.sym_len = self.nfft + self.cp

        # centered active subcarrier indices (DC excluded), then pilot/data split
        c = self.nfft // 2
        half = self.n_active // 2
        active = np.concatenate([c + np.arange(-half, 0), c + np.arange(1, half + 1)])
        self.active = active.astype(int)
        pilot_mask = np.zeros(self.n_active, dtype=bool)
        pilot_mask[0::self.pilot_spacing] = True
        self.pilot_idx = self.active[pilot_mask]
        self.data_idx = self.active[~pilot_mask]
        self.n_pilot = self.pilot_idx.size
        self.n_data_sc = self.data_idx.size

        rng = np.random.default_rng(seed)
        # known reference symbol (BPSK) on all active subcarriers
        self.ref_syms = (1 - 2 * rng.integers(0, 2, self.n_active)).astype(np.complex128)
        # known pilots (QPSK) per data symbol
        pb = rng.integers(0, 2, (self.n_pilot, self.n_data_syms, 2))
        self.pilots = ((1 - 2 * pb[..., 0]) + 1j * (1 - 2 * pb[..., 1])) / np.sqrt(2)

        # capacity and a fixed interleaver over the full coded-bit space
        self.total_coded_bits = self.n_data_syms * self.n_data_sc * self.bps
        self.info_bits = self.total_coded_bits // self.rep
        self.capacity_bytes = self.info_bits // 8
        self.nbits = self.capacity_bytes * 8
        self.coded_used = self.rep * self.nbits
        self.perm = rng.permutation(self.total_coded_bits)
        self.inv_perm = np.argsort(self.perm)

    # ---- public API ----
    def payload_capacity_bytes(self):
        # usable payload = capacity minus 2-byte length and 4-byte CRC
        return self.capacity_bytes - 6

    def burst_len_samples(self):
        return (1 + self.n_data_syms) * self.sym_len

    def encode(self, payload: bytes) -> np.ndarray:
        payload = bytes(payload)
        if len(payload) > self.payload_capacity_bytes():
            raise ValueError(
                f"metadata payload {len(payload)} B exceeds modem capacity "
                f"{self.payload_capacity_bytes()} B; reduce waveforms in the block "
                f"or split into another block")
        frame = struct.pack(">H", len(payload)) + payload
        frame += struct.pack(">I", zlib.crc32(frame) & 0xFFFFFFFF)
        frame = frame + bytes(self.capacity_bytes - len(frame))  # zero pad
        bits = np.unpackbits(np.frombuffer(frame, dtype=np.uint8))[:self.nbits]

        coded = np.zeros(self.total_coded_bits, dtype=np.uint8)
        coded[:self.coded_used] = np.tile(bits, self.rep)
        coded = coded[self.perm]                       # interleave

        syms = self._qpsk_mod(coded)                    # (total_data_subc,)
        syms = syms.reshape(self.n_data_syms, self.n_data_sc)

        out = [self._ofdm_symbol(self.ref_syms, is_ref=True)]
        for k in range(self.n_data_syms):
            grid = np.zeros(self.nfft, dtype=np.complex128)
            grid[self.data_idx] = syms[k]
            grid[self.pilot_idx] = self.pilots[:, k]
            out.append(self._ofdm_time(grid))
        return np.concatenate(out).astype(np.complex64)

    def decode(self, wave: np.ndarray):
        wave = np.asarray(wave).ravel()
        need = self.burst_len_samples()
        if wave.size < need:
            wave = np.concatenate([wave, np.zeros(need - wave.size, dtype=wave.dtype)])
        wave = wave[:need]
        syms = wave.reshape(1 + self.n_data_syms, self.sym_len)

        # channel estimate from the reference symbol (LS, per active subcarrier),
        # smoothed across frequency to suppress estimation noise at low SNR (the
        # channel is flat/smooth, so a moving average is a big robustness win)
        ref_grid = self._demod_grid(syms[0])
        H = self._smooth(ref_grid[self.active] / self.ref_syms, self.smooth_width)
        H_data = H[~self._active_is_pilot()]
        H_pilot = H[self._active_is_pilot()]

        w_data = np.abs(H_data) ** 2          # MRC-like weighting for soft combine
        llr = np.zeros(self.total_data_bits(), dtype=np.float64)
        pos = 0
        for k in range(self.n_data_syms):
            g = self._demod_grid(syms[k + 1])
            eq_data = g[self.data_idx] / H_data
            eq_pilot = g[self.pilot_idx] / H_pilot
            # residual common-phase correction from pilots
            ph = np.angle(np.sum(eq_pilot * np.conj(self.pilots[:, k])))
            eq_data *= np.exp(-1j * ph)
            l = self._qpsk_llr(eq_data, w_data)
            llr[pos:pos + l.size] = l
            pos += l.size
        return self._finish_decode(llr)

    # ---- internals ----
    @staticmethod
    def _smooth(H, w):
        """Edge-corrected complex moving average across subcarriers."""
        if w <= 1 or H.size <= w:
            return H
        k = np.ones(w)
        num = np.convolve(H, k, mode="same")
        den = np.convolve(np.ones(H.size), k, mode="same")
        return num / den

    def total_data_bits(self):
        return self.n_data_syms * self.n_data_sc * self.bps

    def _finish_decode(self, llr_tx_order):
        # llr_tx_order is indexed in transmitted (post-permutation) order; invert
        llr_coded = np.empty(self.total_coded_bits)
        llr_coded[self.perm] = llr_tx_order
        used = llr_coded[:self.coded_used].reshape(self.rep, self.nbits)
        comb = used.sum(axis=0)
        bits = (comb < 0).astype(np.uint8)  # LLR>0 -> bit 0
        frame = np.packbits(bits).tobytes()
        n = struct.unpack(">H", frame[:2])[0]
        if 2 + n + 4 > len(frame):
            return None
        body = frame[:2 + n]
        crc_rx = struct.unpack(">I", frame[2 + n:2 + n + 4])[0]
        if (zlib.crc32(body) & 0xFFFFFFFF) != crc_rx:
            return None
        return frame[2:2 + n]

    def _active_is_pilot(self):
        mask = np.zeros(self.n_active, dtype=bool)
        mask[0::self.pilot_spacing] = True
        return mask

    def _qpsk_mod(self, bits):
        b = bits.reshape(-1, 2).astype(np.float64)   # avoid uint8 underflow in 1-2*b
        return ((1 - 2 * b[:, 0]) + 1j * (1 - 2 * b[:, 1])) / np.sqrt(2)

    def _qpsk_llr(self, sym, weight=None):
        # Gray QPSK: real -> bit0, imag -> bit1. LLR>0 favors bit 0.
        s = np.sqrt(2.0)
        if weight is None:
            weight = np.ones(sym.size)
        llr = np.empty(sym.size * 2)
        llr[0::2] = s * weight * np.real(sym)
        llr[1::2] = s * weight * np.imag(sym)
        return llr

    def _ofdm_time(self, grid_centered):
        t = np.fft.ifft(np.fft.ifftshift(grid_centered)) * np.sqrt(self.nfft)
        return np.concatenate([t[-self.cp:], t])

    def _ofdm_symbol(self, active_syms, is_ref=False):
        grid = np.zeros(self.nfft, dtype=np.complex128)
        grid[self.active] = active_syms
        return self._ofdm_time(grid)

    def _demod_grid(self, sym):
        x = sym[self.cp:]
        return np.fft.fftshift(np.fft.fft(x) / np.sqrt(self.nfft))
