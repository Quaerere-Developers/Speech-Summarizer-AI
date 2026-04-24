"""マイク・ループバック録音と WAV 出力（PyAudioWPatch / WASAPI）。

録音スレッドは read() とキュー投入のみ。NumPy 処理は別スレッドに寄せ、WASAPI の
data discontinuity を抑える。停止時はセンチネルでキューを完走させる。
"""

from __future__ import annotations

import queue
import sys
import threading
import time
import wave
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from speech_summarizer_ai import settings as config

_pa_module = None


def _pyaudio() -> Any:
    """PyAudioWPatch を遅延 import する。"""
    global _pa_module
    if _pa_module is None:
        import pyaudiowpatch as pyaudio

        _pa_module = pyaudio
    return _pa_module


def get_loopback_device_info(pa_instance: Any) -> dict | None:
    """既定スピーカーに対応する WASAPI ループバックデバイス情報を返す。

    Args:
        pa_instance: ``pyaudiowpatch.PyAudio`` インスタンス。

    Returns:
        dict | None: デバイス情報辞書。見つからなければ ``None``。
    """
    try:
        return pa_instance.get_default_wasapi_loopback()
    except Exception:
        return None


def to_mono(samples: np.ndarray) -> np.ndarray:
    """多チャンネル音声をモノラル float32 に変換する。

    Args:
        samples: 入力サンプル（1D または 2D）。

    Returns:
        np.ndarray: モノラル ``float32`` 1 次元配列。
    """
    if samples.size == 0:
        return np.zeros(0, dtype=np.float32)
    if samples.ndim == 1:
        return samples.astype(np.float32, copy=False)
    return np.mean(samples, axis=1).astype(np.float32)


def to_stereo_interleaved(mono_left: np.ndarray, mono_right: np.ndarray) -> np.ndarray:
    """左右モノラルをインターリーブされた int16 PCM に変換する。

    Args:
        mono_left: 左チャンネル（マイク相当）の float サンプル。
        mono_right: 右チャンネル（PC 音相当）の float サンプル。

    Returns:
        np.ndarray: LRLR… の ``int16`` 1 次元配列。
    """
    n = mono_left.shape[0]
    assert mono_right.shape[0] == n
    l = (np.clip(mono_left, -1.0, 1.0) * 32767.0).astype(np.int16)
    r = (np.clip(mono_right, -1.0, 1.0) * 32767.0).astype(np.int16)
    out = np.empty(n * 2, dtype=np.int16)
    out[0::2] = l
    out[1::2] = r
    return out


def mix_to_mono_pcm(mono_mic: np.ndarray, mono_pc: np.ndarray) -> np.ndarray:
    """マイクと PC 音をゲイン付きで合成し int16 PCM にする。

    Args:
        mono_mic: マイクのモノラル float サンプル。
        mono_pc: ループバックのモノラル float サンプル。

    Returns:
        np.ndarray: クリップ済み ``int16`` モノラル PCM。
    """
    mixed = mono_mic * config.MONO_MIC_GAIN + mono_pc * config.MONO_PC_GAIN
    return (np.clip(mixed, -1.0, 1.0) * 32767.0).astype(np.int16)


def _bytes_to_float32(data: bytes, channels: int) -> np.ndarray:
    """int16 PCM バイト列を shape=(frames, channels) の float32 配列に変換する。

    Args:
        data: int16 リトルエンディアン PCM バイト列。
        channels: チャンネル数。

    Returns:
        np.ndarray: shape ``(frames, channels)`` の ``float32`` 配列（-1.0〜1.0 正規化済み）。
    """
    arr = np.frombuffer(data, dtype=np.int16)
    if arr.size == 0 or channels == 0:
        return np.zeros((0, max(channels, 1)), dtype=np.float32)
    frames = arr.size // channels
    return (
        arr[: frames * channels].reshape(frames, channels).astype(np.float32) / 32768.0
    )


def _resample_to_target(
    frames: np.ndarray, src_rate: int, target_len: int
) -> np.ndarray:
    """線形補間で ``frames`` を ``target_len`` サンプルにリサンプルする。

    Args:
        frames: shape ``(src_len, channels)`` の float32 配列。
        src_rate: 入力のサンプルレート（ログ用。実際には src_len と target_len の比で計算）。
        target_len: 出力フレーム数。

    Returns:
        np.ndarray: shape ``(target_len, channels)`` の float32 配列。
    """
    src_len = frames.shape[0]
    if src_len == 0 or target_len == 0:
        return np.zeros(
            (target_len, frames.shape[1] if frames.ndim > 1 else 1), dtype=np.float32
        )
    if src_len == target_len:
        return frames
    indices = np.linspace(0, src_len - 1, target_len)
    low = np.floor(indices).astype(np.int32)
    high = np.minimum(low + 1, src_len - 1)
    frac = (indices - low).astype(np.float32)
    if frames.ndim == 1:
        return frames[low] * (1.0 - frac) + frames[high] * frac
    frac_2d = frac[:, np.newaxis]
    return frames[low] * (1.0 - frac_2d) + frames[high] * frac_2d


def _process_raw_pair(
    m: np.ndarray,
    l: np.ndarray | None,
    nm: int,
    *,
    chunks: list[np.ndarray],
    chunk_lock: threading.Lock,
    live_mono_chunk_callback: Callable[[np.ndarray], None] | None,
) -> None:
    """1 組の生フレームを混合し ``chunks`` に積み、任意でライブ STT に渡す（処理スレッド専用）。

    Args:
        m: マイクの float32 配列 shape ``(nm, channels)``。
        l: ループバック。無い場合は ``None``。
        nm: マイクフレームのサンプル数。
        chunks: PCM チャンクのリスト。
        chunk_lock: ``chunks`` 用ロック。
        live_mono_chunk_callback: 任意。モノラル int16 を渡す。

    Returns:
        None
    """
    mono_m = to_mono(m)
    if l is not None:
        nl = int(l.shape[0])
        if nl == 0:
            mono_l = np.zeros(nm, dtype=np.float32)
        else:
            mono_l = to_mono(l)
            if nl != nm:
                mono_l = to_mono(_resample_to_target(l, 0, nm))
    else:
        mono_l = np.zeros(nm, dtype=np.float32)

    if config.RECORD_LAYOUT == "mono":
        pcm = mix_to_mono_pcm(mono_m, mono_l)
    else:
        pcm = to_stereo_interleaved(mono_m, mono_l)
    with chunk_lock:
        chunks.append(pcm)
    if live_mono_chunk_callback is not None:
        if config.RECORD_LAYOUT == "mono":
            live_mono_chunk_callback(pcm.copy())
        else:
            st = pcm.reshape(-1, 2)
            mono = ((st[:, 0].astype(np.int32) + st[:, 1]) // 2).astype(np.int16)
            live_mono_chunk_callback(mono)


def capture_loop(
    mic_stream: Any,
    loopback_stream: Any | None,
    mic_channels: int,
    loopback_channels: int,
    loopback_rate: int,
    stop_event: threading.Event,
    chunks: list[np.ndarray],
    chunk_lock: threading.Lock,
    live_mono_chunk_callback: Callable[[np.ndarray], None] | None = None,
) -> None:
    """PyAudioWPatch ストリームからマイク（と任意のループバック）を録音し ``chunks`` に積む。

    マイクは ``config.SAMPLE_RATE`` で開く。ループバックはデバイスのネイティブレートで開き、
    サンプルレートが異なる場合は処理スレッド内で線形補間リサンプルする。

    Args:
        mic_stream: マイク PyAudio ストリーム。
        loopback_stream: ループバック PyAudio ストリーム。無ければ ``None``。
        mic_channels: マイクのチャンネル数。
        loopback_channels: ループバックのチャンネル数。
        loopback_rate: ループバックのサンプルレート（Hz）。
        stop_event: ループ終了用イベント。
        chunks: PCM チャンクのリスト。
        chunk_lock: ``chunks`` 用ロック。
        live_mono_chunk_callback: 任意。リアルタイム STT 用 int16 モノラル（処理スレッドから呼ぶ）。

    Returns:
        None
    """
    # ループバック読み取りフレーム数：サンプルレートが異なる場合に比例調整する
    loopback_block_frames = (
        int(config.BLOCK_FRAMES * loopback_rate / config.SAMPLE_RATE)
        if loopback_stream is not None and loopback_rate != config.SAMPLE_RATE
        else config.BLOCK_FRAMES
    )

    raw_queue: queue.Queue[tuple[bytes, bytes | None] | None] = queue.Queue()

    def processor() -> None:
        """キューから生バイト列を取り出し変換・混合する。"""
        while True:
            item = raw_queue.get()
            if item is None:
                return
            m_bytes, l_bytes = item
            try:
                m = _bytes_to_float32(m_bytes, mic_channels)
                nm = m.shape[0]
                if l_bytes is not None and loopback_channels > 0:
                    l_raw = _bytes_to_float32(l_bytes, loopback_channels)
                    # ループバックレートが異なる場合リサンプル
                    if loopback_rate != config.SAMPLE_RATE and l_raw.shape[0] != nm:
                        l_raw = _resample_to_target(l_raw, loopback_rate, nm)
                    l: np.ndarray | None = l_raw
                else:
                    l = None
                _process_raw_pair(
                    m,
                    l,
                    nm,
                    chunks=chunks,
                    chunk_lock=chunk_lock,
                    live_mono_chunk_callback=live_mono_chunk_callback,
                )
            except Exception as e:  # noqa: BLE001
                print(f"[audio] 処理スレッドでエラー: {e}", file=sys.stderr, flush=True)

    proc_thread = threading.Thread(target=processor, name="AudioProcessor", daemon=True)
    proc_thread.start()

    try:
        while not stop_event.is_set():
            try:
                m_bytes = mic_stream.read(
                    config.BLOCK_FRAMES, exception_on_overflow=False
                )
            except Exception as e:  # noqa: BLE001
                print(f"[audio] マイク読み取りエラー: {e}", file=sys.stderr, flush=True)
                time.sleep(0.005)
                continue

            l_bytes: bytes | None = None
            if loopback_stream is not None:
                try:
                    l_bytes = loopback_stream.read(
                        loopback_block_frames, exception_on_overflow=False
                    )
                except Exception:  # noqa: BLE001
                    l_bytes = None

            raw_queue.put((m_bytes, l_bytes))
    finally:
        raw_queue.put(None)
        proc_thread.join()


def run_recording_session(
    stop_event: threading.Event,
    chunks: list[np.ndarray],
    chunk_lock: threading.Lock,
    error_slot: list[str | None],
    *,
    live_mono_chunk_callback: Callable[[np.ndarray], None] | None = None,
) -> None:
    """バックグラウンドスレッドから呼び出し、録音セッションを実行する（PyAudioWPatch）。

    マイクと WASAPI ループバック（PC 再生音）を同時録音する。
    終了時に例外があれば ``error_slot[0]`` にメッセージ文字列を格納する。

    Args:
        stop_event: 録音ループを止めるイベント。
        chunks: 録音 PCM チャンクのリスト。
        chunk_lock: ``chunks`` 用のロック。
        error_slot: 長さ 1 のリスト。エラー時に ``[0]`` に文字列を書き込む。
        live_mono_chunk_callback: 任意。リアルタイム STT 用（``capture_loop`` 参照）。

    Returns:
        None
    """
    error_slot[0] = None
    pyaudio = _pyaudio()
    pa = pyaudio.PyAudio()

    mic_stream = None
    loopback_stream = None

    try:
        # --- マイクデバイス ---
        mic_info = pa.get_default_input_device_info()
        mic_index = int(mic_info["index"])
        mic_channels = min(int(mic_info["maxInputChannels"]), 2)

        # --- ループバックデバイス（PC 再生音）---
        loopback_info = get_loopback_device_info(pa)
        loopback_channels = 0
        loopback_rate = config.SAMPLE_RATE

        with chunk_lock:
            chunks.clear()

        # マイクストリームを開く
        mic_stream = pa.open(
            format=pyaudio.paInt16,
            channels=mic_channels,
            rate=config.SAMPLE_RATE,
            input=True,
            frames_per_buffer=config.BLOCK_FRAMES,
            input_device_index=mic_index,
        )

        # ループバックストリームを開く
        if loopback_info is not None:
            try:
                loopback_channels = int(loopback_info["maxInputChannels"])
                loopback_rate = int(
                    loopback_info.get("defaultSampleRate", config.SAMPLE_RATE)
                )
                loopback_block_frames = (
                    int(config.BLOCK_FRAMES * loopback_rate / config.SAMPLE_RATE)
                    if loopback_rate != config.SAMPLE_RATE
                    else config.BLOCK_FRAMES
                )
                loopback_stream = pa.open(
                    format=pyaudio.paInt16,
                    channels=loopback_channels,
                    rate=loopback_rate,
                    input=True,
                    frames_per_buffer=loopback_block_frames,
                    input_device_index=int(loopback_info["index"]),
                )
                print(
                    f"[audio] ループバック: {loopback_info.get('name', '?')} "
                    f"({loopback_channels}ch, {loopback_rate}Hz)",
                    flush=True,
                )
            except Exception as e:
                print(
                    f"[audio] ループバックストリームの開設に失敗しました: {e}",
                    file=sys.stderr,
                    flush=True,
                )
                loopback_stream = None
                loopback_channels = 0
        else:
            print(
                "[audio] ループバックデバイスが見つかりません。マイクのみで録音します。",
                file=sys.stderr,
                flush=True,
            )

        capture_loop(
            mic_stream,
            loopback_stream,
            mic_channels,
            loopback_channels,
            loopback_rate,
            stop_event,
            chunks,
            chunk_lock,
            live_mono_chunk_callback,
        )

    except Exception as e:
        print(f"[audio] 録音エラー: {e}", file=sys.stderr, flush=True)
        error_slot[0] = str(e)
    finally:
        if mic_stream is not None:
            try:
                mic_stream.stop_stream()
                mic_stream.close()
            except Exception:  # noqa: BLE001
                pass
        if loopback_stream is not None:
            try:
                loopback_stream.stop_stream()
                loopback_stream.close()
            except Exception:  # noqa: BLE001
                pass
        pa.terminate()


def write_wave_file(
    path: Path,
    *,
    chunks: list[np.ndarray],
    chunk_lock: threading.Lock,
) -> bool:
    """蓄積した PCM チャンクを連結して 1 つの WAV ファイルに書き出す。

    Args:
        path: 出力 WAV ファイルパス。
        chunks: 連結するチャンクのリスト（空なら書き込まない）。
        chunk_lock: ``chunks`` 用のロック。

    Returns:
        bool: データがありファイルを書いた場合 ``True``。データが無ければ ``False``。
    """
    with chunk_lock:
        if not chunks:
            return False
        data = np.concatenate(chunks)
        chunks.clear()
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(config.output_channel_count())
        wf.setsampwidth(config.SAMPLE_WIDTH)
        wf.setframerate(config.SAMPLE_RATE)
        wf.writeframes(data.tobytes())
    return True
