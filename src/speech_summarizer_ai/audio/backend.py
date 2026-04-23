"""マイク・ループバック録音と WAV 出力（SoundCard / WASAPI）。

録音スレッドは ``record()`` とキュー投入のみ。NumPy 処理は別スレッドに寄せ、WASAPI の
data discontinuity を抑える。停止時はセンチネルでキューを完走させる。
"""

from __future__ import annotations

import ctypes
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

_sc = None


def _soundcard():
    """SoundCard を遅延 import する。"""
    global _sc
    if _sc is None:
        import soundcard as sc

        _sc = sc
    return _sc


def _win32_com_init() -> bool:
    """録音スレッドで WASAPI を使う前に COM を初期化する。

    Returns:
        bool: ``CoInitialize`` に成功した場合 ``True``。不要または失敗時は ``False``。
    """
    if sys.platform != "win32":
        return False

    hr = ctypes.windll.ole32.CoInitialize(None)
    if hr == 0:
        return True
    if (hr + 2**32) == 0x80010106:  # RPC_E_CHANGED_MODE
        return False
    return False


def _win32_com_done(initialized: bool) -> None:
    """COM の初期化を対になる形で解除する。

    Args:
        initialized: ``_win32_com_init`` が ``True`` を返した場合に ``True``。

    Returns:
        None
    """
    if initialized and sys.platform == "win32":
        ctypes.windll.ole32.CoUninitialize()


def loopback_microphone() -> Any | None:
    """既定スピーカーに対応するループバック入力（PC 再生音）を返す。

    Returns:
        Any | None: ループバック ``Microphone``。見つからなければ ``None``。
    """
    sc = _soundcard()
    spk = sc.default_speaker()
    try:
        mic = sc.get_microphone(spk.id, include_loopback=True)
    except IndexError:
        return None
    if getattr(mic, "isloopback", False):
        return mic
    for m in sc.all_microphones(include_loopback=True):
        if m.isloopback and m.id == spk.id:
            return m
    return None


def recorder_channel_count(device: Any) -> int:
    """デバイスが報告するチャンネル数を返す（WASAPI では最低 2ch を確保）。

    Args:
        device: SoundCard のマイク／ループバックデバイス。

    Returns:
        int: 録音に使用するチャンネル数（少なくとも 2）。
    """
    ch = device.channels
    if isinstance(ch, int):
        return max(ch, 2)
    return max(len(ch), 2)


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
        m: マイク ``record()`` の生配列。
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
            if nl < nm:
                mono_l = np.concatenate([mono_l, np.zeros(nm - nl, dtype=np.float32)])
            elif nl > nm:
                mono_l = mono_l[:nm]
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
    mic_rec: Any,
    loop_rec: Any | None,
    stop_event: threading.Event,
    chunks: list[np.ndarray],
    chunk_lock: threading.Lock,
    live_mono_chunk_callback: Callable[[np.ndarray], None] | None = None,
) -> None:
    """マイク（と任意のループバック）から録音し、処理スレッド経由で ``chunks`` に積む。

    Args:
        mic_rec: マイク ``Recorder``。
        loop_rec: ループバック ``Recorder``。無ければ PC 音なし。
        stop_event: ループ終了用イベント。
        chunks: PCM チャンクのリスト。
        chunk_lock: ``chunks`` 用ロック。
        live_mono_chunk_callback: 任意。リアルタイム STT 用 int16 モノラル（**処理スレッド**から呼ぶ）。

    Returns:
        None
    """
    raw_queue: queue.Queue[tuple[np.ndarray, np.ndarray | None, int] | None] = (
        queue.Queue()
    )

    def processor() -> None:
        """キューから生フレームを取り出し ``_process_raw_pair`` に渡す。"""
        while True:
            item = raw_queue.get()
            if item is None:
                return
            m, l, nm = item
            try:
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
            m = mic_rec.record(numframes=config.BLOCK_FRAMES)
            nm = int(m.shape[0])
            if nm == 0:
                time.sleep(0.005)
                continue
            if loop_rec is not None:
                l = loop_rec.record(numframes=nm)
            else:
                l = None
            raw_queue.put((m, l, nm))
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
    """バックグラウンドスレッドから呼び出し、録音セッションを実行する。

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
    com_ok = _win32_com_init()
    try:
        sc = _soundcard()
        mic_dev = sc.default_microphone()
        loop_dev = loopback_microphone()
        mic_ch = recorder_channel_count(mic_dev)

        with chunk_lock:
            chunks.clear()

        if loop_dev is not None:
            loop_ch = recorder_channel_count(loop_dev)
            with (
                mic_dev.recorder(
                    samplerate=config.SAMPLE_RATE,
                    channels=mic_ch,
                    blocksize=config.BLOCK_FRAMES,
                ) as mic_rec,
                loop_dev.recorder(
                    samplerate=config.SAMPLE_RATE,
                    channels=loop_ch,
                    blocksize=config.BLOCK_FRAMES,
                ) as loop_rec,
            ):
                capture_loop(
                    mic_rec,
                    loop_rec,
                    stop_event,
                    chunks,
                    chunk_lock,
                    live_mono_chunk_callback,
                )
        else:
            print(
                "ループバックが見つかりません。PC 音なしでマイクのみ録音します。",
                file=sys.stderr,
            )
            with mic_dev.recorder(
                samplerate=config.SAMPLE_RATE,
                channels=mic_ch,
                blocksize=config.BLOCK_FRAMES,
            ) as mic_rec:
                capture_loop(
                    mic_rec,
                    None,
                    stop_event,
                    chunks,
                    chunk_lock,
                    live_mono_chunk_callback,
                )
    except Exception as e:
        print(f"録音エラー: {e}", file=sys.stderr)
        error_slot[0] = str(e)
    finally:
        _win32_com_done(com_ok)


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
