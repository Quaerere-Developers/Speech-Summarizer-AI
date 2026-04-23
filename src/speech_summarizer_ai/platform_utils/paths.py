"""セッション・DB・モデル・LLM プローブ用パスの合成。"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from speech_summarizer_ai import settings as config


def new_session_id() -> str:
    """新しいセッション用の一意な ID 文字列を生成する。

    Returns:
        str: 現在時刻に基づく ``%Y%m%d_%H%M%S_%f`` 形式の ID。
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def sessions_root(project_root: Path) -> Path:
    """セッションルートディレクトリ（``sessions`` 等）のパスを返す。

    Args:
        project_root: プロジェクトルートディレクトリ。

    Returns:
        Path: ``project_root / SESSIONS_DIR_NAME``。
    """
    return project_root / config.SESSIONS_DIR_NAME


def session_directory(project_root: Path, session_id: str) -> Path:
    """指定セッション ID のディレクトリパスを返す。

    Args:
        project_root: プロジェクトルートディレクトリ。
        session_id: セッション ID。

    Returns:
        Path: セッション専用ディレクトリ。
    """
    return sessions_root(project_root) / session_id


def database_directory(project_root: Path) -> Path:
    """SQLite 用ディレクトリ（``database/``）のパスを返す。

    Args:
        project_root: プロジェクトルートディレクトリ。

    Returns:
        Path: ``project_root / DATABASE_DIR_NAME``。
    """
    return project_root / config.DATABASE_DIR_NAME


def database_path(project_root: Path) -> Path:
    """SQLite データベースファイルのフルパスを返す。

    Args:
        project_root: プロジェクトルートディレクトリ。

    Returns:
        Path: ``database_directory(project_root) / DATABASE_FILENAME``。
    """
    return database_directory(project_root) / config.DATABASE_FILENAME


def project_root() -> Path:
    """このパッケージからリポジトリルート（``src`` の親）を返す。

    Returns:
        Path: プロジェクトルート。
    """
    return Path(__file__).resolve().parents[3]


def models_directory(project_root: Path) -> Path:
    """``project_root / models`` を返す。

    Args:
        project_root: プロジェクトルート。

    Returns:
        Path: ``models`` ディレクトリ。
    """
    return project_root / config.MODELS_DIR_NAME


def stt_model_directory(project_root: Path, folder: str) -> Path:
    """``models/stt/<folder>`` を返す。

    Args:
        project_root: プロジェクトルート。
        folder: STT モデル用サブフォルダ名。

    Returns:
        Path: STT モデルディレクトリ。
    """
    return models_directory(project_root) / config.MODELS_STT_SUBDIR / folder


def foundry_llm_cache_directory(project_root: Path) -> Path:
    """Foundry の ``model_cache_dir``（``models/llm``）を返す。

    Args:
        project_root: プロジェクトルート。

    Returns:
        Path: LLM キャッシュディレクトリ。
    """
    return models_directory(project_root) / config.MODELS_LLM_SUBDIR


_LLM_PROBE_OK_FILENAME = ".speech_summarizer_llm_probe_ok"


def foundry_llm_probe_marker_path(project_root: Path) -> Path:
    """LLM ロード成功マーカー ``models/llm/.speech_summarizer_llm_probe_ok`` のパス。

    Args:
        project_root: プロジェクトルート。

    Returns:
        Path: マーカーファイル。
    """
    return foundry_llm_cache_directory(project_root) / _LLM_PROBE_OK_FILENAME


def _read_probe_marker_lines(project_root: Path) -> list[str]:
    p = foundry_llm_probe_marker_path(project_root)
    if not p.is_file():
        return []
    try:
        return [
            ln.strip()
            for ln in p.read_text(encoding="utf-8").splitlines()
            if ln.strip()
        ]
    except OSError:
        return []


def llm_probe_marker_matches(project_root: Path, model_alias: str) -> bool:
    """マーカー 1 行目が ``model_alias`` と一致するか。

    Args:
        project_root: プロジェクトルート。
        model_alias: 設定の LLM 別名。

    Returns:
        bool: 一致すれば True。
    """
    lines = _read_probe_marker_lines(project_root)
    return bool(lines) and lines[0] == model_alias.strip()


def read_llm_resolved_id(project_root: Path, model_alias: str) -> str | None:
    """マーカー 2 行目の解決済み完全 ID を返す。

    Args:
        project_root: プロジェクトルート。
        model_alias: 1 行目と突き合わせる別名。

    Returns:
        str | None: 有効なら完全 ID。無ければ ``None``。
    """
    lines = _read_probe_marker_lines(project_root)
    if len(lines) < 2 or lines[0] != model_alias.strip():
        return None
    return lines[1] or None


def write_llm_probe_marker(
    project_root: Path,
    model_alias: str,
    resolved_id: str | None = None,
) -> None:
    """ロード成功後にマーカーを書く（1 行目: 別名、2 行目: 任意で完全 ID）。

    Args:
        project_root: プロジェクトルート。
        model_alias: カタログ別名。
        resolved_id: SDK が解決した完全 ID。省略可。

    Returns:
        None
    """
    root = foundry_llm_cache_directory(project_root)
    root.mkdir(parents=True, exist_ok=True)
    body = model_alias.strip() + "\n"
    if resolved_id:
        body += resolved_id.strip() + "\n"
    foundry_llm_probe_marker_path(project_root).write_text(body, encoding="utf-8")


def clear_llm_probe_marker(project_root: Path) -> None:
    """マーカーを削除し次回起動で再検証させる。

    Args:
        project_root: プロジェクトルート。

    Returns:
        None
    """
    p = foundry_llm_probe_marker_path(project_root)
    try:
        if p.is_file():
            p.unlink()
    except OSError:
        pass


def foundry_llm_model_weights_present(project_root: Path, model_alias: str) -> bool:
    """``models/llm`` 配下に、設定エイリアスに対応する ONNX 重みがあるか（起動時の存在チェック用）。

    Foundry が展開するパス（例: ``.../qwen2.5-0.5b-instruct-.../v2/model.onnx``）を
    ``model_alias`` の部分一致で判定する。音声認識の ``model.bin`` チェックに相当。

    Args:
        project_root: リポジトリルート。
        model_alias: ``config.FOUNDRY_LLM_MODEL_ALIAS`` などカタログのエイリアス。

    Returns:
        bool: エイリアスに一致するパスに ``model.onnx`` が存在し、かつ
        単体で十分大きいか、または ``model.onnx.data`` に重みがある場合 ``True``。
    """
    root = foundry_llm_cache_directory(project_root)
    if not root.is_dir():
        return False
    token = model_alias.strip()
    if not token:
        return False
    min_bytes = 512 * 1024
    for p in root.rglob("model.onnx"):
        try:
            if token not in p.as_posix():
                continue
            if not p.is_file():
                continue
            sz = p.stat().st_size
            data = p.with_name(p.name + ".data")
            data_sz = data.stat().st_size if data.is_file() else 0
            if sz >= min_bytes or data_sz >= min_bytes:
                return True
        except OSError:
            continue
    return False


def foundry_llm_resolved_weights_present(project_root: Path, resolved_id: str) -> bool:
    """``models/llm`` 配下に、マーカー 2 行目の解決済み ID に対応する ONNX があるか。

    カタログ ID に ``:`` が付く場合、フォルダ名側には含まれないことが多いため、
    ``:`` より前のプレフィックスでもパス一致を見る。

    Args:
        project_root: リポジトリルート。
        resolved_id: マーカーに保存した解決済みモデル ID。

    Returns:
        bool: 該当パスに実体のある ``model.onnx``（または十分な ``.data``）があれば ``True``。
    """
    rid = resolved_id.strip()
    if not rid:
        return False
    tokens = {rid}
    if ":" in rid:
        tokens.add(rid.split(":", 1)[0])
    root = foundry_llm_cache_directory(project_root)
    if not root.is_dir():
        return False
    min_bytes = 512 * 1024
    for p in root.rglob("model.onnx"):
        try:
            pos = p.as_posix()
            if not any(t in pos for t in tokens):
                continue
            if not p.is_file():
                continue
            sz = p.stat().st_size
            data = p.with_name(p.name + ".data")
            data_sz = data.stat().st_size if data.is_file() else 0
            if sz >= min_bytes or data_sz >= min_bytes:
                return True
        except OSError:
            continue
    return False


def new_session_audio_path(project_root: Path) -> tuple[Path, Path]:
    """新しいセッション用ディレクトリと、その中の音声ファイルパスを返す。

    Args:
        project_root: プロジェクトルートディレクトリ。

    Returns:
        tuple[Path, Path]: ``(セッションディレクトリ, 音声 WAV ファイルパス)`` のタプル。
    """
    sid = new_session_id()
    sdir = session_directory(project_root, sid)
    return sdir, sdir / config.SESSION_AUDIO_FILENAME
