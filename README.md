# Speech Summarizer AI

Speech Summarizer AI アプリケーションです。

## レイアウト

- `src/speech_summarizer_ai/` — Python パッケージ本体（`settings` / `platform_utils` / `domain` / `data` / `audio` / `stt` / `llm` / `controllers` / `ui`）
- `resources/icons/` — アプリアイコン（`app.svg` / Windows 用 `app.ico` / `app.png`）。PyInstaller ビルド時に同梱され、EXE にも埋め込まれる
- `scripts/` — 開発用エントリーポイント（`run_dev.py` など）
- `database/` — 開発時の SQLite（実行ファイル版では exe と同じフォルダに `database/` が作成されます）
- `sessions/` — 録音セッション（同上）
- `models/` — STT 等のローカルモデル（同上）
- `tests/` — テスト
- `speech_summarizer_ai.spec` — Windows 向け PyInstaller 定義

## セットアップと起動

仮想環境を作成して依存関係を入れます。

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

または、パッケージを editable install すると `speech-summarizer-ai` コマンドが使えます。

```bash
pip install -e .
```

開発時はリポジトリルートから次のいずれかで起動できます。

```bash
python scripts/run_dev.py
```

editable install 後は次のコマンドでも起動できます。

```bash
speech-summarizer-ai
python -m speech_summarizer_ai
```

録音はプロジェクトルート直下の `sessions/<セッションID>/audio.wav` に保存されます（配布用 exe を使う場合は **exe と同じフォルダ** 直下の `sessions/`）。

## Windows 実行ファイルの作成（PyInstaller）

前提:

- Windows PC、上記のとおり `requirements.txt` を満たした仮想環境
- リポジトリルートで作業する

手順:

```bash
venv\Scripts\activate
pyinstaller --noconfirm speech_summarizer_ai.spec
```

クリーンビルドする場合:

```bash
pyinstaller --noconfirm --clean speech_summarizer_ai.spec
```
