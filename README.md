# Video Stabilizer

光学フローを使って動画を手ブレ補正し、元の音声を保ったまま出力する CLI / GUI ツールです。

## 主な機能
- 光学フローによる手ブレ補正
- 平滑化半径・補正量の調整（CLI）
- ズーム＆クロップで黒縁を軽減
- 元動画の音声を合成
- GUI の進捗バーとドラッグ＆ドロップ入力

## 必要環境
- Python 3.9+
- OpenCV (`opencv-python`)
- NumPy (`numpy`)
- MoviePy (`moviepy`)
- tqdm (`tqdm`) ※CLI の進捗表示
- tkinter（通常は Python に同梱）
- TkinterDnD2 (`tkinterdnd2`) ※GUI のドラッグ＆ドロップ

## インストール例（pip）

```bash
pip install opencv-python numpy moviepy tqdm tkinterdnd2
```

## CLI の使い方

```bash
python video_stabilizer.py input.mp4 output.mp4
```

パラメータ指定（任意）:

```bash
python video_stabilizer.py input.mp4 output.mp4 --radius 5 --scaling_factor 1.0
```

## GUI の使い方

```bash
python video_stabilizer_GUI.py
```

入力／出力ファイルを選択（または入力欄にドラッグ＆ドロップ）して開始してください。

## 注意点
- 処理中に一時ファイル `temp_stabilized.mp4` を生成します。
- 出力は `libx264` / `aac` で書き出します。

## リポジトリ
https://github.com/amaterupc/video_stabilizer
