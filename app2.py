from __future__ import annotations

import csv
import os
import math
import queue
import shutil
import subprocess
import sys
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable

import cv2
import numpy as np

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk, ImageDraw, ImageFont


def imread_unicode(image_path, flags=None):
    """日本語などのUnicode文字を含むパスから画像を読み込む。"""
    if flags is None:
        flags = cv2.IMREAD_COLOR

    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(f"画像が見つかりません: {path}")

    try:
        file_data = np.fromfile(str(path), dtype=np.uint8)
        image = cv2.imdecode(file_data, flags)
    except OSError as error:
        raise OSError(f"画像ファイルを読み込めません: {path}") from error

    if image is None:
        raise ValueError(
            f"画像のデコードに失敗しました: {path}\n"
            "画像形式が非対応、またはファイルが破損している可能性があります。"
        )

    return image


def imwrite_unicode(output_path, image, params=None):
    """日本語などのUnicode文字を含むパスへ画像を保存する。"""
    path = Path(output_path)
    if not path.suffix:
        raise ValueError(
            f"保存先に拡張子がありません: {path}\n"
            ".png や .jpg を指定してください。"
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    encode_params = params if params is not None else []

    success, encoded_image = cv2.imencode(path.suffix, image, encode_params)
    if not success:
        raise ValueError(f"画像のエンコードに失敗しました: {path}")

    try:
        encoded_image.tofile(str(path))
    except OSError as error:
        raise OSError(f"画像を保存できません: {path}") from error


def write_detections_csv(output_path, detections):
    """検出結果をCSVファイルへ保存する。"""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with path.open("w", encoding="utf-8-sig", newline="") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow([
                "検出順",
                "中心X",
                "中心Y",
                "外接矩形X",
                "外接矩形Y",
                "幅",
                "高さ",
                "輪郭面積",
            ])

            for index, detection in enumerate(detections, start=1):
                center_x, center_y = detection["center"]
                x, y, width, height = detection["bbox"]
                area = detection["area"]
                writer.writerow([
                    index,
                    f"{center_x:.1f}",
                    f"{center_y:.1f}",
                    x,
                    y,
                    width,
                    height,
                    f"{area:.1f}",
                ])
    except OSError as error:
        raise OSError(f"CSVファイルを保存できません: {path}") from error


def detect_red_number_positions(
    image_path,
    min_area=100.0,
    min_aspect_ratio=0.4,
    max_aspect_ratio=2.5,
):
    """赤色の番号マークを抽出し、外接矩形と中心座標を取得する。"""
    image = imread_unicode(image_path)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # lower_red_1 = np.array([0, 80, 80], dtype=np.uint8)
    # upper_red_1 = np.array([10, 255, 255], dtype=np.uint8)
    # lower_red_2 = np.array([170, 80, 80], dtype=np.uint8)
    # upper_red_2 = np.array([179, 255, 255], dtype=np.uint8)

    lower_red_1 = np.array([0, 40, 40], dtype=np.uint8)
    upper_red_1 = np.array([15, 255, 255], dtype=np.uint8)
    lower_red_2 = np.array([165, 40, 40], dtype=np.uint8)
    upper_red_2 = np.array([179, 255, 255], dtype=np.uint8)

    mask_1 = cv2.inRange(hsv, lower_red_1, upper_red_1)
    mask_2 = cv2.inRange(hsv, lower_red_2, upper_red_2)
    red_mask = cv2.bitwise_or(mask_1, mask_2)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    contours, _ = cv2.findContours(
        red_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    detections = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue

        x, y, width, height = cv2.boundingRect(contour)
        if height == 0:
            continue

        aspect_ratio = width / float(height)
        if not min_aspect_ratio <= aspect_ratio <= max_aspect_ratio:
            continue

        center_x = x + width / 2.0
        center_y = y + height / 2.0
        detections.append({
            "center": (center_x, center_y),
            "bbox": (x, y, width, height),
            "area": area,
        })

    detections.sort(key=lambda item: (item["center"][1], item["center"][0]))
    return image, red_mask, detections


def draw_detection_result(image, detections):
    """検出結果を画像上に描画する。"""
    result = image.copy()
    for index, detection in enumerate(detections, start=1):
        x, y, width, height = detection["bbox"]

        cv2.rectangle(
            result,
            (x, y),
            (x + width, y + height),
            (255, 0, 0),
            2,
        )
        cv2.putText(
            result,
            str(index),
            (x, max(20, y - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 0, 0),
            2,
            cv2.LINE_AA,
        )

    return result


# =========================================================
# 検査項目番号トリミング・マトリクス画像作成処理
# crop_number.py の処理をGUI用に組み込み
# =========================================================
REQUIRED_COLUMNS = ("中心X", "中心Y", "幅", "高さ")


def validate_columns(fieldnames: Iterable[str] | None) -> None:
    columns = set(fieldnames or [])
    missing = [name for name in REQUIRED_COLUMNS if name not in columns]
    if missing:
        raise ValueError(
            f"CSVに必要な列がありません: {', '.join(missing)}。"
            f" 必要な列: {', '.join(REQUIRED_COLUMNS)}"
        )


def read_number(row: Dict[str, str], column: str, row_number: int) -> float:
    raw = row.get(column, "").strip()
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(
            f"CSV {row_number}行目の「{column}」が数値ではありません: {raw!r}"
        ) from exc

    if not math.isfinite(value):
        raise ValueError(f"CSV {row_number}行目の「{column}」が有限値ではありません。")

    return value


def read_image_unicode(image_path: Path) -> np.ndarray:
    """日本語を含むパスでも読み込めるように画像を読み込む。"""
    data = np.fromfile(str(image_path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise OSError(f"画像を読み込めませんでした: {image_path}")
    return image


def write_image_unicode(output_path: Path, image: np.ndarray, quality: int = 95) -> None:
    """日本語を含むパスでも保存できるように画像を書き出す。"""
    suffix = output_path.suffix.lower()
    params: list[int] = []

    if suffix in {".jpg", ".jpeg"}:
        params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    elif suffix == ".webp":
        params = [cv2.IMWRITE_WEBP_QUALITY, quality]

    success, encoded = cv2.imencode(suffix, image, params)
    if not success:
        raise OSError(f"画像のエンコードに失敗しました: {output_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    encoded.tofile(str(output_path))


def clipped_crop_box(
    center_x: float,
    center_y: float,
    crop_width: float,
    crop_height: float,
    image_width: int,
    image_height: int,
) -> tuple[int, int, int, int]:
    """OpenCVスライス用の(x1, y1, x2, y2)を画像範囲内で返す。"""
    x1 = math.floor(center_x - crop_width / 2.0)
    y1 = math.floor(center_y - crop_height / 2.0)
    x2 = math.ceil(center_x + crop_width / 2.0)
    y2 = math.ceil(center_y + crop_height / 2.0)

    x1 = max(0, min(image_width, x1))
    y1 = max(0, min(image_height, y1))
    x2 = max(0, min(image_width, x2))
    y2 = max(0, min(image_height, y2))

    if x2 <= x1 or y2 <= y1:
        raise ValueError(
            "切り出し範囲が画像内にありません "
            f"(x1={x1}, y1={y1}, x2={x2}, y2={y2})"
        )

    return x1, y1, x2, y2


def crop_objects(
    image_path: Path,
    csv_path: Path,
    output_dir: Path,
    width_offset: float = 0.0,
    height_offset: float = 0.0,
    output_prefix: str = "object",
    output_format: str = "png",
    csv_encoding: str = "utf-8-sig",
) -> None:
    """CSVに記録された各オブジェクトをOpenCVで切り出して保存する。"""
    output_format = output_format.lower().lstrip(".")

    if width_offset < 0 or height_offset < 0:
        raise ValueError("幅・高さのオフセットには0以上の値を指定してください。")
    if output_format not in {"png", "jpg", "jpeg", "webp"}:
        raise ValueError("保存形式は png、jpg、jpeg、webp のいずれかを指定してください。")
    if not image_path.is_file():
        raise FileNotFoundError(f"画像ファイルが見つかりません: {image_path}")
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSVファイルが見つかりません: {csv_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    image = read_image_unicode(image_path)
    image_height, image_width = image.shape[:2]

    saved_count = 0
    skipped_count = 0

    with csv_path.open("r", encoding=csv_encoding, newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        validate_columns(reader.fieldnames)

        for data_index, row in enumerate(reader, start=1):
            csv_row_number = data_index + 1

            try:
                center_x = read_number(row, "中心X", csv_row_number)
                center_y = read_number(row, "中心Y", csv_row_number)
                detected_width = read_number(row, "幅", csv_row_number)
                detected_height = read_number(row, "高さ", csv_row_number)

                crop_width = detected_width + width_offset
                crop_height = detected_height + height_offset

                if crop_width <= 0 or crop_height <= 0:
                    raise ValueError("切り出し後の幅・高さは0より大きい必要があります。")

                x1, y1, x2, y2 = clipped_crop_box(
                    center_x=center_x,
                    center_y=center_y,
                    crop_width=crop_width,
                    crop_height=crop_height,
                    image_width=image_width,
                    image_height=image_height,
                )

                cropped = image[y1:y2, x1:x2].copy()

                detection_no = row.get("検出順", "").strip()
                try:
                    file_no = int(float(detection_no)) if detection_no else data_index
                except ValueError:
                    file_no = data_index

                output_path = output_dir / f"{output_prefix}_{file_no:04d}.{output_format}"
                write_image_unicode(output_path, cropped)

                saved_count += 1
                print(
                    f"保存: {output_path} "
                    f"範囲=({x1}, {y1}, {x2}, {y2}) "
                    f"サイズ={cropped.shape[1]}x{cropped.shape[0]}"
                )

            except (ValueError, OSError) as exc:
                skipped_count += 1
                print(f"スキップ: CSV {csv_row_number}行目: {exc}")

    print(f"完了: {saved_count}件保存、{skipped_count}件スキップ")


def natural_sort_key(path: Path) -> list[object]:
    """ファイル名中の数字を数値として扱う自然順ソート用キー。"""
    import re

    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", path.name)
    ]


def to_bgr(image: np.ndarray) -> np.ndarray:
    """グレースケール・BGRA画像をBGR画像へ統一する。"""
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.ndim == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    if image.ndim == 3 and image.shape[2] == 3:
        return image
    raise ValueError(f"未対応の画像形式です: shape={image.shape}")


def create_image_matrices(
    input_dir: Path,
    output_dir: Path,
    max_rows: int,
    max_cols: int,
    output_prefix: str = "matrix",
    output_format: str = "png",
    cell_width: int | None = None,
    cell_height: int | None = None,
    margin: int = 10,
    background_value: int = 255,
) -> list[Path]:
    """
    input_dir内の画像をファイル名順に並べ、複数のマトリクス画像として保存する。
    """
    if max_rows <= 0 or max_cols <= 0:
        raise ValueError("max_rows と max_cols は1以上を指定してください。")
    if margin < 0:
        raise ValueError("margin は0以上を指定してください。")
    if not 0 <= background_value <= 255:
        raise ValueError("background_value は0～255を指定してください。")

    output_format = output_format.lower().lstrip(".")
    if output_format not in {"png", "jpg", "jpeg", "webp"}:
        raise ValueError("保存形式は png、jpg、jpeg、webp のいずれかを指定してください。")
    if not input_dir.is_dir():
        raise FileNotFoundError(f"入力フォルダが見つかりません: {input_dir}")

    extensions = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
    image_paths = sorted(
        [p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in extensions],
        key=natural_sort_key,
    )
    if not image_paths:
        raise FileNotFoundError(f"並べる画像がありません: {input_dir}")

    images: list[tuple[Path, np.ndarray]] = []
    for path in image_paths:
        image = to_bgr(read_image_unicode(path))
        images.append((path, image))

    if cell_width is None:
        cell_width = max(image.shape[1] for _, image in images)
    if cell_height is None:
        cell_height = max(image.shape[0] for _, image in images)
    if cell_width <= 0 or cell_height <= 0:
        raise ValueError("cell_width と cell_height は1以上を指定してください。")

    output_dir.mkdir(parents=True, exist_ok=True)
    per_page = max_rows * max_cols
    saved_paths: list[Path] = []

    for page_index, start in enumerate(range(0, len(images), per_page), start=1):
        page_items = images[start:start + per_page]
        used_rows = math.ceil(len(page_items) / max_cols)

        canvas_width = margin + max_cols * (cell_width + margin)
        canvas_height = margin + used_rows * (cell_height + margin)
        canvas = np.full(
            (canvas_height, canvas_width, 3),
            background_value,
            dtype=np.uint8,
        )

        for index, (source_path, image) in enumerate(page_items):
            row = index // max_cols
            col = index % max_cols

            scale = min(cell_width / image.shape[1], cell_height / image.shape[0], 1.0)
            resized_width = max(1, int(round(image.shape[1] * scale)))
            resized_height = max(1, int(round(image.shape[0] * scale)))
            if resized_width != image.shape[1] or resized_height != image.shape[0]:
                image = cv2.resize(
                    image,
                    (resized_width, resized_height),
                    interpolation=cv2.INTER_AREA,
                )

            cell_x = margin + col * (cell_width + margin)
            cell_y = margin + row * (cell_height + margin)
            x = cell_x + (cell_width - resized_width) // 2
            y = cell_y + (cell_height - resized_height) // 2
            canvas[y:y + resized_height, x:x + resized_width] = image

            print(
                f"配置: page={page_index}, row={row + 1}, col={col + 1}, "
                f"file={source_path.name}"
            )

        output_path = output_dir / f"{output_prefix}_{page_index:03d}.{output_format}"
        write_image_unicode(output_path, canvas)
        saved_paths.append(output_path)
        print(f"マトリクス画像を保存: {output_path}")

    return saved_paths


def get_gui_base_directory():
    """GUIのpyファイルと同じディレクトリを返す。"""
    try:
        return Path(__file__).resolve().parent
    except NameError:
        return Path.cwd()


def process_number_position_image(image_path):
    """検査項目番号つき図面画像を処理してCSV・トリミング画像・マトリクス画像を保存する。"""
    base_directory = get_gui_base_directory()

    output_directory = base_directory / "output"
    crop_output_directory = base_directory / "crop_img"
    matrix_output_directory = base_directory / "matrix_img"

    output_directory.mkdir(parents=True, exist_ok=True)
    crop_output_directory.mkdir(parents=True, exist_ok=True)
    matrix_output_directory.mkdir(parents=True, exist_ok=True)

    for target_dir in (crop_output_directory, matrix_output_directory):
        for old_file in target_dir.iterdir():
            if old_file.is_file():
                old_file.unlink()
            elif old_file.is_dir():
                shutil.rmtree(old_file)

    mask_output_path = output_directory / "赤色抽出マスク.png"
    result_output_path = output_directory / "赤番号検出結果.png"
    csv_output_path = output_directory / "赤番号検出結果.csv"

    image, red_mask, detections = detect_red_number_positions(
        image_path=image_path,
        min_area=100.0,
        min_aspect_ratio=0.4,
        max_aspect_ratio=2.5,
    )
    result_image = draw_detection_result(image, detections)

    imwrite_unicode(mask_output_path, red_mask)
    imwrite_unicode(result_output_path, result_image)
    write_detections_csv(csv_output_path, detections)

    matrix_paths = []
    if detections:
        crop_objects(
            image_path=Path(image_path),
            csv_path=csv_output_path,
            output_dir=crop_output_directory,
            width_offset=0,
            height_offset=0,
            output_prefix="object",
            output_format="png",
            csv_encoding="utf-8-sig",
        )

        matrix_paths = create_image_matrices(
            input_dir=crop_output_directory,
            output_dir=matrix_output_directory,
            max_rows=6,
            max_cols=6,
            output_prefix="matrix",
            output_format="png",
            cell_width=None,
            cell_height=None,
            margin=10,
            background_value=255,
        )

    if matrix_paths:
        copy_image_to_clipboard_windows(matrix_paths[0])

    return {
        "count": len(detections),
        "output_directory": output_directory,
        "mask_output_path": mask_output_path,
        "result_output_path": result_output_path,
        "csv_output_path": csv_output_path,
        "crop_output_directory": crop_output_directory,
        "matrix_output_directory": matrix_output_directory,
        "matrix_paths": matrix_paths,
    }


def copy_image_to_clipboard_windows(image_path):
    """Windowsでは画像をクリップボードへコピーする。失敗してもGUI処理は継続する。"""
    try:
        from io import BytesIO
        import win32clipboard
    except ImportError:
        return

    image = Image.open(image_path)
    output = BytesIO()

    image.convert("RGB").save(output, "BMP")
    data = output.getvalue()[14:]
    output.close()

    try:
        win32clipboard.OpenClipboard()
        try:
            win32clipboard.EmptyClipboard()
            win32clipboard.SetClipboardData(win32clipboard.CF_DIB, data)
        finally:
            win32clipboard.CloseClipboard()
    except Exception:
        return




# =========================================================
# 番号描画設定（app.pyの番号配置処理）
# =========================================================
CSV_NO_COLUMN = "読み取り結果"
CSV_X_COLUMN = "中心X"
CSV_Y_COLUMN = "中心Y"
CSV_SIZE_COLUMN = "サイズ"
CSV_COLUMN_NAME_FLEXIBLE_MATCH = True
DEFAULT_MARKER_SIZE = 28
MIN_MARKER_SIZE = 8
MAX_MARKER_SIZE = 200

class ZoomableImageViewer(tk.Frame):
    """画像上に編集可能な検査項目番号オブジェクトを配置するビューア。

    操作:
    - 図面を開く: 背景画像を読み込み
    - CSVを開く: CSV_NO_COLUMN / CSV_X_COLUMN / CSV_Y_COLUMN から番号オブジェクトを配置
    - マウスホイール: 拡大/縮小
    - 背景を左ドラッグ: 表示位置移動
    - 番号オブジェクトを左ドラッグ: 番号位置を調整
    - 番号オブジェクトをダブルクリック: 検査項目番号を変更
    """

    def __init__(
        self,
        master,
        width=800,
        height=500,
        bg="#B4B4B4",
        empty_text="図面表示エリア",
    ):
        super().__init__(master, bg=bg)

        self.width = width
        self.height = height
        self.bg = bg
        self.empty_text = empty_text
        self.scale = 1.0
        self.offset_x = width // 2
        self.offset_y = height // 2

        self._pil_image = None
        self._tk_image = None
        self._pan_start = None
        self._object_drag = None
        self.selected_annotation_ids = set()
        self.selection_status_callback = None

        self.display_width = None
        self.display_height = None

        # annotation_id -> {"no": str, "x": float, "y": float, "size": float}
        # x, y は「表示用画像サイズ」に対する画像座標(px)
        # size は倍率1.0時の赤丸直径(px)
        self.annotations = {}
        self._next_annotation_id = 1
        self.show_annotations = True

        self.canvas = tk.Canvas(
            self,
            width=width,
            height=height,
            bg=bg,
            highlightthickness=0,
            bd=0,
        )
        self.canvas.pack(fill="both", expand=True)

        self.canvas.bind("<Configure>", self._on_canvas_configure)
        self.canvas.bind("<MouseWheel>", self._on_mousewheel)      # Windows / macOS
        self.canvas.bind("<Button-4>", self._on_mousewheel_linux)  # Linux up
        self.canvas.bind("<Button-5>", self._on_mousewheel_linux)  # Linux down

        # 背景ドラッグは専用タグにだけ紐づけ、番号オブジェクト操作と競合させない
        self.canvas.bind("<ButtonPress-1>", self._on_pan_start)
        self.canvas.bind("<B1-Motion>", self._on_pan_move)
        self.canvas.bind("<ButtonRelease-1>", self._on_pan_end)

    # =========================================================
    # 画像・CSV
    # =========================================================
    def set_image(self, image_source):
        """画像を差し替える。image_source は画像ファイルパスを想定。"""
        self._pil_image = Image.open(image_source)
        self.display_width, self.display_height = self._pil_image.size
        self.reset_view()
        self.redraw()

    def set_display_size(self, width, height, scale_annotations=True):
        """表示用の画像サイズを設定する。元画像ファイル自体は変更しない。

        scale_annotations=True の場合は、既存番号の画像座標も同じ倍率で変換する。
        画像切替時に保存済みの表示サイズを復元するだけの場合は、番号座標を
        変換してはいけないため False を指定する。
        """
        if self._pil_image is None:
            return

        old_w = self.display_width or self._pil_image.size[0]
        old_h = self.display_height or self._pil_image.size[1]

        new_w = max(1, int(width))
        new_h = max(1, int(height))

        # 既に配置済みの番号は、画像サイズ変更後も相対位置が保たれるよう変換
        if scale_annotations and old_w > 0 and old_h > 0:
            sx = new_w / old_w
            sy = new_h / old_h
            for ann in self.annotations.values():
                ann["x"] *= sx
                ann["y"] *= sy

        self.display_width = new_w
        self.display_height = new_h
        self.reset_view()
        self.redraw()

    def get_display_size(self):
        """現在の表示用サイズを返す。画像未読込の場合は None を返す。"""
        if self._pil_image is None:
            return None
        return self.display_width, self.display_height

    def load_annotations_from_csv(self, csv_path):
        """CSV列名設定で指定した列から番号オブジェクトを作成する。"""
        if self._pil_image is None:
            messagebox.showinfo("番号描画", "先に図面画像を開いてください。")
            return

        rows = self._read_csv_rows(csv_path)
        if not rows:
            messagebox.showwarning("番号描画", "CSVから配置できるデータが見つかりませんでした。")
            return

        self.clear_annotations()

        for row in rows:
            ann_id = self._next_annotation_id
            self._next_annotation_id += 1
            self.annotations[ann_id] = {
                "no": row["no"],
                "x": row["x"],
                "y": row["y"],
                "size": row.get("size", DEFAULT_MARKER_SIZE),
            }

        self.redraw()
        messagebox.showinfo("番号描画", f"{len(rows)}件の検査項目番号を配置しました。")

    def _read_csv_rows(self, csv_path):
        """UTF-8/Shift_JIS系のCSVを読み込む。列名はファイル上部の定数で指定する。"""
        encodings = ("utf-8-sig", "cp932", "shift_jis", "utf-8")
        last_error = None

        for enc in encodings:
            try:
                with open(csv_path, newline="", encoding=enc) as f:
                    reader = csv.DictReader(f)
                    if reader.fieldnames is None:
                        continue

                    no_col = self._find_first_csv_column(
                        reader.fieldnames,
                        ("読み取り結果", "検査項目No.", "検査項目No", "番号", "No.", "No", "検出順"),
                    )
                    x_col = self._find_first_csv_column(
                        reader.fieldnames, ("中心X", "X", "座標X")
                    )
                    y_col = self._find_first_csv_column(
                        reader.fieldnames, ("中心Y", "Y", "座標Y")
                    )
                    size_col = self._find_first_csv_column(
                        reader.fieldnames, ("サイズ", "大きさ", "直径")
                    )

                    if not (no_col and x_col and y_col):
                        raise ValueError(
                            "CSVに必要な列が見つかりません。\n"
                            "必要な列: 番号（読み取り結果など）, 中心X, 中心Y\n"
                            f"CSV内の列: {', '.join(reader.fieldnames)}"
                        )

                    rows = []
                    for line_no, row in enumerate(reader, start=2):
                        no = str(row.get(no_col, "")).strip()
                        x_text = str(row.get(x_col, "")).strip()
                        y_text = str(row.get(y_col, "")).strip()
                        size_text = str(row.get(size_col, "")).strip() if size_col else ""

                        if not no and not x_text and not y_text:
                            continue

                        try:
                            x = float(x_text)
                            y = float(y_text)
                        except ValueError:
                            raise ValueError(
                                f"{line_no}行目のX/Y座標が数値ではありません。"
                            )

                        size = DEFAULT_MARKER_SIZE
                        if size_text:
                            try:
                                size = float(size_text)
                            except ValueError:
                                raise ValueError(f"{line_no}行目のサイズが数値ではありません。")
                            size = max(MIN_MARKER_SIZE, min(MAX_MARKER_SIZE, size))

                        rows.append({"no": no, "x": x, "y": y, "size": size})

                    return rows

            except UnicodeDecodeError as e:
                last_error = e
                continue
            except Exception as e:
                messagebox.showerror("CSV読込エラー", str(e))
                return []

        messagebox.showerror("CSV読込エラー", f"CSVを読み込めませんでした。\n{last_error}")
        return []

    @staticmethod
    def _normalize_column_name(name):
        return str(name).strip().lower().replace(" ", "")

    @classmethod
    def _find_csv_column(cls, fieldnames, target_column):
        """CSV内から指定列名に対応する実際の列名を探す。"""
        if target_column in fieldnames:
            return target_column

        if not CSV_COLUMN_NAME_FLEXIBLE_MATCH:
            return None

        normalized_target = cls._normalize_column_name(target_column)
        for fieldname in fieldnames:
            if cls._normalize_column_name(fieldname) == normalized_target:
                return fieldname

        return None

    @classmethod
    def _find_first_csv_column(cls, fieldnames, candidates):
        for candidate in candidates:
            found = cls._find_csv_column(fieldnames, candidate)
            if found:
                return found
        return None

    def clear_annotations(self):
        self.annotations.clear()
        self.selected_annotation_ids.clear()
        self._next_annotation_id = 1
        self._notify_selection_status()
        self.redraw()

    def set_selection_status_callback(self, callback):
        self.selection_status_callback = callback
        self._notify_selection_status()

    def _notify_selection_status(self):
        if self.selection_status_callback is not None:
            self.selection_status_callback(len(self.selected_annotation_ids), len(self.annotations))

    def select_all_annotations(self):
        """配置済みの番号オブジェクトを全選択する。"""
        if not self.annotations:
            messagebox.showinfo(
                "全選択",
                "選択する番号がありません。",
                parent=self.winfo_toplevel(),
            )
            return

        self.selected_annotation_ids = set(self.annotations.keys())
        self._notify_selection_status()
        messagebox.showinfo(
            "全選択",
            f"{len(self.selected_annotation_ids)}件の番号を選択しました。\n選択した番号をドラッグすると、全番号を同じ方向へ移動できます。",
            parent=self.winfo_toplevel(),
        )

    def deselect_all_annotations(self):
        """番号オブジェクトの全選択状態を解除する。"""
        self.selected_annotation_ids.clear()
        self._notify_selection_status()

    def set_all_annotation_sizes(self, size):
        """配置済みの全番号オブジェクトの赤丸サイズを一括変更する。"""
        new_size = float(size)
        if not (MIN_MARKER_SIZE <= new_size <= MAX_MARKER_SIZE):
            raise ValueError(
                f"赤丸サイズは {MIN_MARKER_SIZE}〜{MAX_MARKER_SIZE} の範囲で入力してください。"
            )

        for ann in self.annotations.values():
            ann["size"] = new_size

        self.redraw()

    def get_common_annotation_size(self):
        """一括サイズ設定画面の初期値を返す。"""
        if not self.annotations:
            return DEFAULT_MARKER_SIZE

        sizes = [
            float(ann.get("size", DEFAULT_MARKER_SIZE))
            for ann in self.annotations.values()
        ]
        first_size = sizes[0]

        # すべて同じサイズならその値、異なる場合は平均値を初期値にする。
        if all(abs(size - first_size) < 0.001 for size in sizes):
            return first_size
        return sum(sizes) / len(sizes)

    def set_annotation_visibility(self, visible):
        """表示画像に応じて番号オブジェクトの表示/非表示を切り替える。"""
        self.show_annotations = bool(visible)
        self.redraw()

    def redraw_image(self):
        """app2.pyの表示状態管理との互換用。"""
        self.redraw()

    def redraw_rectangle(self):
        """app2.pyの既存呼び出しとの互換用。"""
        pass

    # =========================================================
    # 座標変換
    # =========================================================
    def image_to_canvas(self, image_x, image_y):
        """画像座標(px) -> Canvas座標"""
        base_w = self.display_width or 1
        base_h = self.display_height or 1
        left = self.offset_x - (base_w * self.scale) / 2
        top = self.offset_y - (base_h * self.scale) / 2
        return left + image_x * self.scale, top + image_y * self.scale

    def canvas_to_image(self, canvas_x, canvas_y):
        """Canvas座標 -> 画像座標(px)"""
        base_w = self.display_width or 1
        base_h = self.display_height or 1
        left = self.offset_x - (base_w * self.scale) / 2
        top = self.offset_y - (base_h * self.scale) / 2
        return (canvas_x - left) / self.scale, (canvas_y - top) / self.scale

    # =========================================================
    # 描画
    # =========================================================
    def reset_view(self):
        """倍率と表示位置を初期状態へ戻す。"""
        self.scale = 1.0
        self.offset_x = max(1, self.canvas.winfo_width()) // 2
        self.offset_y = max(1, self.canvas.winfo_height()) // 2

    def redraw(self):
        self.canvas.delete("all")
        if self._pil_image is None:
            canvas_w = max(1, self.canvas.winfo_width())
            canvas_h = max(1, self.canvas.winfo_height())
            self.canvas.create_text(
                canvas_w / 2,
                canvas_h / 2,
                text=self.empty_text,
                fill="#1f1f1f",
                font=("Meiryo UI", 12),
                anchor="center",
                tags=("empty_message",),
            )
            return

        self._draw_image()
        self._draw_annotations()

        canvas_w = max(1, self.canvas.winfo_width())
        canvas_h = max(1, self.canvas.winfo_height())
        self.canvas.configure(
            scrollregion=(-canvas_w, -canvas_h, canvas_w * 2, canvas_h * 2)
        )

    def _draw_image(self):
        base_w = self.display_width or self._pil_image.size[0]
        base_h = self.display_height or self._pil_image.size[1]
        draw_w = max(1, int(base_w * self.scale))
        draw_h = max(1, int(base_h * self.scale))

        display_image = self._pil_image.resize((draw_w, draw_h), Image.LANCZOS)
        self._tk_image = ImageTk.PhotoImage(display_image)

        self.canvas.create_image(
            self.offset_x,
            self.offset_y,
            image=self._tk_image,
            anchor="center",
            tags=("image",),
        )

    def _draw_annotations(self):
        if not self.show_annotations:
            return

        for ann_id, ann in self.annotations.items():
            cx, cy = self.image_to_canvas(ann["x"], ann["y"])
            self._create_annotation_object(ann_id, cx, cy, ann["no"])

    def _to_display_number(self, value):
        """赤丸の中に表示する番号文字列を作る。

        以前は 1 -> ① のような丸数字へ変換していたが、
        その上から create_oval で赤丸を描くと二重丸になるため、
        表示文字は通常の番号にする。
        CSV保存・編集ダイアログでは元の番号文字列を保持する。
        """
        text = str(value).strip()

        try:
            # "10.0" のような整数相当の文字列は "10" として表示する
            number_float = float(text)
            number = int(number_float)
            if number_float == number:
                return str(number)
        except ValueError:
            pass

        return text

    def _create_annotation_object(self, ann_id, cx, cy, text):
        """Excel/PowerPointの図形のように、Canvas上の独立オブジェクトとして作る。"""
        tag = f"ann_{ann_id}"
        common_tags = ("annotation", tag)

        ann = self.annotations.get(ann_id, {})
        base_size = float(ann.get("size", DEFAULT_MARKER_SIZE))

        # 画像の拡大縮小に追従させるため、赤丸直径・文字サイズ・線幅を scale 倍する。
        display_diameter = max(4, base_size * self.scale)
        line_width = max(1, int(round(2 * self.scale)))

        display_text = self._to_display_number(text)

        # 番号文字は赤丸のサイズに合わせて自動調整する。
        # 10, 100 など桁数が増えても赤丸内に収まりやすいようにする。
        text_len = max(1, len(display_text))
        font_size = max(6, int(display_diameter * (0.56 if text_len == 1 else 0.46 if text_len == 2 else 0.36)))

        oval_item = self.canvas.create_oval(
            cx - display_diameter / 2,
            cy - display_diameter / 2,
            cx + display_diameter / 2,
            cy + display_diameter / 2,
            outline="#ff0000",
            width=line_width,
            tags=common_tags + ("ann_oval",),
        )

        text_item = self.canvas.create_text(
            cx,
            cy,
            text=display_text,
            font=("Meiryo UI", font_size, "bold"),
            fill="#ff0000",
            anchor="center",
            tags=common_tags + ("ann_text",),
        )

        # 赤丸を背面、文字を前面にする
        self.canvas.tag_lower(oval_item, text_item)
        self.canvas.tag_raise(text_item)

        # この番号オブジェクトだけを操作対象にする
        self.canvas.tag_bind(tag, "<ButtonPress-1>", lambda e, aid=ann_id: self._on_object_drag_start(e, aid))
        self.canvas.tag_bind(tag, "<B1-Motion>", lambda e, aid=ann_id: self._on_object_drag_move(e, aid))
        self.canvas.tag_bind(tag, "<ButtonRelease-1>", lambda e, aid=ann_id: self._on_object_drag_end(e, aid))
        self.canvas.tag_bind(tag, "<Double-Button-1>", lambda e, aid=ann_id: self._edit_annotation_no(aid))

    # =========================================================
    # 画像パン・ズーム
    # =========================================================
    def _on_canvas_configure(self, event=None):
        self.redraw()

    def _on_mousewheel(self, event):
        zoom_in = event.delta > 0
        self._zoom_at(event.x, event.y, zoom_in)

    def _on_mousewheel_linux(self, event):
        zoom_in = event.num == 4
        self._zoom_at(event.x, event.y, zoom_in)

    def _zoom_at(self, mouse_x, mouse_y, zoom_in):
        if self._pil_image is None:
            return

        old_scale = self.scale
        factor = 1.1 if zoom_in else 1 / 1.1
        new_scale = max(0.1, min(old_scale * factor, 10.0))

        if new_scale == old_scale:
            return

        self.offset_x = mouse_x - (mouse_x - self.offset_x) * (new_scale / old_scale)
        self.offset_y = mouse_y - (mouse_y - self.offset_y) * (new_scale / old_scale)
        self.scale = new_scale
        self.redraw()

    def _on_pan_start(self, event):
        # 番号オブジェクト上のクリックなら、背景パンは開始しない
        current_tags = self.canvas.gettags("current")
        if "annotation" in current_tags:
            self._pan_start = None
            return
        self._pan_start = (event.x, event.y)

    def _on_pan_move(self, event):
        # 番号オブジェクトをドラッグ中は、マウスが番号の外へ出ても
        # Canvas全体のドラッグイベントで位置を更新する。
        if self._object_drag is not None:
            self._on_object_drag_move(event, self._object_drag["ann_id"])
            return

        if self._pan_start is None:
            return

        prev_x, prev_y = self._pan_start
        self.offset_x += event.x - prev_x
        self.offset_y += event.y - prev_y
        self._pan_start = (event.x, event.y)
        self.redraw()

    def _on_pan_end(self, event):
        # 番号オブジェクトのドラッグ終了もCanvas側で受ける。
        # grab_setしているため、キャンバス外で離してもここに届きやすい。
        if self._object_drag is not None:
            self._on_object_drag_end(event, self._object_drag["ann_id"])
        self._pan_start = None

    # =========================================================
    # 番号オブジェクト操作
    # =========================================================
    def _on_object_drag_start(self, event, ann_id):
        ann = self.annotations.get(ann_id)
        if ann is None:
            return "break"

        # ドラッグ中にマウスが番号オブジェクトやCanvasの外へ出ても
        # イベントを取り続けられるようにする。
        try:
            self.canvas.grab_set()
        except tk.TclError:
            pass

        current_image_x, current_image_y = self.canvas_to_image(event.x, event.y)
        if ann_id in self.selected_annotation_ids:
            target_ids = set(self.selected_annotation_ids)
        else:
            target_ids = {ann_id}

        self._object_drag = {
            "ann_id": ann_id,
            "start_x": current_image_x,
            "start_y": current_image_y,
            "targets": {
                target_id: (target_ann["x"], target_ann["y"])
                for target_id, target_ann in self.annotations.items()
                if target_id in target_ids
            },
        }
        return "break"

    def _on_object_drag_move(self, event, ann_id):
        if self._object_drag is None or self._object_drag["ann_id"] != ann_id:
            return "break"

        # grab_set中はevent.x/event.yがCanvas外の値になることがあります。
        # その値もそのままCanvas座標として使うことで、画面全体へ自由に移動できます。
        image_x, image_y = self.canvas_to_image(event.x, event.y)
        move_x = image_x - self._object_drag["start_x"]
        move_y = image_y - self._object_drag["start_y"]

        for target_id, (start_ann_x, start_ann_y) in self._object_drag["targets"].items():
            target_ann = self.annotations.get(target_id)
            if target_ann is None:
                continue
            target_ann["x"] = start_ann_x + move_x
            target_ann["y"] = start_ann_y + move_y

        # 位置制限はしない。
        # 画像外・Canvas外に相当する座標にも移動できるようにする。
        self.redraw()
        return "break"

    def _on_object_drag_end(self, event, ann_id):
        self._object_drag = None
        try:
            self.canvas.grab_release()
        except tk.TclError:
            pass
        return "break"

    def add_annotation(self):
        """新しい番号を画像の左上へ仮配置し、番号・サイズ設定画面を開く。"""
        if self._pil_image is None:
            messagebox.showinfo("番号追加", "先に図面画像を開いてください。", parent=self.winfo_toplevel())
            return

        # 番号の座標は赤丸の中心位置として管理している。
        # 画像左上の内側に収まるよう、赤丸半径 + 余白 を初期位置にする。
        initial_size = DEFAULT_MARKER_SIZE
        top_left_margin = 10
        image_x = top_left_margin + initial_size / 2
        image_y = top_left_margin + initial_size / 2

        # 小さい画像でも画像範囲内へ収まるようにする。
        base_w = self.display_width or self._pil_image.size[0]
        base_h = self.display_height or self._pil_image.size[1]
        image_x = min(max(0, image_x), base_w)
        image_y = min(max(0, image_y), base_h)

        ann_id = self._next_annotation_id
        self._next_annotation_id += 1
        self.annotations[ann_id] = {
            "no": "",
            "x": image_x,
            "y": image_y,
            "size": initial_size,
        }
        self.redraw()
        self._edit_annotation_no(ann_id, is_new=True)

    def _edit_annotation_no(self, ann_id, is_new=False):
        ann = self.annotations.get(ann_id)
        if ann is None:
            return "break"

        dialog = tk.Toplevel(self.winfo_toplevel())
        dialog.title("検査項目番号・サイズの変更")
        dialog.resizable(False, False)
        dialog.transient(self.winfo_toplevel())
        dialog.grab_set()

        body = tk.Frame(dialog, padx=18, pady=16)
        body.pack(fill="both", expand=True)

        no_var = tk.StringVar(value=str(ann.get("no", "")))
        size_var = tk.StringVar(value=str(int(round(float(ann.get("size", DEFAULT_MARKER_SIZE))))))
        size_scale_var = tk.DoubleVar(value=float(size_var.get()))

        tk.Label(body, text="検査項目番号", font=("Meiryo UI", 10)).grid(
            row=0, column=0, sticky="w", padx=(0, 10), pady=5
        )
        no_entry = tk.Entry(body, textvariable=no_var, width=16, font=("Meiryo UI", 10))
        no_entry.grid(row=0, column=1, sticky="w", pady=5)

        tk.Label(body, text="赤丸サイズ(px)", font=("Meiryo UI", 10)).grid(
            row=1, column=0, sticky="w", padx=(0, 10), pady=5
        )

        size_frame = tk.Frame(body)
        size_frame.grid(row=1, column=1, sticky="w", pady=5)

        size_spin = tk.Spinbox(
            size_frame,
            from_=MIN_MARKER_SIZE,
            to=MAX_MARKER_SIZE,
            increment=1,
            textvariable=size_var,
            width=8,
            font=("Meiryo UI", 10),
        )
        size_spin.pack(side="left")

        size_scale = tk.Scale(
            body,
            from_=MIN_MARKER_SIZE,
            to=MAX_MARKER_SIZE,
            orient="horizontal",
            variable=size_scale_var,
            length=220,
            showvalue=False,
            command=lambda value: size_var.set(str(int(round(float(value))))),
        )
        size_scale.grid(row=2, column=0, columnspan=2, sticky="we", pady=(4, 10))

        help_label = tk.Label(
            body,
            text="",
            font=("Meiryo UI", 8),
            fg="#555555",
            justify="left",
        )
        help_label.grid(row=3, column=0, columnspan=2, sticky="w", pady=(0, 12))

        button_row = tk.Frame(body)
        button_row.grid(row=4, column=0, columnspan=2, sticky="e")

        def close_dialog(remove_new=False):
            if remove_new and is_new:
                self.annotations.pop(ann_id, None)
                self.redraw()
            try:
                dialog.grab_release()
            except tk.TclError:
                pass
            dialog.destroy()

        def delete_annotation():
            self.annotations.pop(ann_id, None)
            self.redraw()
            close_dialog()

        def apply_changes():
            try:
                new_size = float(size_var.get())
            except ValueError:
                messagebox.showerror("入力エラー", "赤丸サイズには数値を入力してください。", parent=dialog)
                return

            if not (MIN_MARKER_SIZE <= new_size <= MAX_MARKER_SIZE):
                messagebox.showerror(
                    "入力エラー",
                    f"赤丸サイズは {MIN_MARKER_SIZE}〜{MAX_MARKER_SIZE} の範囲で入力してください。",
                    parent=dialog,
                )
                return

            ann["no"] = no_var.get().strip()
            ann["size"] = new_size
            self.redraw()
            close_dialog()

        tk.Button(
            button_row,
            text="削除",
            width=8,
            command=delete_annotation,
            fg="#D13438",
        ).pack(side="left", padx=(0, 18))
        tk.Button(button_row, text="OK", width=8, command=apply_changes).pack(side="left", padx=(0, 8))
        tk.Button(
            button_row,
            text="キャンセル",
            width=10,
            command=lambda: close_dialog(remove_new=True),
        ).pack(side="left")

        dialog.protocol("WM_DELETE_WINDOW", lambda: close_dialog(remove_new=True))
        no_entry.focus_set()
        no_entry.selection_range(0, tk.END)

        self.update_idletasks()
        x = self.winfo_toplevel().winfo_rootx() + 80
        y = self.winfo_toplevel().winfo_rooty() + 80
        dialog.geometry(f"+{max(0, x)}+{max(0, y)}")
        return "break"

    def export_annotations_to_csv(self, csv_path):
        """現在の番号・座標をCSVに保存する。位置調整後の確認用。"""
        with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow([CSV_NO_COLUMN, CSV_X_COLUMN, CSV_Y_COLUMN, CSV_SIZE_COLUMN])
            for ann in self.annotations.values():
                writer.writerow([
                    ann["no"],
                    round(ann["x"], 2),
                    round(ann["y"], 2),
                    round(float(ann.get("size", DEFAULT_MARKER_SIZE)), 2),
                ])


    # =========================================================
    # 配置済み画像の出力
    # =========================================================
    def _get_pil_font(self, font_size):
        """画像出力時に使うフォントを取得する。"""
        candidates = [
            "meiryo.ttc",       # Windows
            "meiryob.ttc",      # Windows bold
            "YuGothB.ttc",      # Windows
            "/System/Library/Fonts/ヒラギノ角ゴシック W6.ttc",  # macOS Japanese
            "/System/Library/Fonts/Helvetica.ttc",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
            "/usr/share/fonts/truetype/noto/NotoSansCJK-Bold.ttc",
        ]

        for font_path in candidates:
            try:
                return ImageFont.truetype(font_path, font_size)
            except Exception:
                continue

        return ImageFont.load_default()

    def export_annotated_image(self, output_path):
        """現在配置されている赤丸番号を画像に焼き込んで出力する。

        Canvas上のオブジェクトは編集用のまま保持し、出力時だけPillowで
        背景画像のコピーに赤丸番号を描画する。
        出力画像は現在の表示用画像サイズ(display_width/display_height)で作成する。
        """
        if self._pil_image is None:
            raise ValueError("先に図面画像を開いてください。")

        base_w = self.display_width or self._pil_image.size[0]
        base_h = self.display_height or self._pil_image.size[1]

        # 表示用画像サイズに合わせた背景画像を作る
        base_image = self._pil_image.convert("RGBA").resize((int(base_w), int(base_h)), Image.LANCZOS)

        # 円や文字のギザつきを抑えるため、高解像度キャンバスへ描いてから縮小する
        aa = 4
        work_image = base_image.resize((int(base_w) * aa, int(base_h) * aa), Image.LANCZOS)
        draw = ImageDraw.Draw(work_image)

        red = (255, 0, 0, 255)

        for ann in self.annotations.values():
            x = float(ann.get("x", 0)) * aa
            y = float(ann.get("y", 0)) * aa
            diameter = float(ann.get("size", DEFAULT_MARKER_SIZE)) * aa
            line_width = max(1, int(round(2 * aa)))
            display_text = self._to_display_number(ann.get("no", ""))

            # 赤丸
            left = x - diameter / 2
            top = y - diameter / 2
            right = x + diameter / 2
            bottom = y + diameter / 2
            draw.ellipse((left, top, right, bottom), outline=red, width=line_width)

            # 番号文字
            text_len = max(1, len(display_text))
            font_size = max(
                6 * aa,
                int(diameter * (0.56 if text_len == 1 else 0.46 if text_len == 2 else 0.36)),
            )
            font = self._get_pil_font(font_size)
            bbox = draw.textbbox((0, 0), display_text, font=font)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
            text_x = x - text_w / 2 - bbox[0]
            text_y = y - text_h / 2 - bbox[1]
            draw.text((text_x, text_y), display_text, fill=red, font=font)

        output_image = work_image.resize((int(base_w), int(base_h)), Image.LANCZOS)

        ext = os.path.splitext(output_path)[1].lower()
        if ext in (".jpg", ".jpeg"):
            output_image = output_image.convert("RGB")
        else:
            output_image = output_image.convert("RGBA")

        output_image.save(output_path)


class CreationWizard(tk.Toplevel):
    """検査項目番号作成を工程順に案内する非モーダルサブ画面。"""

    IMAGE_FILE_TYPES = [
        ("画像ファイル", "*.png *.jpg *.jpeg *.bmp *.gif *.webp *.tif *.tiff"),
        ("すべてのファイル", "*.*"),
    ]

    def __init__(self, owner):
        super().__init__(owner)
        self.owner = owner
        self.selected_paths = {
            "drawing": owner.drawing_image_path,
            "inspection": owner.inspection_drawing_image_path,
        }
        self.result = None
        self.crop_dialog = None
        self.processing_queue = None

        self.title("検査項目番号作成")
        # メイン画面の所有ウィンドウにして、メインをクリックしても
        # 誘導画面がメイン画面の後ろへ回らないようにする。
        self.transient(owner)
        self.geometry(self._centered_geometry(1100, 600))
        self.minsize(860, 500)
        self.configure(bg="#ffffff")
        self._owner_focus_bind_id = owner.bind(
            "<FocusIn>",
            self._keep_above_owner,
            add="+",
        )

        # grab_set()/wait_window()を使わず、メイン画面を操作可能なままにする。
        self.protocol("WM_DELETE_WINDOW", self.close)
        self.bind("<Escape>", lambda event: self.close())

        self.outer = tk.Frame(
            self,
            bg="#ffffff",
            highlightbackground="#1f1f1f",
            highlightthickness=1,
            bd=0,
        )
        self.outer.pack(fill="both", expand=True, padx=4, pady=4)

        self.button_options = {
            "font": ("Meiryo UI", 12),
            "relief": "solid",
            "bd": 1,
            "bg": "#ffffff",
            "activebackground": "#e5e5e5",
            "fg": owner.text_main,
            "pady": 5,
        }

        self.show_image_selection_step()
        self.lift()

    def _keep_above_owner(self, event=None):
        """メイン画面が操作された後も誘導画面を手前に保つ。"""
        try:
            if self.winfo_exists():
                self.after_idle(self.lift)
        except tk.TclError:
            pass

    def _centered_geometry(self, width, height):
        self.owner.update_idletasks()
        x = self.owner.winfo_rootx() + max(0, (self.owner.winfo_width() - width) // 2)
        y = self.owner.winfo_rooty() + max(0, (self.owner.winfo_height() - height) // 2)
        return f"{width}x{height}+{x}+{y}"

    def _clear_page(self):
        for widget in self.outer.winfo_children():
            widget.destroy()
        for column in range(4):
            self.outer.grid_columnconfigure(column, weight=0, minsize=0)
        for row in range(8):
            self.outer.grid_rowconfigure(row, weight=0, minsize=0)

    def _add_explanation(self, text, row=6):
        explanation = tk.Frame(
            self.outer,
            bg="#ffffff",
            highlightbackground="#0e2d3b",
            highlightcolor="#0e2d3b",
            highlightthickness=1,
            bd=0,
        )
        explanation.grid(
            row=row,
            column=0,
            columnspan=4,
            sticky="nsew",
            padx=95,
            pady=(28, 18),
        )
        self.outer.grid_rowconfigure(row, weight=1)

        tk.Label(
            explanation,
            text="説明",
            bg="#ffffff",
            fg="#e52521",
            font=("Meiryo UI", 11),
            anchor="w",
        ).pack(fill="x", padx=13, pady=(10, 0))
        tk.Label(
            explanation,
            text=text,
            bg="#ffffff",
            fg=self.owner.text_main,
            font=("Meiryo UI", 11),
            justify="left",
            anchor="nw",
            wraplength=850,
        ).pack(fill="both", expand=True, padx=18, pady=(25, 14))

    def close(self):
        if self.processing_queue is not None:
            messagebox.showinfo(
                "検出処理中",
                "検査項目番号の検出処理が完了するまでお待ちください。",
                parent=self,
            )
            return
        if self.crop_dialog is not None:
            try:
                if self.crop_dialog.winfo_exists():
                    self.crop_dialog.destroy()
            except tk.TclError:
                pass
            self.crop_dialog = None
        if getattr(self.owner, "creation_wizard", None) is self:
            self.owner.creation_wizard = None
        if self._owner_focus_bind_id:
            self.owner.unbind("<FocusIn>", self._owner_focus_bind_id)
            self._owner_focus_bind_id = None
        self.destroy()

    def select_image(self, key, title):
        path = filedialog.askopenfilename(
            parent=self,
            title=title,
            filetypes=self.IMAGE_FILE_TYPES,
        )
        if not path:
            return

        try:
            with Image.open(path) as image:
                image.verify()
        except Exception as exc:
            messagebox.showerror(
                "画像読み込みエラー",
                f"画像ファイルを読み込めませんでした。\n\n{exc}",
                parent=self,
            )
            return

        self.selected_paths[key] = path
        if key == "drawing":
            self.owner.cropped_drawing_image_path = None
            self.owner.drawing_crop_box = None
        self._refresh_image_selection_status()

    def show_image_selection_step(self):
        self._clear_page()
        self.title("検査項目番号作成 - 画像選択")
        self.outer.grid_columnconfigure(0, weight=1)
        self.outer.grid_columnconfigure(1, minsize=190)
        self.outer.grid_columnconfigure(2, minsize=170)

        label_font = ("Meiryo UI", 13)
        status_font = ("Meiryo UI", 12)

        tk.Label(
            self.outer,
            text="①図面画像",
            bg="#ffffff",
            fg=self.owner.text_main,
            font=label_font,
        ).grid(row=0, column=0, sticky="w", padx=(75, 20), pady=(95, 18))
        tk.Label(
            self.outer,
            text="②検査項目番号つき図面画像",
            bg="#ffffff",
            fg=self.owner.text_main,
            font=label_font,
        ).grid(row=1, column=0, sticky="w", padx=(75, 20), pady=10)

        tk.Button(
            self.outer,
            text="選択",
            width=14,
            command=lambda: self.select_image("drawing", "①図面画像を選択"),
            **self.button_options,
        ).grid(row=0, column=1, padx=10, pady=(95, 18))
        tk.Button(
            self.outer,
            text="選択",
            width=14,
            command=lambda: self.select_image(
                "inspection", "②検査項目番号つき図面画像を選択"
            ),
            **self.button_options,
        ).grid(row=1, column=1, padx=10, pady=10)

        self.drawing_status_var = tk.StringVar()
        self.inspection_status_var = tk.StringVar()
        self.drawing_status_label = tk.Label(
            self.outer,
            textvariable=self.drawing_status_var,
            bg="#ffffff",
            font=status_font,
        )
        self.inspection_status_label = tk.Label(
            self.outer,
            textvariable=self.inspection_status_var,
            bg="#ffffff",
            font=status_font,
        )
        self.drawing_status_label.grid(
            row=0, column=2, sticky="w", padx=(16, 24), pady=(95, 18)
        )
        self.inspection_status_label.grid(
            row=1, column=2, sticky="w", padx=(16, 24), pady=10
        )

        button_area = tk.Frame(self.outer, bg="#ffffff")
        button_area.grid(
            row=2, column=0, columnspan=3, sticky="e", padx=170, pady=(20, 6)
        )
        self.next_button = tk.Button(
            button_area,
            text="次へ",
            width=14,
            command=self.apply_image_selection,
            **self.button_options,
        )
        self.next_button.pack(side="left", padx=(0, 12))
        tk.Button(
            button_area,
            text="キャンセル",
            width=14,
            command=self.close,
            **self.button_options,
        ).pack(side="left")

        self._add_explanation(
            "図面画像と検査項目番号つき図面画像を選択してください。",
            row=6,
        )
        self._refresh_image_selection_status()

    def _refresh_image_selection_status(self):
        drawing_selected = bool(self.selected_paths["drawing"])
        inspection_selected = bool(self.selected_paths["inspection"])
        self.drawing_status_var.set("選択済み" if drawing_selected else "未選択")
        self.inspection_status_var.set("選択済み" if inspection_selected else "未選択")
        self.drawing_status_label.configure(
            fg=self.owner.text_main if drawing_selected else "#008cff"
        )
        self.inspection_status_label.configure(
            fg=self.owner.text_main if inspection_selected else "#008cff"
        )
        self.next_button.configure(
            state="normal" if drawing_selected and inspection_selected else "disabled"
        )

    def apply_image_selection(self):
        self.owner.drawing_image_path = self.selected_paths["drawing"]
        self.owner.inspection_drawing_image_path = self.selected_paths["inspection"]
        self.owner.cropped_drawing_image_path = None
        self.owner.drawing_crop_box = None
        self.owner.number_position_csv_path = None
        self.owner.number_position_result_image_path = None
        self.owner.number_position_mask_image_path = None
        self.owner.viewer.clear_annotations()

        self.owner.image_sources["図面画像"] = self.selected_paths["drawing"]
        self.owner.image_sources["検査項目番号つき図面画像"] = self.selected_paths[
            "inspection"
        ]
        self.owner.image_sources["処理画像1"] = None
        self.owner.image_sources["処理画像2"] = None
        self.owner.image_view_display_sizes["図面画像"] = None
        self.owner.image_view_display_sizes["検査項目番号つき図面画像"] = None
        self.owner.image_view_states["図面画像"] = self.owner.get_default_image_view_state()
        self.owner.image_view_states[
            "検査項目番号つき図面画像"
        ] = self.owner.get_default_image_view_state()
        self.owner.set_viewer_image_with_view_state(
            self.selected_paths["drawing"],
            image_type="図面画像",
        )
        self.show_trimming_step()

    def show_trimming_step(self):
        # トリミング範囲を判断できるよう、メイン画面は参照用の
        # 検査項目番号つき図面画像へ強制的に切り替える。
        inspection_path = (
            self.selected_paths.get("inspection")
            or self.owner.inspection_drawing_image_path
        )
        if inspection_path:
            self.owner.image_sources["検査項目番号つき図面画像"] = inspection_path
            self.owner.set_viewer_image_with_view_state(
                inspection_path,
                image_type="検査項目番号つき図面画像",
            )

        self._clear_page()
        self.title("検査項目番号作成 - トリミング")
        self.outer.grid_columnconfigure(0, weight=1)
        self.outer.grid_columnconfigure(1, weight=1)

        button_area = tk.Frame(self.outer, bg="#ffffff")
        button_area.grid(
            row=1, column=0, columnspan=2, sticky="ew", padx=115, pady=(155, 0)
        )
        button_area.grid_columnconfigure(0, weight=1)
        button_area.grid_columnconfigure(1, weight=1)

        tk.Button(
            button_area,
            text="トリミング開始",
            width=26,
            command=self.start_trimming,
            **self.button_options,
        ).grid(row=0, column=0, padx=(0, 85))
        tk.Button(
            button_area,
            text="トリミングせず次へ",
            width=26,
            command=self.skip_trimming,
            **self.button_options,
        ).grid(row=0, column=1)

        self._add_explanation(
            "検査項目番号つき図面画像のサイズに合うように"
            "図面画像をトリミングしてください。",
            row=6,
        )
        self.lift()

    def start_trimming(self):
        if self.crop_dialog is not None:
            try:
                if self.crop_dialog.winfo_exists():
                    self.crop_dialog.lift()
                    self.crop_dialog.focus_force()
                    return
            except tk.TclError:
                self.crop_dialog = None

        self.crop_dialog = self.owner.open_drawing_crop_dialog(
            on_complete=self.show_detection_confirmation_step,
            modal=False,
        )

    def skip_trimming(self):
        self.owner.cropped_drawing_image_path = None
        self.owner.drawing_crop_box = None
        self.owner.image_sources["図面画像"] = self.owner.drawing_image_path
        self.owner.image_view_display_sizes["図面画像"] = None
        self.owner.image_view_states["図面画像"] = self.owner.get_default_image_view_state()
        self.owner.set_viewer_image_with_view_state(
            self.owner.drawing_image_path,
            image_type="図面画像",
        )
        self.show_detection_confirmation_step()

    def show_detection_confirmation_step(self):
        self.crop_dialog = None
        try:
            if not self.winfo_exists():
                return
        except tk.TclError:
            return
        self._clear_page()
        self.title("番号座標CSV作成")
        self.outer.grid_columnconfigure(0, weight=1)
        self.outer.grid_rowconfigure(1, weight=1)

        tk.Label(
            self.outer,
            text="検査項目番号の検出処理を開始します",
            bg="#ffffff",
            fg=self.owner.text_main,
            font=("Meiryo UI", 13),
            anchor="w",
        ).grid(row=0, column=0, sticky="ew", padx=70, pady=(80, 0))

        button_area = tk.Frame(self.outer, bg="#ffffff")
        button_area.grid(row=2, column=0, sticky="e", padx=95, pady=(0, 55))
        self.process_button = tk.Button(
            button_area,
            text="OK",
            width=12,
            command=self.run_detection,
            **self.button_options,
        )
        self.process_button.pack(side="left", padx=(0, 12))
        self.process_cancel_button = tk.Button(
            button_area,
            text="キャンセル",
            width=12,
            command=self.close,
            **self.button_options,
        )
        self.process_cancel_button.pack(side="left")

    def run_detection(self):
        self.process_button.configure(state="disabled")
        self.process_cancel_button.configure(state="disabled")
        self.processing_queue = queue.Queue()

        def worker():
            try:
                result = process_number_position_image(self.selected_paths["inspection"])
            except Exception as exc:
                self.processing_queue.put(("error", exc))
                return
            self.processing_queue.put(("success", result))

        threading.Thread(target=worker, daemon=True).start()
        self.after(100, self._poll_detection_result)

    def _poll_detection_result(self):
        if self.processing_queue is None:
            return
        try:
            status, payload = self.processing_queue.get_nowait()
        except queue.Empty:
            self.after(100, self._poll_detection_result)
            return

        self.processing_queue = None
        if status == "error":
            self.process_button.configure(text="再実行", state="normal")
            self.process_cancel_button.configure(text="閉じる", state="normal")
            messagebox.showerror(
                "番号座標CSV作成エラー",
                f"番号座標CSVの作成に失敗しました。\n\n{payload}",
                parent=self,
            )
            return

        result = payload
        self.result = result
        self.owner.number_position_csv_path = str(result["csv_output_path"])
        self.owner.number_position_result_image_path = str(result["result_output_path"])
        self.owner.number_position_mask_image_path = str(result["mask_output_path"])
        self.owner.update_image_sources_after_number_detection(self.selected_paths, result)
        self.owner.set_viewer_image_with_view_state(
            self.selected_paths["inspection"],
            image_type="検査項目番号つき図面画像",
        )
        self.show_detection_complete_step()

    def show_detection_complete_step(self):
        self._clear_page()
        self.title("番号座標CSV作成 - 完了")
        self.outer.grid_columnconfigure(0, weight=1)

        tk.Label(
            self.outer,
            text="検査項目番号の検出処理が完了しました",
            bg="#ffffff",
            fg=self.owner.text_main,
            font=("Meiryo UI", 14),
            anchor="w",
        ).grid(row=0, column=0, sticky="ew", padx=95, pady=(70, 22))

        detail = (
            f"検出数: {self.result['count']}\n\n"
            f"入力画像: {self.selected_paths['inspection']}\n"
            f"保存先: {self.result['csv_output_path']}"
        )
        tk.Label(
            self.outer,
            text=detail,
            bg="#ffffff",
            fg=self.owner.text_main,
            font=("Meiryo UI", 10),
            justify="left",
            anchor="w",
            wraplength=880,
        ).grid(row=1, column=0, sticky="ew", padx=95, pady=(0, 25))

        self.number_result_button = tk.Button(
            self.outer,
            text="番号読み取り結果入力",
            width=26,
            command=self.apply_number_reading_result,
            **self.button_options,
        )
        self.number_result_button.grid(row=2, column=0, sticky="e", padx=120, pady=(0, 0))

        self._add_explanation(
            "クリップボードに検査項目番号画像が貼り付けられました。\n"
            "チャット欄に張り付けて番号読み取りエージェントで"
            "番号の読み取りをしてください。",
            row=6,
        )

    def apply_number_reading_result(self):
        if not self.owner.open_number_reading_result_csv(parent=self):
            return
        if not self.owner.draw_numbers_on_drawing(parent=self):
            return

        self.close()
        self.owner.lift()
        self.owner.focus_force()


class ImageDisplayAreaOnly(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("検査項目番号作成")
        self.geometry("1227x630")
        self.minsize(900, 500)

        self.bg_main = "#f7f7f7"
        self.panel_bg = "#f2f2f2"
        self.inner_border = "#0e2d3b"
        self.button_bg = "#f7f7f7"
        self.button_active_bg = "#e1e1e1"
        self.text_main = "#1F1F1F"

        self.configure(bg=self.bg_main)

        self.image_view_types = ("図面画像", "検査項目番号つき図面画像", "処理画像1", "処理画像2")
        self.image_view_states = {image_type: None for image_type in self.image_view_types}
        # 画像切替で set_image() すると display_width/display_height が元画像サイズへ戻るため、
        # 図面画像を検査項目番号つき図面画像サイズへ合わせた状態も画像種別ごとに保持する。
        self.image_view_display_sizes = {image_type: None for image_type in self.image_view_types}
        self.current_image_view_type = "図面画像"
        self.image_sources = {image_type: None for image_type in self.image_view_types}
        self.image_type_var = tk.StringVar(value=self.current_image_view_type)

        self.drawing_image_path = None
        self.inspection_drawing_image_path = None
        # 図面画像を番号描画前にトリミングした場合の一時ファイルと範囲
        self.cropped_drawing_image_path = None
        self.drawing_crop_box = None
        self.number_position_csv_path = None
        self.number_position_result_image_path = None
        self.number_position_mask_image_path = None
        self.creation_wizard = None

        self._setup_style()
        self._build_layout()

    def _setup_style(self):
        self.style = ttk.Style()
        self.style.theme_use("default")
        self.style.configure(
            "Modern.TCombobox",
            fieldbackground="#ffffff",
            background="#ffffff",
            foreground=self.text_main,
            bordercolor=self.inner_border,
            lightcolor=self.inner_border,
            darkcolor=self.inner_border,
            arrowsize=16,
            padding=6,
            relief="solid",
        )

    def _thin_frame(self, parent, bg=None):
        return tk.Frame(
            parent,
            bg=bg if bg else self.panel_bg,
            highlightbackground=self.inner_border,
            highlightcolor=self.inner_border,
            highlightthickness=1,
            bd=0,
        )

    def _build_layout(self):
        main_frame = tk.Frame(self, bg=self.bg_main)
        main_frame.pack(fill="both", expand=True)
        self._build_image_area(main_frame)

    def _make_header_button(self, parent, text, command):
        return tk.Button(
            parent,
            text=text,
            font=("Meiryo UI", 12),
            relief="solid",
            bd=1,
            bg=self.button_bg,
            activebackground=self.button_active_bg,
            fg=self.text_main,
            padx=11,
            pady=3,
            highlightthickness=0,
            command=command,
        )

    # =========================================================
    # 画像表示エリア
    # =========================================================
    def _build_image_area(self, parent):
        header = tk.Frame(parent, bg=self.bg_main)
        header.pack(fill="x", padx=23, pady=(21, 0))

        self._make_header_button(
            header,
            text="番号追加",
            command=self.viewer_add_annotation,
        ).pack(side="left", padx=(0, 8))

        self._make_header_button(
            header,
            text="サイズ設定",
            command=self.open_annotation_size_settings,
        ).pack(side="left", padx=(0, 8))

        self._make_header_button(
            header,
            text="全選択",
            command=self.viewer_select_all_annotations,
        ).pack(side="left", padx=(0, 8))

        self._make_header_button(
            header,
            text="選択解除",
            command=self.viewer_deselect_all_annotations,
        ).pack(side="left", padx=(0, 8))

        self._make_header_button(
            header,
            text="画像出力",
            command=self.export_current_annotated_image,
        ).pack(side="right", padx=(8, 0))

        self.image_type_combo = ttk.Combobox(
            header,
            textvariable=self.image_type_var,
            values=self.image_view_types,
            state="readonly",
            width=12,
            font=("Meiryo UI", 12),
            style="Modern.TCombobox",
        )
        self.image_type_combo.pack(side="right", padx=(8, 0), ipady=2)
        self.image_type_combo.bind("<<ComboboxSelected>>", self.on_image_view_type_changed)

        self.image_placeholder = tk.Frame(
            parent,
            bg=self.panel_bg,
            highlightbackground=self.inner_border,
            highlightcolor=self.inner_border,
            highlightthickness=1,
            bd=0,
        )
        self.image_placeholder.pack(
            fill="both",
            expand=True,
            padx=23,
            pady=(14, 0),
        )

        self.viewer = ZoomableImageViewer(
            master=self.image_placeholder,
            width=800,
            height=500,
            bg=self.panel_bg,
            empty_text="図面表示エリア",
        )
        self.viewer.pack(fill="both", expand=True, padx=1, pady=1)
        self.viewer.set_selection_status_callback(self.update_selection_status_display)
        self.save_current_image_view_state("図面画像")

        footer = tk.Frame(parent, bg=self.bg_main)
        footer.pack(fill="x", padx=(19, 19), pady=(7, 7))

        self.selection_status_var = tk.StringVar(value="選択状態：未選択")
        tk.Label(
            footer,
            textvariable=self.selection_status_var,
            bg=self.bg_main,
            fg=self.text_main,
            font=("Meiryo UI", 11),
        ).pack(side="left")

        self._make_header_button(
            footer,
            text="作成開始",
            command=self.start_creation,
        ).pack(side="right")

    def start_creation(self):
        """工程誘導サブ画面を非モーダルで開く。"""
        if self.creation_wizard is not None:
            try:
                if self.creation_wizard.winfo_exists():
                    self.creation_wizard.deiconify()
                    self.creation_wizard.lift()
                    self.creation_wizard.focus_force()
                    return
            except tk.TclError:
                pass

        self.creation_wizard = CreationWizard(self)

    def should_show_annotations_for_image_type(self, image_type):
        """番号編集オブジェクトは図面画像表示時だけ表示する。"""
        return image_type == "図面画像"

    def on_image_view_type_changed(self, event=None):
        """コンボボックスで選択された画像へ表示を切り替える。"""
        selected_type = self.get_selected_image_view_type()
        image_source = self.image_sources.get(selected_type)

        if not image_source or not Path(image_source).is_file():
            messagebox.showinfo(
                "表示画像切替",
                f"{selected_type} はまだ表示できません。\n先に「図面読み込み」で番号検出処理を完了してください。",
                parent=self,
            )
            self.image_type_var.set(self.current_image_view_type)
            return

        self.set_viewer_image_with_view_state(image_source, image_type=selected_type)

    def update_image_sources_after_number_detection(self, selected_paths, result):
        """番号検出処理完了後、コンボボックスで切替できる画像パスを登録する。"""
        self.image_sources["図面画像"] = selected_paths.get("drawing")
        self.image_sources["検査項目番号つき図面画像"] = selected_paths.get("inspection")
        self.image_sources["処理画像1"] = str(result["mask_output_path"])
        self.image_sources["処理画像2"] = str(result["result_output_path"])

        # 新しい画像を読み込んだ時点では、各画像は元サイズ表示に戻す。
        # 番号描画後の図面画像だけ、検査画像サイズへ合わせた表示サイズを後で保存する。
        for image_type in self.image_view_types:
            self.image_view_display_sizes[image_type] = None

        if hasattr(self, "image_type_combo"):
            self.image_type_combo.configure(values=self.image_view_types)

    def open_folder_in_explorer(self, folder_path):
        """保存先フォルダをOS標準のファイル管理画面で開く。"""
        folder_path = Path(folder_path)
        try:
            if os.name == "nt":
                os.startfile(str(folder_path))
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(folder_path)])
            else:
                subprocess.Popen(["xdg-open", str(folder_path)])
        except Exception as exc:
            messagebox.showwarning(
                "フォルダ表示",
                f"保存先フォルダを自動で開けませんでした。\n\n保存先:\n{folder_path}\n\n{exc}",
                parent=self,
            )

    def export_current_annotated_image(self):
        """現在の番号描画済み画像を app2.py と同じ階層の edit_img フォルダへ保存する。"""
        try:
            if getattr(self.viewer, "_pil_image", None) is None:
                messagebox.showinfo(
                    "画像出力",
                    "先に図面画像を読み込んでください。",
                    parent=self,
                )
                return

            base_directory = get_gui_base_directory()
            edit_img_directory = base_directory / "edit_img"
            edit_img_directory.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = edit_img_directory / f"edit_image_{timestamp}.png"

            self.viewer.export_annotated_image(output_path)

        except Exception as exc:
            messagebox.showerror(
                "画像出力",
                f"画像の保存に失敗しました。\n\n{exc}",
                parent=self,
            )
            return

        messagebox.showinfo(
            "画像出力 完了",
            f"画像を保存しました。\n\n保存先:\n{output_path}",
            parent=self,
        )
        self.open_folder_in_explorer(edit_img_directory)

    def open_annotation_size_settings(self):
        """配置済み番号の赤丸サイズを一括変更するサブ画面を開く。"""
        if not getattr(self.viewer, "annotations", None):
            messagebox.showinfo(
                "サイズ設定",
                "サイズを変更する番号がありません。\n先に「番号追加」または「番号描画」で番号を配置してください。",
                parent=self,
            )
            return

        dialog = tk.Toplevel(self)
        dialog.title("サイズ設定")
        dialog.resizable(False, False)
        dialog.transient(self)
        dialog.configure(bg="#ffffff")
        dialog.grab_set()

        body = tk.Frame(dialog, bg="#ffffff", padx=20, pady=18)
        body.pack(fill="both", expand=True)

        current_size = int(round(self.viewer.get_common_annotation_size()))
        size_var = tk.StringVar(value=str(current_size))
        size_scale_var = tk.DoubleVar(value=current_size)

        tk.Label(
            body,
            text="配置されている番号のサイズを一斉に変更します。",
            bg="#ffffff",
            fg=self.text_main,
            font=("Meiryo UI", 10),
            justify="left",
        ).grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 14))

        tk.Label(
            body,
            text="赤丸サイズ(px)",
            bg="#ffffff",
            fg=self.text_main,
            font=("Meiryo UI", 10),
        ).grid(row=1, column=0, sticky="w", padx=(0, 12), pady=6)

        size_spin = tk.Spinbox(
            body,
            from_=MIN_MARKER_SIZE,
            to=MAX_MARKER_SIZE,
            increment=1,
            textvariable=size_var,
            width=8,
            font=("Meiryo UI", 10),
        )
        size_spin.grid(row=1, column=1, sticky="w", pady=6)

        size_scale = tk.Scale(
            body,
            from_=MIN_MARKER_SIZE,
            to=MAX_MARKER_SIZE,
            orient="horizontal",
            variable=size_scale_var,
            length=280,
            showvalue=False,
            bg="#ffffff",
            highlightthickness=0,
            command=lambda value: size_var.set(str(int(round(float(value))))),
        )
        size_scale.grid(row=2, column=0, columnspan=2, sticky="we", pady=(4, 12))

        count_label = tk.Label(
            body,
            text=f"対象: {len(self.viewer.annotations)}件",
            bg="#ffffff",
            fg="#555555",
            font=("Meiryo UI", 9),
        )
        count_label.grid(row=3, column=0, columnspan=2, sticky="w", pady=(0, 14))

        button_row = tk.Frame(body, bg="#ffffff")
        button_row.grid(row=4, column=0, columnspan=2, sticky="e")

        def close_dialog():
            try:
                dialog.grab_release()
            except tk.TclError:
                pass
            dialog.destroy()

        def apply_size():
            try:
                new_size = float(size_var.get())
            except ValueError:
                messagebox.showerror(
                    "入力エラー",
                    "赤丸サイズには数値を入力してください。",
                    parent=dialog,
                )
                return

            try:
                self.viewer.set_all_annotation_sizes(new_size)
            except ValueError as exc:
                messagebox.showerror("入力エラー", str(exc), parent=dialog)
                return

            # 一括変更後は、編集できる図面画像表示へ戻す。
            if self.current_image_view_type != "図面画像" and self.image_sources.get("図面画像"):
                self.set_viewer_image_with_view_state(
                    self.image_sources["図面画像"],
                    image_type="図面画像",
                )
            else:
                self.viewer.set_annotation_visibility(True)

            messagebox.showinfo(
                "サイズ設定",
                f"{len(self.viewer.annotations)}件の番号サイズを {new_size:g}px に変更しました。",
                parent=dialog,
            )
            close_dialog()

        tk.Button(
            button_row,
            text="OK",
            width=8,
            command=apply_size,
            bg="#ffffff",
            activebackground="#e5e5e5",
            fg=self.text_main,
            relief="solid",
            bd=1,
            font=("Meiryo UI", 10),
        ).pack(side="left", padx=(0, 8))

        tk.Button(
            button_row,
            text="キャンセル",
            width=10,
            command=close_dialog,
            bg="#ffffff",
            activebackground="#e5e5e5",
            fg=self.text_main,
            relief="solid",
            bd=1,
            font=("Meiryo UI", 10),
        ).pack(side="left")

        dialog.protocol("WM_DELETE_WINDOW", close_dialog)
        dialog.bind("<Escape>", lambda event: close_dialog())
        dialog.bind("<Return>", lambda event: apply_size())

        self.update_idletasks()
        dialog_width = 420
        dialog_height = 230
        x = self.winfo_rootx() + max(0, (self.winfo_width() - dialog_width) // 2)
        y = self.winfo_rooty() + max(0, (self.winfo_height() - dialog_height) // 2)
        dialog.geometry(f"{dialog_width}x{dialog_height}+{x}+{y}")

        size_spin.focus_set()
        size_spin.selection_range(0, tk.END)
        dialog.wait_window()

    def viewer_add_annotation(self):
        """番号追加ボタンから新規番号の設定画面を開く。"""
        self.viewer.add_annotation()

    def viewer_select_all_annotations(self):
        """配置済み番号を全選択する。"""
        self.viewer.select_all_annotations()

    def viewer_deselect_all_annotations(self):
        """配置済み番号の全選択状態を解除する。"""
        self.viewer.deselect_all_annotations()

    def update_selection_status_display(self, selected_count, total_count):
        """全選択状態を画面に表示する。"""
        if selected_count and selected_count == total_count:
            text = f"選択状態：全選択中（{selected_count}件）"
        else:
            text = "選択状態：未選択"

        if hasattr(self, "selection_status_var"):
            self.selection_status_var.set(text)

    def open_drawing_crop_dialog(self, on_complete=None, modal=True):
        """図面画像をマウス矩形でトリミングする。

        modal=False の場合はメイン画面を操作可能なまま開き、トリミング完了後に
        on_complete を呼び出す。
        """
        if not self.drawing_image_path:
            messagebox.showinfo(
                "トリミング",
                "先に「図面読み込み」で図面画像を選択してください。",
                parent=self,
            )
            return None

        try:
            source_image = Image.open(self.drawing_image_path).convert("RGBA")
        except Exception as exc:
            messagebox.showerror(
                "トリミング",
                f"図面画像を読み込めませんでした。\n\n{exc}",
                parent=self,
            )
            return None

        dialog = tk.Toplevel(self)
        dialog.title("図面画像のトリミング")
        dialog.transient(self)
        dialog.geometry("900x650")
        dialog.minsize(650, 450)
        dialog.configure(bg="#ffffff")
        if modal:
            dialog.grab_set()

        info_var = tk.StringVar(
            value=(
                "左ドラッグで範囲作成。白い点をドラッグでサイズ変更、赤枠の内側をドラッグで移動。"
                "右ドラッグで画像移動、ホイールで拡大縮小。"
            )
        )
        tk.Label(
            dialog,
            textvariable=info_var,
            bg="#ffffff",
            fg=self.text_main,
            font=("Meiryo UI", 10),
            anchor="w",
        ).pack(fill="x", padx=12, pady=(10, 6))

        canvas = tk.Canvas(dialog, bg="#808080", highlightthickness=0)
        canvas.pack(fill="both", expand=True, padx=12, pady=6)

        button_row = tk.Frame(dialog, bg="#ffffff")
        button_row.pack(fill="x", padx=12, pady=(4, 12))

        img_w, img_h = source_image.size
        handle_size = 8
        hit_radius = 10
        min_rect_size = 2

        state = {
            "scale": 1.0,
            "offset_x": 450.0,
            "offset_y": 300.0,
            "tk_image": None,
            "start_image": None,
            "start_rect": None,
            "rect": None,  # (x1, y1, x2, y2) in source image coordinates
            "mode": None,  # create / move / resize_nw ... resize_se
            "pan_start": None,
        }

        def image_to_canvas(ix, iy):
            left = state["offset_x"] - img_w * state["scale"] / 2
            top = state["offset_y"] - img_h * state["scale"] / 2
            return left + ix * state["scale"], top + iy * state["scale"]

        def canvas_to_image(cx, cy, clamp=True):
            left = state["offset_x"] - img_w * state["scale"] / 2
            top = state["offset_y"] - img_h * state["scale"] / 2
            ix = (cx - left) / state["scale"]
            iy = (cy - top) / state["scale"]
            if clamp:
                ix = max(0, min(img_w, ix))
                iy = max(0, min(img_h, iy))
            return ix, iy

        def normalize_float_rect(rect=None):
            if rect is None:
                rect = state["rect"]
            if rect is None:
                return None
            x1, y1, x2, y2 = rect
            x1, x2 = sorted((float(x1), float(x2)))
            y1, y2 = sorted((float(y1), float(y2)))
            x1 = max(0.0, min(float(img_w), x1))
            x2 = max(0.0, min(float(img_w), x2))
            y1 = max(0.0, min(float(img_h), y1))
            y2 = max(0.0, min(float(img_h), y2))
            return x1, y1, x2, y2

        def normalized_rect():
            rect = normalize_float_rect()
            if rect is None:
                return None
            x1, y1, x2, y2 = rect
            x1 = max(0, min(img_w, int(round(x1))))
            x2 = max(0, min(img_w, int(round(x2))))
            y1 = max(0, min(img_h, int(round(y1))))
            y2 = max(0, min(img_h, int(round(y2))))
            if x2 < x1:
                x1, x2 = x2, x1
            if y2 < y1:
                y1, y2 = y2, y1
            return x1, y1, x2, y2

        def get_handle_positions_image(rect=None):
            rect = normalize_float_rect(rect)
            if rect is None:
                return {}
            x1, y1, x2, y2 = rect
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            return {
                "nw": (x1, y1),
                "n": (cx, y1),
                "ne": (x2, y1),
                "w": (x1, cy),
                "e": (x2, cy),
                "sw": (x1, y2),
                "s": (cx, y2),
                "se": (x2, y2),
            }

        def get_handle_positions_canvas(rect=None):
            return {
                key: image_to_canvas(ix, iy)
                for key, (ix, iy) in get_handle_positions_image(rect).items()
            }

        def point_in_rect_canvas(cx, cy):
            rect = normalize_float_rect()
            if rect is None:
                return False
            ix, iy = canvas_to_image(cx, cy, clamp=False)
            x1, y1, x2, y2 = rect
            return x1 <= ix <= x2 and y1 <= iy <= y2

        def detect_hit(cx, cy):
            if state["rect"] is None:
                return None

            handles = get_handle_positions_canvas()
            for key in ("nw", "n", "ne", "w", "e", "sw", "s", "se"):
                hx, hy = handles[key]
                if abs(cx - hx) <= hit_radius and abs(cy - hy) <= hit_radius:
                    return f"resize_{key}"

            if point_in_rect_canvas(cx, cy):
                return "move"

            return None

        def cursor_for_mode(mode):
            # Windows/Linux/macOSで比較的通りやすいTk標準カーソル名を使用
            if mode in ("resize_nw", "resize_se"):
                return "size_nw_se"
            if mode in ("resize_ne", "resize_sw"):
                return "size_ne_sw"
            if mode in ("resize_n", "resize_s"):
                return "sb_v_double_arrow"
            if mode in ("resize_w", "resize_e"):
                return "sb_h_double_arrow"
            if mode == "move":
                return "fleur"
            return "crosshair"

        def redraw():
            canvas.delete("all")
            cw = max(1, canvas.winfo_width())
            ch = max(1, canvas.winfo_height())
            if state["offset_x"] == 450.0 and cw != 1:
                state["offset_x"] = cw / 2
            if state["offset_y"] == 300.0 and ch != 1:
                state["offset_y"] = ch / 2

            draw_w = max(1, int(round(img_w * state["scale"])))
            draw_h = max(1, int(round(img_h * state["scale"])))
            resized = source_image.resize((draw_w, draw_h), Image.LANCZOS)
            state["tk_image"] = ImageTk.PhotoImage(resized)
            canvas.create_image(
                state["offset_x"],
                state["offset_y"],
                image=state["tk_image"],
                anchor="center",
            )

            rect = normalize_float_rect()
            if rect is not None:
                x1, y1, x2, y2 = rect
                cx1, cy1 = image_to_canvas(x1, y1)
                cx2, cy2 = image_to_canvas(x2, y2)

                # 添付画像に近い見た目: 赤枠 + 周囲8か所の白いハンドル
                canvas.create_rectangle(
                    cx1,
                    cy1,
                    cx2,
                    cy2,
                    outline="#ff0000",
                    width=2,
                    tags=("crop_rect",),
                )

                for key, (hx, hy) in get_handle_positions_canvas(rect).items():
                    canvas.create_rectangle(
                        hx - handle_size / 2,
                        hy - handle_size / 2,
                        hx + handle_size / 2,
                        hy + handle_size / 2,
                        fill="#ffffff",
                        outline="#0000ff",
                        width=1,
                        tags=("crop_handle", f"crop_handle_{key}"),
                    )

        def update_info():
            if state["rect"] is None:
                info_var.set(
                    "左ドラッグで範囲作成。白い点をドラッグでサイズ変更、赤枠の内側をドラッグで移動。"
                    "右ドラッグで画像移動、ホイールで拡大縮小。"
                )
                return
            x1, y1, x2, y2 = normalized_rect()
            info_var.set(
                f"選択範囲: x={x1}, y={y1}, 幅={x2 - x1}, 高さ={y2 - y1} px"
            )

        def fit_rect_inside_image(rect):
            """矩形移動時に幅・高さを変えず、画像範囲内へ位置だけ補正する。"""
            if rect is None:
                return None

            # ここで normalize_float_rect() を使うと、画像外へはみ出した時点で
            # 座標がクリップされ、移動だけのつもりでも矩形サイズが縮む。
            # そのため、まず元の幅・高さを保持してから位置だけを補正する。
            x1, y1, x2, y2 = rect
            x1, x2 = sorted((float(x1), float(x2)))
            y1, y2 = sorted((float(y1), float(y2)))

            width = x2 - x1
            height = y2 - y1

            if width >= img_w:
                x1 = 0.0
                x2 = float(img_w)
            else:
                if x1 < 0.0:
                    x1 = 0.0
                    x2 = x1 + width
                if x2 > img_w:
                    x2 = float(img_w)
                    x1 = x2 - width

            if height >= img_h:
                y1 = 0.0
                y2 = float(img_h)
            else:
                if y1 < 0.0:
                    y1 = 0.0
                    y2 = y1 + height
                if y2 > img_h:
                    y2 = float(img_h)
                    y1 = y2 - height

            return x1, y1, x2, y2

        def resize_rect(mode, ix, iy, base_rect):
            x1, y1, x2, y2 = normalize_float_rect(base_rect)
            key = mode.replace("resize_", "")

            if "w" in key:
                x1 = min(ix, x2 - min_rect_size)
            if "e" in key:
                x2 = max(ix, x1 + min_rect_size)
            if key == "n" or "n" in key:
                y1 = min(iy, y2 - min_rect_size)
            if key == "s" or "s" in key:
                y2 = max(iy, y1 + min_rect_size)

            x1 = max(0.0, min(float(img_w), x1))
            x2 = max(0.0, min(float(img_w), x2))
            y1 = max(0.0, min(float(img_h), y1))
            y2 = max(0.0, min(float(img_h), y2))

            # 画像端でつかんだ時も最小サイズを維持する
            if x2 - x1 < min_rect_size:
                if "w" in key:
                    x1 = max(0.0, x2 - min_rect_size)
                else:
                    x2 = min(float(img_w), x1 + min_rect_size)
            if y2 - y1 < min_rect_size:
                if "n" in key:
                    y1 = max(0.0, y2 - min_rect_size)
                else:
                    y2 = min(float(img_h), y1 + min_rect_size)

            return x1, y1, x2, y2

        def on_left_press(event):
            ix, iy = canvas_to_image(event.x, event.y)
            hit = detect_hit(event.x, event.y)
            state["start_image"] = (ix, iy)
            state["start_rect"] = None if state["rect"] is None else normalize_float_rect()

            if hit == "move":
                state["mode"] = "move"
            elif hit and hit.startswith("resize_"):
                state["mode"] = hit
            else:
                state["mode"] = "create"
                state["rect"] = (ix, iy, ix, iy)
                state["start_rect"] = state["rect"]

            try:
                canvas.grab_set()
            except tk.TclError:
                pass
            redraw()
            update_info()
            return "break"

        def on_left_drag(event):
            if state["mode"] is None or state["start_image"] is None:
                return "break"

            ix, iy = canvas_to_image(event.x, event.y)
            sx, sy = state["start_image"]
            base_rect = state["start_rect"]

            if state["mode"] == "create":
                state["rect"] = (sx, sy, ix, iy)

            elif state["mode"] == "move" and base_rect is not None:
                dx = ix - sx
                dy = iy - sy
                x1, y1, x2, y2 = base_rect
                state["rect"] = fit_rect_inside_image((x1 + dx, y1 + dy, x2 + dx, y2 + dy))

            elif state["mode"].startswith("resize_") and base_rect is not None:
                state["rect"] = resize_rect(state["mode"], ix, iy, base_rect)

            redraw()
            update_info()
            return "break"

        def on_left_release(event):
            state["mode"] = None
            state["start_image"] = None
            state["start_rect"] = None
            try:
                canvas.grab_release()
            except tk.TclError:
                pass
            update_info()
            redraw()
            return "break"

        def on_motion(event):
            if state["mode"] is not None:
                canvas.configure(cursor=cursor_for_mode(state["mode"]))
                return
            canvas.configure(cursor=cursor_for_mode(detect_hit(event.x, event.y)))

        def on_right_press(event):
            state["pan_start"] = (event.x, event.y)
            canvas.configure(cursor="fleur")

        def on_right_drag(event):
            if state["pan_start"] is None:
                return
            px, py = state["pan_start"]
            state["offset_x"] += event.x - px
            state["offset_y"] += event.y - py
            state["pan_start"] = (event.x, event.y)
            redraw()

        def on_right_release(event):
            state["pan_start"] = None
            canvas.configure(cursor="")

        def on_mousewheel(event):
            old_scale = state["scale"]
            if getattr(event, "num", None) == 4 or getattr(event, "delta", 0) > 0:
                new_scale = old_scale * 1.1
            else:
                new_scale = old_scale / 1.1
            new_scale = max(0.1, min(new_scale, 10.0))
            if new_scale == old_scale:
                return
            state["offset_x"] = event.x - (event.x - state["offset_x"]) * (new_scale / old_scale)
            state["offset_y"] = event.y - (event.y - state["offset_y"]) * (new_scale / old_scale)
            state["scale"] = new_scale
            redraw()

        def clear_crop():
            state["rect"] = None
            state["mode"] = None
            state["start_image"] = None
            state["start_rect"] = None
            update_info()
            redraw()

        def apply_crop():
            rect = normalized_rect()
            if rect is None:
                messagebox.showinfo("トリミング", "トリミング範囲を指定してください。", parent=dialog)
                return

            x1, y1, x2, y2 = rect
            if x2 - x1 < min_rect_size or y2 - y1 < min_rect_size:
                messagebox.showinfo("トリミング", "トリミング範囲が小さすぎます。", parent=dialog)
                return

            base_directory = get_gui_base_directory()
            crop_dir = base_directory / "trim_img"
            crop_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            crop_path = crop_dir / f"drawing_trim_{timestamp}.png"

            cropped = source_image.crop((x1, y1, x2, y2))
            cropped.save(crop_path)

            self.cropped_drawing_image_path = str(crop_path)
            self.drawing_crop_box = rect
            self.image_sources["図面画像"] = str(crop_path)
            self.image_view_states["図面画像"] = self.get_default_image_view_state()
            self.set_viewer_image_with_view_state(str(crop_path), image_type="図面画像")

            try:
                dialog.grab_release()
            except tk.TclError:
                pass
            dialog.destroy()
            messagebox.showinfo(
                "トリミング",
                f"トリミング画像を作成しました。\n\n保存先:\n{crop_path}",
                parent=self,
            )
            if on_complete is not None:
                on_complete()

        canvas.bind("<Configure>", lambda event: redraw())
        canvas.bind("<ButtonPress-1>", on_left_press)
        canvas.bind("<B1-Motion>", on_left_drag)
        canvas.bind("<ButtonRelease-1>", on_left_release)
        canvas.bind("<Motion>", on_motion)
        canvas.bind("<Leave>", lambda event: canvas.configure(cursor=""))
        canvas.bind("<ButtonPress-3>", on_right_press)
        canvas.bind("<B3-Motion>", on_right_drag)
        canvas.bind("<ButtonRelease-3>", on_right_release)
        canvas.bind("<MouseWheel>", on_mousewheel)
        canvas.bind("<Button-4>", on_mousewheel)
        canvas.bind("<Button-5>", on_mousewheel)

        tk.Button(
            button_row,
            text="範囲クリア",
            width=12,
            command=clear_crop,
            bg="#ffffff",
            relief="solid",
            bd=1,
            font=("Meiryo UI", 10),
        ).pack(side="left")

        tk.Button(
            button_row,
            text="OK",
            width=10,
            command=apply_crop,
            bg="#ffffff",
            relief="solid",
            bd=1,
            font=("Meiryo UI", 10),
        ).pack(side="right", padx=(8, 0))

        def close_dialog():
            try:
                dialog.grab_release()
            except tk.TclError:
                pass
            dialog.destroy()

        tk.Button(
            button_row,
            text="キャンセル",
            width=10,
            command=close_dialog,
            bg="#ffffff",
            relief="solid",
            bd=1,
            font=("Meiryo UI", 10),
        ).pack(side="right")

        dialog.protocol("WM_DELETE_WINDOW", close_dialog)
        dialog.bind("<Escape>", lambda event: close_dialog())
        dialog.update_idletasks()
        redraw()
        if modal:
            dialog.wait_window()
        return dialog

    def draw_numbers_on_drawing(self, parent=None):
        """検査項目番号つき図面画像のサイズに図面画像を合わせ、番号を配置する。"""
        message_parent = parent or self
        if not self.drawing_image_path:
            messagebox.showinfo(
                "番号描画",
                "先に「図面読み込み」で図面画像を選択してください。",
                parent=message_parent,
            )
            return False

        if not self.inspection_drawing_image_path:
            messagebox.showinfo(
                "番号描画",
                "先に「図面読み込み」で検査項目番号つき図面画像を選択してください。",
                parent=message_parent,
            )
            return False

        csv_path = get_gui_base_directory() / "draw_data" / "結合結果.csv"
        if not csv_path.is_file():
            messagebox.showinfo(
                "番号描画",
                f"結合結果.csvが見つかりません。\n先に「番号読み取り結果」を実行してください。\n\n{csv_path}",
                parent=message_parent,
            )
            return False

        try:
            # CSVの中心X・中心Yは検査項目番号つき図面画像を基準にしているため、
            # その画像サイズを取得し、番号のない図面画像を同じ表示サイズへ変換する。
            with Image.open(self.inspection_drawing_image_path) as inspection_image:
                target_width, target_height = inspection_image.size

            # 検査項目番号つき図面画像から、番号のない図面画像へ切り替える。
            # トリミング済み画像がある場合は、元図面ではなくトリミング画像を使用する。
            drawing_source_path = self.cropped_drawing_image_path or self.drawing_image_path
            self.image_sources["図面画像"] = drawing_source_path
            self.image_type_var.set("図面画像")
            self.set_viewer_image_with_view_state(
                drawing_source_path, image_type="図面画像"
            )

            # 元ファイルは変更せず、ビューア上の図面画像/トリミング画像を検査画像と同じサイズにする。
            self.viewer.set_display_size(target_width, target_height)
            self.image_view_display_sizes["図面画像"] = (target_width, target_height)

            # サイズ変換後の図面画像へ番号を配置する。
            self.viewer.load_annotations_from_csv(csv_path)
        except Exception as exc:
            messagebox.showerror(
                "番号描画",
                f"番号の描画に失敗しました。\n\n{exc}",
                parent=message_parent,
            )
            return False
        return True

    def open_number_reading_result_csv(self, parent=None):
        """番号読み取り結果CSVを選択し、赤番号検出結果CSVと結合してdraw_dataへ保存する。"""
        message_parent = parent or self
        read_results_path = filedialog.askopenfilename(
            parent=message_parent,
            title="番号読み取り結果CSVを選択",
            filetypes=[
                ("CSVファイル", "*.csv"),
                ("すべてのファイル", "*.*"),
            ],
        )
        if not read_results_path:
            return False

        try:
            from merge_csv import merge_csv_files

            base_directory = get_gui_base_directory()
            detection_results_path = base_directory / "output" / "赤番号検出結果.csv"
            output_path = base_directory / "draw_data" / "結合結果.csv"

            # 既存の結合結果.csvがある場合は、再出力前に削除する。
            if output_path.exists():
                output_path.unlink()

            merged_path = merge_csv_files(
                read_results_path=read_results_path,
                detection_results_path=detection_results_path,
                output_path=output_path,
            )

        except Exception as exc:
            messagebox.showerror(
                "番号読み取り結果",
                f"番号読み取りに失敗しました。\n\n{exc}",
                parent=message_parent,
            )
            return False

        messagebox.showinfo(
            "番号読み取り結果",
            "番号の読み取りに成功しました。",
            parent=message_parent,
        )
        return True

    def open_drawing_load_dialog(self):
        """2種類の図面画像を選択するモーダルサブ画面を開く。"""
        dialog = tk.Toplevel(self)
        dialog.title("図面読み込み")
        dialog.transient(self)
        dialog.resizable(False, False)
        dialog.configure(bg="#ffffff")
        dialog.grab_set()

        dialog_width = 760
        dialog_height = 260
        number_dialog_height = 330
        self.update_idletasks()
        x = self.winfo_rootx() + max(0, (self.winfo_width() - dialog_width) // 2)
        y = self.winfo_rooty() + max(0, (self.winfo_height() - dialog_height) // 2)
        dialog.geometry(f"{dialog_width}x{dialog_height}+{x}+{y}")

        selected_paths = {
            "drawing": self.drawing_image_path,
            "inspection": self.inspection_drawing_image_path,
        }

        status_drawing = tk.StringVar(value="選択済み" if selected_paths["drawing"] else "未選択")
        status_inspection = tk.StringVar(value="選択済み" if selected_paths["inspection"] else "未選択")

        outer = tk.Frame(
            dialog,
            bg="#ffffff",
            highlightbackground="#1f1f1f",
            highlightthickness=1,
            bd=0,
        )
        outer.pack(fill="both", expand=True, padx=12, pady=12)
        outer.grid_columnconfigure(0, weight=1)
        outer.grid_columnconfigure(1, minsize=180)
        outer.grid_columnconfigure(2, minsize=150)

        label_font = ("Meiryo UI", 13)
        button_font = ("Meiryo UI", 12)
        status_font = ("Meiryo UI", 12)

        tk.Label(
            outer, text="①図面画像", bg="#ffffff", fg=self.text_main, font=label_font
        ).grid(row=0, column=0, sticky="w", padx=(34, 20), pady=(48, 14))

        tk.Label(
            outer,
            text="②検査項目番号つき図面画像",
            bg="#ffffff",
            fg=self.text_main,
            font=label_font,
        ).grid(row=1, column=0, sticky="w", padx=(34, 20), pady=8)

        drawing_status_label = tk.Label(
            outer, textvariable=status_drawing, bg="#ffffff", font=status_font
        )
        inspection_status_label = tk.Label(
            outer, textvariable=status_inspection, bg="#ffffff", font=status_font
        )

        def refresh_status():
            drawing_selected = bool(selected_paths["drawing"])
            inspection_selected = bool(selected_paths["inspection"])

            status_drawing.set("選択済み" if drawing_selected else "未選択")
            status_inspection.set("選択済み" if inspection_selected else "未選択")
            drawing_status_label.configure(
                fg=self.text_main if drawing_selected else "#0078D7"
            )
            inspection_status_label.configure(
                fg=self.text_main if inspection_selected else "#0078D7"
            )
            ok_button.configure(
                state="normal" if drawing_selected and inspection_selected else "disabled"
            )

        def select_image(key, title):
            path = filedialog.askopenfilename(
                parent=dialog,
                title=title,
                filetypes=[
                    ("画像ファイル", "*.png *.jpg *.jpeg *.bmp *.gif *.webp *.tif *.tiff"),
                    ("すべてのファイル", "*.*"),
                ],
            )
            if not path:
                return

            try:
                with Image.open(path) as image:
                    image.verify()
            except Exception as exc:
                messagebox.showerror(
                    "画像読み込みエラー",
                    f"画像ファイルを読み込めませんでした。\n\n{exc}",
                    parent=dialog,
                )
                return

            selected_paths[key] = path
            if key == "drawing":
                # 図面画像を選び直した場合、以前のトリミング結果は無効にする。
                self.cropped_drawing_image_path = None
                self.drawing_crop_box = None
            refresh_status()

        common_button = {
            "font": button_font,
            "relief": "solid",
            "bd": 1,
            "bg": "#ffffff",
            "activebackground": "#e5e5e5",
            "fg": self.text_main,
            "width": 12,
            "pady": 4,
        }

        def show_number_position_step():
            """サブ画面を閉じずに、番号座標CSV作成ステップへ切り替える。"""
            for widget in outer.winfo_children():
                widget.destroy()

            dialog.title("番号座標CSV作成")
            self.update_idletasks()
            number_y = self.winfo_rooty() + max(0, (self.winfo_height() - number_dialog_height) // 2)
            dialog.geometry(f"{dialog_width}x{number_dialog_height}+{x}+{number_y}")
            dialog.update_idletasks()

            for col in range(3):
                outer.grid_columnconfigure(col, weight=1 if col == 0 else 0)
            for row in range(5):
                outer.grid_rowconfigure(row, weight=1 if row == 3 else 0)

            output_directory = get_gui_base_directory() / "output"
            csv_output_path = output_directory / "赤番号検出結果.csv"

            title_text = tk.StringVar(value="検査項目番号の検出処理を開始します")
            title_label = tk.Label(
                outer,
                textvariable=title_text,
                bg="#ffffff",
                fg=self.text_main,
                font=("Meiryo UI", 13),
                justify="left",
                wraplength=680,
            )
            title_label.grid(
                row=0, column=0, columnspan=3,
                sticky="w", padx=48, pady=(40, 10),
            )

            status_text = tk.StringVar(value="")
            status_label = tk.Label(
                outer,
                textvariable=status_text,
                bg="#ffffff",
                fg="#0078D7",
                font=("Meiryo UI", 11),
                justify="left",
                wraplength=680,
            )
            status_label.grid(
                row=1, column=0, columnspan=3,
                sticky="w", padx=48, pady=(4, 6),
            )

            detail_text = tk.StringVar(value="")
            detail_label = tk.Label(
                outer,
                textvariable=detail_text,
                bg="#ffffff",
                fg=self.text_main,
                font=("Meiryo UI", 9),
                justify="left",
                wraplength=780,
            )
            detail_label.grid(
                row=2, column=0, columnspan=3,
                sticky="w", padx=48, pady=(0, 8),
            )

            spacer = tk.Frame(outer, bg="#ffffff")
            spacer.grid(row=3, column=0, columnspan=3, sticky="nsew")

            button_area = tk.Frame(outer, bg="#ffffff")
            button_area.grid(
                row=4, column=0, columnspan=3,
                sticky="e", padx=18, pady=(0, 22),
            )

            ok_process_button = tk.Button(
                button_area,
                text="OK",
                **common_button,
            )
            ok_process_button.pack(side="left", padx=(0, 10))

            cancel_process_button = tk.Button(
                button_area,
                text="キャンセル",
                command=dialog.destroy,
                **common_button,
            )
            cancel_process_button.pack(side="left")

            def run_processing():
                ok_process_button.configure(state="disabled")
                cancel_process_button.configure(state="disabled")
                title_text.set("検査項目番号を検出処理中です")
                status_text.set("処理中です。しばらくお待ちください.")
                status_label.configure(fg="#0078D7")
                detail_text.set(
                    f"入力画像: {selected_paths['inspection']}\n"
                    f"保存先: {csv_output_path}"
                )
                dialog.update_idletasks()

                try:
                    result = process_number_position_image(selected_paths["inspection"])
                except Exception as exc:
                    title_text.set("検査項目番号の検出処理に失敗しました")
                    status_text.set("処理に失敗しました。")
                    status_label.configure(fg="#D13438")
                    ok_process_button.configure(text="再実行", state="normal")
                    cancel_process_button.configure(text="閉じる", state="normal")
                    messagebox.showerror(
                        "番号座標CSV作成エラー",
                        f"番号座標CSVの作成に失敗しました。\n\n{exc}",
                        parent=dialog,
                    )
                    return

                self.number_position_csv_path = str(result["csv_output_path"])
                self.number_position_result_image_path = str(result["result_output_path"])
                self.number_position_mask_image_path = str(result["mask_output_path"])

                self.update_image_sources_after_number_detection(selected_paths, result)
                self.image_type_var.set("検査項目番号つき図面画像")
                self.set_viewer_image_with_view_state(
                    selected_paths["inspection"], image_type="検査項目番号つき図面画像"
                )

                title_text.set("検査項目番号の検出処理が完了しました")
                status_text.set(
                    f"検出数: {result['count']}\n"
                )
                status_label.configure(fg=self.text_main)
                ok_process_button.configure(text="閉じる", command=dialog.destroy, state="normal")
                cancel_process_button.configure(state="disabled")

            ok_process_button.configure(command=run_processing)

        def apply_selection():
            self.drawing_image_path = selected_paths["drawing"]
            self.inspection_drawing_image_path = selected_paths["inspection"]

            # 図面選択直後はメイン画面へ画像を表示しない。
            # 番号検出処理が完了してから、②検査項目番号つき図面画像を表示する。
            show_number_position_step()

        tk.Button(
            outer,
            text="選択",
            command=lambda: select_image("drawing", "①図面画像を選択"),
            **common_button,
        ).grid(row=0, column=1, padx=10, pady=(48, 14))

        tk.Button(
            outer,
            text="選択",
            command=lambda: select_image(
                "inspection", "②検査項目番号つき図面画像を選択"
            ),
            **common_button,
        ).grid(row=1, column=1, padx=10, pady=8)

        drawing_status_label.grid(row=0, column=2, sticky="w", padx=(16, 24), pady=(48, 14))
        inspection_status_label.grid(row=1, column=2, sticky="w", padx=(16, 24), pady=8)

        button_area = tk.Frame(outer, bg="#ffffff")
        button_area.grid(row=2, column=0, columnspan=3, sticky="e", padx=18, pady=(18, 20))

        ok_button = tk.Button(
            button_area, text="次へ", command=apply_selection, **common_button
        )
        ok_button.pack(side="left", padx=(0, 10))

        tk.Button(
            button_area, text="キャンセル", command=dialog.destroy, **common_button
        ).pack(side="left")

        refresh_status()
        dialog.protocol("WM_DELETE_WINDOW", dialog.destroy)
        dialog.bind("<Escape>", lambda event: dialog.destroy())
        dialog.wait_window()

    def open_image(self):
        """互換用。図面読み込みダイアログを開く。"""
        self.open_drawing_load_dialog()

    def get_selected_image_view_type(self):
        if not hasattr(self, "image_type_combo"):
            return "図面画像"

        selected_type = self.image_type_combo.get()
        if selected_type in self.image_view_types:
            return selected_type
        return "図面画像"

    def get_default_image_view_state(self):
        canvas = getattr(self.viewer, "canvas", None)
        canvas_w = canvas.winfo_width() if canvas is not None else 0
        canvas_h = canvas.winfo_height() if canvas is not None else 0

        if canvas_w <= 1:
            canvas_w = int(getattr(self.viewer, "width", 800))
        if canvas_h <= 1:
            canvas_h = int(getattr(self.viewer, "height", 500))

        return {
            "scale": 1.0,
            "offset_x": canvas_w // 2,
            "offset_y": canvas_h // 2,
        }

    def save_current_image_view_state(self, image_type=None):
        if not hasattr(self, "viewer"):
            return

        image_type = image_type or self.current_image_view_type
        if image_type not in self.image_view_types:
            return

        self.image_view_states[image_type] = {
            "scale": float(getattr(self.viewer, "scale", 1.0)),
            "offset_x": float(getattr(self.viewer, "offset_x", 0.0)),
            "offset_y": float(getattr(self.viewer, "offset_y", 0.0)),
        }

        if getattr(self.viewer, "_pil_image", None) is not None:
            display_size = self.viewer.get_display_size()
            if display_size is not None:
                self.image_view_display_sizes[image_type] = display_size

    def restore_image_view_state(self, image_type):
        if not hasattr(self, "viewer"):
            return

        state = self.image_view_states.get(image_type)
        if state is None:
            state = self.get_default_image_view_state()
            self.image_view_states[image_type] = state.copy()

        self.viewer.scale = max(0.1, min(float(state["scale"]), 10.0))
        self.viewer.offset_x = float(state["offset_x"])
        self.viewer.offset_y = float(state["offset_y"])
        self.viewer.redraw_image()
        self.viewer.redraw_rectangle()

    def set_viewer_image_with_view_state(self, image_source, image_type=None):
        next_image_type = image_type or self.get_selected_image_view_type()
        if next_image_type not in self.image_view_types:
            next_image_type = "図面画像"

        self.save_current_image_view_state()
        self.viewer.set_image(image_source)

        # set_image() で display_width/display_height は元画像サイズへ戻る。
        # 番号描画後の図面画像は、CSV座標と同じ検査画像サイズで表示する必要があるため、
        # 保存済みの表示サイズを復元する。ただし番号座標はすでに検査画像基準なので再スケールしない。
        display_size = self.image_view_display_sizes.get(next_image_type)
        if display_size is not None:
            self.viewer.set_display_size(
                display_size[0],
                display_size[1],
                scale_annotations=False,
            )

        self.viewer.set_annotation_visibility(
            self.should_show_annotations_for_image_type(next_image_type)
        )
        self.current_image_view_type = next_image_type
        self.image_type_var.set(next_image_type)
        self.restore_image_view_state(next_image_type)


if __name__ == "__main__":
    app = ImageDisplayAreaOnly()
    app.mainloop()
