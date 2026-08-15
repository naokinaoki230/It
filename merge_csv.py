from pathlib import Path
import csv


OUTPUT_COLUMNS = [
    "検出順",
    "読み取り結果",
    "中心X",
    "中心Y",
    "外接矩形X",
    "外接矩形Y",
    "幅",
    "高さ",
    "輪郭面積",
]


def merge_csv_files(
    read_results_path: str | Path,
    detection_results_path: str | Path,
    output_path: str | Path,
) -> Path:
    """
    番号読み取り結果CSVと赤番号検出結果CSVを「No.」/「検出順」で結合する。

    Parameters
    ----------
    read_results_path:
        ダイアログで選択した番号読み取り結果CSV。
        「No.」「読み取り結果」列を想定。
    detection_results_path:
        outputフォルダ内の「赤番号検出結果.csv」。
        「検出順」「中心X」「中心Y」「外接矩形X」「外接矩形Y」「幅」「高さ」「輪郭面積」列を想定。
    output_path:
        結合後CSVの保存先。
    """
    read_results_path = Path(read_results_path)
    detection_results_path = Path(detection_results_path)
    output_path = Path(output_path)

    if not read_results_path.is_file():
        raise FileNotFoundError(f"番号読み取り結果CSVが見つかりません: {read_results_path}")
    if not detection_results_path.is_file():
        raise FileNotFoundError(f"赤番号検出結果CSVが見つかりません: {detection_results_path}")

    # 赤番号検出結果を「検出順」をキーにして辞書へ格納
    with detection_results_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None or "検出順" not in reader.fieldnames:
            raise ValueError("赤番号検出結果CSVに「検出順」列がありません。")

        detection_by_order = {
            row.get("検出順", "").strip(): row
            for row in reader
            if row.get("検出順", "").strip()
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with read_results_path.open("r", encoding="utf-8-sig", newline="") as input_file, output_path.open(
        "w", encoding="utf-8-sig", newline=""
    ) as output_file:
        reader = csv.DictReader(input_file)
        if reader.fieldnames is None:
            raise ValueError("番号読み取り結果CSVのヘッダー行を読み取れません。")
        if "No." not in reader.fieldnames:
            raise ValueError("番号読み取り結果CSVに「No.」列がありません。")
        if "読み取り結果" not in reader.fieldnames:
            raise ValueError("番号読み取り結果CSVに「読み取り結果」列がありません。")

        writer = csv.DictWriter(output_file, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()

        for read_row in reader:
            # read_results側の「No.」を結合キーとして使用
            order = read_row.get("No.", "").strip()
            detection_row = detection_by_order.get(order, {})

            writer.writerow(
                {
                    "検出順": order,
                    "読み取り結果": read_row.get("読み取り結果", ""),
                    "中心X": detection_row.get("中心X", ""),
                    "中心Y": detection_row.get("中心Y", ""),
                    "外接矩形X": detection_row.get("外接矩形X", ""),
                    "外接矩形Y": detection_row.get("外接矩形Y", ""),
                    "幅": detection_row.get("幅", ""),
                    "高さ": detection_row.get("高さ", ""),
                    "輪郭面積": detection_row.get("輪郭面積", ""),
                }
            )

    return output_path


def main() -> None:
    # 単体実行用の例。GUIからは merge_csv_files(...) を呼び出します。
    input_dir = Path(__file__).resolve().parent
    read_results_path = input_dir / "read_results.csv"
    detection_results_path = input_dir / "output" / "赤番号検出結果.csv"
    output_path = input_dir / "draw_data" / "結合結果.csv"

    merged_path = merge_csv_files(
        read_results_path=read_results_path,
        detection_results_path=detection_results_path,
        output_path=output_path,
    )
    print(f"結合が完了しました: {merged_path}")


if __name__ == "__main__":
    main()