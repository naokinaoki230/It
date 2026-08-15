import pymupdf
from pathlib import Path

pdf_path = "図面/元データ/FAM0195048_FAM0195289_merged.pdf"
output_dir = Path("output")
output_dir.mkdir(exist_ok=True)

doc = pymupdf.open(pdf_path)

for i, page in enumerate(doc, start=1):
    pix = page.get_pixmap(dpi=200)
    pix.save(output_dir / f"page_{i:03}.png")

doc.close()