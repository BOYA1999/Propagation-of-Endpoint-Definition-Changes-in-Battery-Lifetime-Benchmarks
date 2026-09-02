import argparse
import json
import re
from pathlib import Path

from PIL import Image
from pypdf import PdfReader


def main():
    parser = argparse.ArgumentParser(description="Validate the five exported paper figures.")
    parser.add_argument("--figures", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    figures = args.figures
    manifest = json.loads((figures / "figure_manifest_v12.json").read_text(encoding="utf-8"))
    pngs = [figures / f"Fig{number}.png" for number in range(1, 6)]
    svgs = [figures / f"Fig{number}.svg" for number in range(1, 6)]
    pdfs = [figures / f"Fig{number}.pdf" for number in range(1, 6)]
    eps_files = [figures / f"Fig{number}.eps" for number in range(1, 6)]
    svg_text = "\n".join(path.read_text(encoding="utf-8", errors="ignore") for path in svgs)
    forbidden_svg = re.compile(r"[A-Za-z]:\\" + "|" + "/" + "Users/" + "|" + "/" + "home/" + r"|Figure\s+[1-5]:")
    checks = {
        "five_complete_export_sets": all(path.is_file() for path in pngs + svgs + pdfs + eps_files),
        "editable_svg_text": all("<text" in path.read_text(encoding="utf-8", errors="ignore") for path in svgs),
        "pdfs_readable": all(len(PdfReader(path).pages) == 1 for path in pdfs),
        "eps_headers_valid": all(path.read_bytes().startswith(b"%!PS-Adobe") for path in eps_files),
        "pngs_are_high_resolution": all(Image.open(path).width == 4080 for path in pngs),
        "panel_labels_complete": all(all(f">{letter}<" in (figures / f"Fig{number}.svg").read_text(encoding="utf-8", errors="ignore") for letter in letters) for number, letters in {1: "ab", 2: "abcd", 3: "abc", 4: "ab", 5: "abc"}.items()),
        "no_figure_titles_or_local_paths": not forbidden_svg.search(svg_text),
        "eight_source_tables_present": len(list(args.data.glob("*.csv"))) == 8,
        "manifest_has_twenty_outputs": manifest["status"] == "PASS" and len(manifest["figure_files"]) == 20,
        "catalogue_has_five_claims": len(json.loads((figures / "figure_catalog_v12.json").read_text(encoding="utf-8"))) == 5,
    }
    report = {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks, "passed": sum(checks.values()), "total": len(checks)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not all(checks.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
