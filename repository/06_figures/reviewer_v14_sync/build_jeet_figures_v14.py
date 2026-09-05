import hashlib
import argparse
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser(description="Build five figures using the current layouts and derived evidence.")
parser.add_argument('--data-root', type=Path, required=True)
parser.add_argument('--analysis-root', type=Path, help='Use freshly recomputed controlled-analysis tables.')
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
OUT = args.output.resolve()
SOURCE = ROOT / '06_figures/reviewer_v13_sync/build_jeet_figures_v13.py'
spec = importlib.util.spec_from_file_location('retained_figures', SOURCE)
engine = importlib.util.module_from_spec(spec)
spec.loader.exec_module(engine)
engine.configure(args.data_root, OUT, args.analysis_root)
engine.VERSION = 'v14'
original_save = engine.save_figure
receipt = {'source_script': SOURCE.relative_to(ROOT).as_posix(), 'source_sha256': hashlib.sha256(SOURCE.read_bytes()).hexdigest(), 'layout': 'Minimum font 8 pt at 165.1 mm placement width', 'figures': {}, 'source_table_checks': {}}

def save(fig, number, conclusion):
    if number == 1:
        for patch in fig.axes[1].patches:
            patch.set_clip_on(False)
        for text in fig.axes[0].texts:
            if text.get_text().startswith('integer, one-based position'):
                text.set_text('integer, one-based position on the retained\nprocessed ordinary-cycle sequence')
                text.set_y(0.15)
        for patch in fig.axes[0].patches:
            if isinstance(patch, engine.FancyBboxPatch) and abs(patch.get_width() - 0.52) < 0.001:
                patch.set_y(0.075)
                patch.set_height(0.25)
        for text in fig.axes[0].texts:
            if text.get_text() == 'Shared benchmark coordinate':
                text.set_y(0.26)
    if number == 3:
        fig.axes[1].get_legend().set_loc('lower right')
    if number == 4:
        box = fig.axes[0].get_position()
        fig.axes[0].set_position([box.x0 + 0.025, box.y0, box.width - 0.025, box.height])
    if number == 5:
        for text in fig.axes[0].texts:
            if 'A' in text.get_text() and 'B' in text.get_text():
                text.set_position((0.04, 0.95))
                text.set_verticalalignment('top')
    texts = fig.findobj(engine.matplotlib.text.Text)
    for text in texts:
        if text.get_text():
            text.set_fontsize(max(text.get_fontsize(), 8.4))
    receipt['figures'][str(number)] = {'width_in': fig.get_figwidth(), 'min_export_pt': min(t.get_fontsize() for t in texts if t.get_text()), 'min_word_pt': min(t.get_fontsize() for t in texts if t.get_text()) * 6.5 / fig.get_figwidth()}
    original_save(fig, number, conclusion)

engine.save_figure = save
for number in range(1, 6):
    getattr(engine, f'figure{number}')()
    for path in engine.DATA.glob(f'Fig{number}*.csv'):
        old = args.data_root / '06_figures/reviewer_v14_sync/figure_data' / path.name
        try:
            engine.pd.testing.assert_frame_equal(engine.pd.read_csv(old), engine.pd.read_csv(path), check_exact=False, rtol=1e-12, atol=1e-12)
            receipt['source_table_checks'][path.name] = True
        except AssertionError:
            receipt['source_table_checks'][path.name] = False
            raise
engine.write_metadata()
receipt['files_sha256'] = {p.relative_to(OUT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(OUT.rglob('*')) if p.is_file() and p.suffix in {'.csv', '.svg', '.pdf', '.eps', '.png', '.txt'}}
receipt['status'] = 'SOURCE_TABLES_MATCH'
(OUT / 'figure_layout_receipt_v14.json').write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
print(OUT)
