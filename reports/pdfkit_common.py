"""
Shared ReportLab helpers for the PDF reports: fonts with Greek/math glyphs (DejaVu, shipped
with matplotlib), paragraph styles, tables, figures with captions, page decorations, and
loaders for the saved analysis results.
"""
import os
import json
import numpy as np
import matplotlib
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER, TA_LEFT
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.fonts import addMapping
from reportlab.platypus import (Paragraph, Spacer, Table, TableStyle, Image, KeepTogether,
                                PageBreak, Flowable)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PROC = os.path.join(ROOT, "data", "processed")
FIG = os.path.join(ROOT, "docs", "figures")
DOCS = os.path.join(ROOT, "docs")

# ---------------------------------------------------------------------------
# Fonts
# ---------------------------------------------------------------------------
_TTF = os.path.join(os.path.dirname(matplotlib.__file__), "mpl-data", "fonts", "ttf")


def register_fonts():
    for fam, files in {
        "Serif": ("DejaVuSerif.ttf", "DejaVuSerif-Bold.ttf", "DejaVuSerif-Italic.ttf", "DejaVuSerif-BoldItalic.ttf"),
        "Sans": ("DejaVuSans.ttf", "DejaVuSans-Bold.ttf", "DejaVuSans-Oblique.ttf", "DejaVuSans-BoldOblique.ttf"),
    }.items():
        names = [fam, fam + "-Bold", fam + "-Italic", fam + "-BoldItalic"]
        for n, f in zip(names, files):
            pdfmetrics.registerFont(TTFont(n, os.path.join(_TTF, f)))
        addMapping(fam, 0, 0, names[0])
        addMapping(fam, 1, 0, names[1])
        addMapping(fam, 0, 1, names[2])
        addMapping(fam, 1, 1, names[3])
    pdfmetrics.registerFont(TTFont("Mono", os.path.join(_TTF, "DejaVuSansMono.ttf")))


register_fonts()

# ---------------------------------------------------------------------------
# Styles
# ---------------------------------------------------------------------------
INK = colors.HexColor("#1b1f24")
MUTED = colors.HexColor("#5a636e")
ACCENT = colors.HexColor("#1f4e79")
RULE = colors.HexColor("#c9d1da")
SHADE = colors.HexColor("#eef2f6")
GOOD = colors.HexColor("#e3f1e6")
WARN = colors.HexColor("#fdf3df")
BAD = colors.HexColor("#f8e3e1")


def make_styles(base="Serif", size=10.0):
    lead = size * 1.38
    S = {
        'body': ParagraphStyle('body', fontName=base, fontSize=size, leading=lead, textColor=INK,
                               alignment=TA_JUSTIFY, spaceAfter=size * 0.6),
        'bodyL': ParagraphStyle('bodyL', fontName=base, fontSize=size, leading=lead, textColor=INK,
                                alignment=TA_LEFT, spaceAfter=size * 0.6),
        'small': ParagraphStyle('small', fontName=base, fontSize=size - 1.5, leading=(size - 1.5) * 1.35,
                                textColor=INK),
        'caption': ParagraphStyle('caption', fontName=base, fontSize=size - 1.2, leading=(size - 1.2) * 1.38,
                                  textColor=MUTED, alignment=TA_JUSTIFY, spaceBefore=3, spaceAfter=size),
        'h1': ParagraphStyle('h1', fontName=base + "-Bold", fontSize=size + 5, leading=(size + 5) * 1.25,
                             textColor=ACCENT, spaceBefore=size * 1.4, spaceAfter=size * 0.7, keepWithNext=1),
        'h2': ParagraphStyle('h2', fontName=base + "-Bold", fontSize=size + 2, leading=(size + 2) * 1.3,
                             textColor=INK, spaceBefore=size * 1.0, spaceAfter=size * 0.45, keepWithNext=1),
        'h3': ParagraphStyle('h3', fontName=base + "-BoldItalic", fontSize=size, leading=lead,
                             textColor=INK, spaceBefore=size * 0.6, spaceAfter=size * 0.3, keepWithNext=1),
        'title': ParagraphStyle('title', fontName=base + "-Bold", fontSize=size + 11, leading=(size + 11) * 1.22,
                                textColor=INK, alignment=TA_CENTER, spaceAfter=size),
        'subtitle': ParagraphStyle('subtitle', fontName=base, fontSize=size + 2, leading=(size + 2) * 1.35,
                                   textColor=MUTED, alignment=TA_CENTER, spaceAfter=size * 0.5),
        'abstract': ParagraphStyle('abstract', fontName=base, fontSize=size - 0.5, leading=(size - 0.5) * 1.4,
                                   textColor=INK, alignment=TA_JUSTIFY, leftIndent=12 * mm, rightIndent=12 * mm),
        'cell': ParagraphStyle('cell', fontName=base, fontSize=size - 1.8, leading=(size - 1.8) * 1.3, textColor=INK),
        'cellB': ParagraphStyle('cellB', fontName=base + "-Bold", fontSize=size - 1.8, leading=(size - 1.8) * 1.3,
                                textColor=INK),
        'bullet': ParagraphStyle('bullet', fontName=base, fontSize=size, leading=lead, textColor=INK,
                                 leftIndent=12, bulletIndent=2, spaceAfter=size * 0.3, alignment=TA_LEFT),
        'box': ParagraphStyle('box', fontName=base, fontSize=size, leading=lead, textColor=INK, alignment=TA_LEFT),
        'eq': ParagraphStyle('eq', fontName=base, fontSize=size + 0.5, leading=(size + 0.5) * 1.5, textColor=INK,
                             alignment=TA_CENTER, spaceBefore=4, spaceAfter=8),
        'ref': ParagraphStyle('ref', fontName=base, fontSize=size - 1.5, leading=(size - 1.5) * 1.35,
                              textColor=INK, leftIndent=14, firstLineIndent=-14, spaceAfter=2),
    }
    return S


# ---------------------------------------------------------------------------
# Flowable helpers
# ---------------------------------------------------------------------------
def bullets(items, S, style='bullet'):
    return [Paragraph(it, S[style], bulletText='•') for it in items]


def figure(path, width, caption, S, max_height=None):
    from reportlab.lib.utils import ImageReader
    iw, ih = ImageReader(path).getSize()
    h = width * ih / iw
    if max_height and h > max_height:
        width, h = width * max_height / h, max_height
    img = Image(path, width=width, height=h)
    return KeepTogether([img, Paragraph(caption, S['caption'])])


def table(rows, S, col_widths, header_rows=1, zebra=True, align_right_from=None, highlight=None):
    """rows: list of lists of str (converted to Paragraphs). highlight: {row_index: color}."""
    data = []
    for i, r in enumerate(rows):
        st = S['cellB'] if i < header_rows else S['cell']
        data.append([c if isinstance(c, Flowable) else Paragraph(str(c), st) for c in r])
    t = Table(data, colWidths=col_widths, repeatRows=header_rows)
    cmds = [
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('LINEABOVE', (0, 0), (-1, 0), 0.9, INK),
        ('LINEBELOW', (0, header_rows - 1), (-1, header_rows - 1), 0.6, INK),
        ('LINEBELOW', (0, -1), (-1, -1), 0.9, INK),
        ('TOPPADDING', (0, 0), (-1, -1), 2.5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2.5),
        ('LEFTPADDING', (0, 0), (-1, -1), 3),
        ('RIGHTPADDING', (0, 0), (-1, -1), 3),
    ]
    if zebra:
        for i in range(header_rows, len(rows)):
            if (i - header_rows) % 2 == 1:
                cmds.append(('BACKGROUND', (0, i), (-1, i), SHADE))
    for i, c in (highlight or {}).items():
        cmds.append(('BACKGROUND', (0, i), (-1, i), c))
    t.setStyle(TableStyle(cmds))
    return t


def callout(text_flowables, S, fill=SHADE, border=ACCENT, width=None):
    t = Table([[text_flowables]], colWidths=[width] if width else None)
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), fill),
        ('LINEBEFORE', (0, 0), (0, -1), 2.2, border),
        ('LEFTPADDING', (0, 0), (-1, -1), 9), ('RIGHTPADDING', (0, 0), (-1, -1), 9),
        ('TOPPADDING', (0, 0), (-1, -1), 7), ('BOTTOMPADDING', (0, 0), (-1, -1), 7),
    ]))
    return t


def page_decorator(header_text, base="Serif"):
    def _draw(canvas, doc):
        canvas.saveState()
        w, h = A4
        canvas.setFont(base, 7.5)
        canvas.setFillColor(MUTED)
        if doc.page > 1:
            canvas.drawString(18 * mm, h - 11 * mm, header_text)
            canvas.setStrokeColor(RULE)
            canvas.setLineWidth(0.4)
            canvas.line(18 * mm, h - 12.5 * mm, w - 18 * mm, h - 12.5 * mm)
        canvas.drawRightString(w - 18 * mm, 10 * mm, f"{doc.page}")
        canvas.restoreState()
    return _draw


# ---------------------------------------------------------------------------
# Results access
# ---------------------------------------------------------------------------
def load(name):
    with open(os.path.join(PROC, name)) as fh:
        return json.load(fh)


def fnum(x, nd=2):
    if x is None:
        return "n/a"
    try:
        x = float(x)
    except (TypeError, ValueError):
        return str(x)
    if not np.isfinite(x):
        return "none"
    return f"{x:.{nd}f}"


def sci(x, nd=2):
    """Scientific notation with a proper superscript, e.g. -2.22×10<super>−5</super>."""
    x = float(x)
    if x == 0 or not np.isfinite(x):
        return fnum(x, nd)
    e = int(np.floor(np.log10(abs(x))))
    m = x / 10 ** e
    sign = "−" if m < 0 else ""
    return f"{sign}{abs(m):.{nd}f}×10<super>{'−' if e < 0 else ''}{abs(e)}</super>"


def r2fmt(x):
    if x is None or not np.isfinite(float(x)):
        return "n/a"
    return "< 0" if float(x) < 0 else f"{float(x):.2f}"


def pct(x, nd=0):
    if x is None or not np.isfinite(float(x)):
        return "n/a"
    return f"{100 * float(x):.{nd}f}%"


def abs_err(r):
    v = r['t_arr_rel_err']
    if v is None:
        return 1.0
    v = float(v)
    return 1.0 if not np.isfinite(v) else min(abs(v), 1.0)


def boot_ci_median(x, n=5000, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float)
    b = [np.median(rng.choice(x, len(x))) for _ in range(n)]
    return float(np.median(x)), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def forecast_summary():
    """Per-model metrics for protocol (a) and (b), with bootstrap CIs over forces."""
    F = load("forecast_results.json")
    A = F['protocol_a']
    forces = sorted(A.keys(), key=float)
    arrested = [k for k in forces if A[k][next(iter(A[k]))]['t_arr_true'] is not None
                and np.isfinite(float(A[k][next(iter(A[k]))]['t_arr_true']))]
    models_a = [m for m in ('NeighbourBaseline', 'HDMD-param', 'DeepKoopman', 'MasterCurve', 'ODE-B') if m in A[forces[0]]]
    out_a = {}
    for m in models_a:
        e = [abs_err(A[k][m]) for k in arrested]
        md, lo, hi = boot_ci_median(e)
        rm = [A[k][m]['rmse_slow'] for k in arrested if A[k][m]['rmse_slow'] is not None]
        ma = [A[k][m].get('mae_slow') for k in arrested if A[k][m].get('mae_slow') is not None]
        r2 = [A[k][m].get('r2_slow') for k in arrested if A[k][m].get('r2_slow') is not None]
        missed = sum(1 for k in arrested if A[k][m]['t_arr_pred'] is None or not np.isfinite(float(A[k][m]['t_arr_pred'])))
        nonarr = [k for k in forces if k not in arrested]
        false = sum(1 for k in nonarr if A[k][m]['t_arr_pred'] is not None and np.isfinite(float(A[k][m]['t_arr_pred'])))
        within25 = sum(1 for x in e if x <= 0.25)
        out_a[m] = dict(med=md, lo=lo, hi=hi, rmse=float(np.median(rm)) if rm else np.nan,
                        mae=float(np.median(ma)) if ma else np.nan, r2=float(np.median(r2)) if r2 else np.nan,
                        missed=missed, false=false, n=len(arrested), n_fluid=len(nonarr), within25=within25,
                        per_force={k: A[k][m] for k in forces})
    B = F['protocol_b']
    ps = sorted(next(iter(B.values())).keys(), key=float)
    models_b = [m for m in ('HDMD-prefix', 'DeepKoopman', 'MasterCurve', 'ODE-B')]
    out_b = {}
    for m in models_b:
        out_b[m] = {}
        for p in ps:
            e = [abs_err(B[k][p][m]) for k in arrested]
            md, lo, hi = boot_ci_median(e)
            miss = sum(1 for k in arrested if B[k][p][m]['t_arr_pred'] is None
                       or not np.isfinite(float(B[k][p][m]['t_arr_pred'])))
            rm = [B[k][p][m]['rmse_slow'] for k in arrested if B[k][p][m]['rmse_slow'] is not None]
            out_b[m][p] = dict(med=md, lo=lo, hi=hi, missed=miss, rmse=float(np.median(rm)) if rm else np.nan)
    orig = F.get('original_model', {})
    return dict(a=out_a, b=out_b, prefixes=ps, forces=forces, arrested=arrested, original=orig)
