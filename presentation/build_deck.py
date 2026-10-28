"""
Builds the conference talk on the Canva template (White and Black Modern ... .pptx):
presentation/B9_Active_Glass_Talk.pptx

The template's own chrome (left rule, logo mark, brand and header labels), fonts
(Open Sauce Bold / Open Sans, embedded in the template) and line/frame style are reused;
content slides are assembled from native, editable shapes plus the slide figures made by
src/slide_figures.py. All numbers are read from data/processed/*.json.

Run from the project root:  python presentation/build_deck.py
"""
import os
import re
import sys
import copy
import json
import numpy as np
from lxml import etree
from pptx import Presentation
from pptx.util import Emu, Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE, MSO_CONNECTOR
from pptx.oxml.ns import qn

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "reports"))
from pdfkit_common import forecast_summary, pct  # noqa: E402

TEMPLATE = os.path.join(ROOT, "White and Black Modern Company Research Proposal Presentation.pptx")
OUT = os.path.join(ROOT, "presentation", "B9_Active_Glass_Talk.pptx")
FIG = os.path.join(ROOT, "presentation", "figures")
PROC = os.path.join(ROOT, "data", "processed")

INK, MID, LIGHT, ACCENT = "424242", "8A8A8A", "D9D9D9", "C0502E"
GREEN, AMBER, RED = "4E8A5A", "D39B2A", "B8432F"
TITLE_FONT, BODY, BODY_B, BODY_I = "Open Sauce Bold", "Open Sans 1", "Open Sans 1 Bold", "Open Sans 1 Italics"
HEADER = "Decoding Dense Active Fluids"
TEAM = "Team B-9"
MEMBERS = "Rithvik Arulprakash  ·  Harshith KV  ·  Vipin Sudhakar  ·  Vijay Murugan"
REPO = "github.com/Rithvik188/Compflu"
X0, X1 = 1.74, 18.2          # content margins (inches)


def E(x):
    return int(round(x * 914400))


def load(name):
    with open(os.path.join(PROC, name)) as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# Low-level shape helpers (template style)
# ---------------------------------------------------------------------------
def _strip_style(shape):
    st = shape._element.find(qn('p:style'))
    if st is not None:
        shape._element.remove(st)


def _ln(shape, color=INK, w=1.5, dash=None, head=False, tail=False):
    ln = shape.line
    ln.color.rgb = RGBColor.from_string(color)
    ln.width = Pt(w)
    lnel = shape._element.spPr.find(qn('a:ln'))
    lnel.set('cap', 'sq')
    if dash:
        d = etree.SubElement(lnel, qn('a:prstDash'))
        d.set('val', dash)
    for tag, on in (('a:headEnd', head), ('a:tailEnd', tail)):
        if on:
            e = etree.SubElement(lnel, qn(tag))
            e.set('type', 'triangle')
            e.set('w', 'med')
            e.set('len', 'med')


def rule(slide, x1, y1, x2, y2, color=INK, w=1.5, dash=None, arrow=False):
    c = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, E(x1), E(y1), E(x2), E(y2))
    _strip_style(c)
    _ln(c, color, w, dash, tail=arrow)
    return c


def box(slide, x, y, w, h, line=INK, fill=None, lw=1.5, shape=MSO_SHAPE.RECTANGLE, dash=None):
    s = slide.shapes.add_shape(shape, E(x), E(y), E(w), E(h))
    _strip_style(s)
    if fill:
        s.fill.solid()
        s.fill.fore_color.rgb = RGBColor.from_string(fill)
    else:
        s.fill.background()
    if line:
        _ln(s, line, lw, dash)
    else:
        s.line.fill.background()
    if s.has_text_frame:
        s.text_frame.text = ""
    return s


TOKEN = re.compile(r'(\*\*.+?\*\*|\{a\}.+?\{/a\}|\{i\}.+?\{/i\}|\{m\}.+?\{/m\}|_\{[^{}]+?\}|\^\{[^{}]+?\})')


def _runs(text, base=None):
    """Mini markup: **bold**, _{sub}, ^{sup}, {a}accent{/a}, {i}italic{/i}, {m}muted{/m}.
    Styled segments may contain sub/superscripts (handled recursively)."""
    base = base or {}
    out = []
    for part in TOKEN.split(text):
        if not part:
            continue
        if part.startswith('**'):
            out += _runs(part[2:-2], {**base, 'bold': True})
        elif part.startswith('_{'):
            out.append((part[2:-1], {**base, 'baseline': -25000}))
        elif part.startswith('^{'):
            out.append((part[2:-1], {**base, 'baseline': 30000}))
        elif part.startswith('{a}'):
            out += _runs(part[3:-4], {**base, 'color': ACCENT, 'bold': True})
        elif part.startswith('{i}'):
            out += _runs(part[3:-4], {**base, 'italic': True})
        elif part.startswith('{m}'):
            out += _runs(part[3:-4], {**base, 'color': MID})
        else:
            out.append((part, dict(base)))
    return out


def text(slide, x, y, w, h, paras, size=20, font=BODY, color=INK, align='l', anchor='t', spacing=1.25,
         space_after=0.35, title=False, autofit=False):
    """paras: str or list of str (each a paragraph, markup allowed). Returns the shape."""
    tb = slide.shapes.add_textbox(E(x), E(y), E(w), E(h))
    body = tb.text_frame._txBody
    bp = body.find(qn('a:bodyPr'))
    for k in ('lIns', 'tIns', 'rIns', 'bIns'):
        bp.set(k, '0')
    bp.set('wrap', 'square')
    bp.set('anchor', anchor)
    if autofit:
        etree.SubElement(bp, qn('a:normAutofit'))
    for p in body.findall(qn('a:p')):
        body.remove(p)
    if isinstance(paras, str):
        paras = [paras]
    spc_scale = -0.07 if title else -0.03
    for i, ptxt in enumerate(paras):
        p = etree.SubElement(body, qn('a:p'))
        ppr = etree.SubElement(p, qn('a:pPr'))
        ppr.set('algn', align)
        ls = etree.SubElement(etree.SubElement(ppr, qn('a:lnSpc')), qn('a:spcPct'))
        ls.set('val', str(int(spacing * 100000)))
        if i < len(paras) - 1 and space_after:
            sa = etree.SubElement(etree.SubElement(ppr, qn('a:spcAft')), qn('a:spcPts'))
            sa.set('val', str(int(space_after * size * 100)))
        for t, st in _runs(ptxt):
            r = etree.SubElement(p, qn('a:r'))
            rpr = etree.SubElement(r, qn('a:rPr'))
            rpr.set('lang', 'en-US')
            rpr.set('sz', str(int(size * 100)))
            rpr.set('spc', str(int(spc_scale * size * 100)))
            fnt = font
            if st.get('bold') and font == BODY:
                fnt = BODY_B
            if st.get('italic') and font == BODY:
                fnt = BODY_I
            if fnt in (TITLE_FONT, BODY_B):
                rpr.set('b', '1')
            if 'baseline' in st:
                rpr.set('baseline', str(st['baseline']))
            sf = etree.SubElement(rpr, qn('a:solidFill'))
            etree.SubElement(sf, qn('a:srgbClr')).set('val', st.get('color', color))
            for tag in ('a:latin', 'a:ea', 'a:cs', 'a:sym'):
                etree.SubElement(rpr, qn(tag)).set('typeface', fnt)
            etree.SubElement(r, qn('a:t')).text = t
    return tb


def picture(slide, name, x, y, w=None, h=None, frame=False):
    path = os.path.join(FIG, name)
    pic = slide.shapes.add_picture(path, E(x), E(y), E(w) if w else None, E(h) if h else None)
    if frame:
        box(slide, Emu(pic.left).inches, Emu(pic.top).inches, Emu(pic.width).inches, Emu(pic.height).inches)
    return pic


def title(slide, t, size=60, y=1.85, w=16.4, rule_y=3.2):
    text(slide, X0, y, w, 1.2, t, size=size, font=TITLE_FONT, spacing=0.9, title=True)
    if rule_y:
        rule(slide, X0, rule_y, X1, rule_y)


def card(slide, x, y, w, h, num, head, body, num_size=36, head_size=22, body_size=17, accent=False):
    box(slide, x, y, w, h, line=ACCENT if accent else INK)
    pad = 0.3
    yy = y + 0.22
    if num:
        text(slide, x + pad, yy, w - 2 * pad, 0.7, num, size=num_size, font=BODY_B, color=ACCENT if accent else INK)
        yy += num_size / 72 * 1.35
    if head:
        text(slide, x + pad, yy, w - 2 * pad, 0.6, head, size=head_size, font=BODY_B)
        yy += head_size / 72 * 1.35
    if body:
        text(slide, x + pad, yy, w - 2 * pad, h - (yy - y) - 0.15, body, size=body_size, font=BODY_I, spacing=1.2)


def grid_table(slide, x, y, col_w, rows, size=17, row_h=0.62, head_size=None, colors=None, fonts=None,
               aligns=None, header=True):
    """Minimal hand-ruled table: bold header, thin rules between rows."""
    head_size = head_size or size
    total_w = sum(col_w)
    rule(slide, x, y, x + total_w, y, w=1.5)
    yy = y + 0.1
    for i, row in enumerate(rows):
        xx = x
        hdr = header and i == 0
        rh = row_h
        for j, cell in enumerate(row):
            col = (colors or {}).get((i, j), INK)
            fnt = BODY_B if hdr else (fonts or {}).get((i, j), BODY)
            al = (aligns or {}).get(j, 'l')
            text(slide, xx + 0.08, yy, col_w[j] - 0.16, rh, cell, size=head_size if hdr else size, font=fnt,
                 color=col, spacing=1.1, align=al)
            xx += col_w[j]
        yy += rh
        rule(slide, x, yy - 0.04, x + total_w, yy - 0.04, w=1.5 if hdr or i == len(rows) - 1 else 0.6,
             color=INK if hdr or i == len(rows) - 1 else LIGHT)
    return yy


def dot(slide, x, y, d, color):
    box(slide, x, y, d, d, line=None, fill=color, shape=MSO_SHAPE.OVAL)


# ---------------------------------------------------------------------------
# Template plumbing
# ---------------------------------------------------------------------------
class Deck:
    def __init__(self):
        self.prs = Presentation(TEMPLATE)
        self.n_template = len(self.prs.slides)
        self.blank = [l for l in self.prs.slide_layouts if l.name == 'Blank'][0]
        src = self.prs.slides[3]                       # 'Background of the study' page
        self.chrome = [copy.deepcopy(sh._element) for sh in src.shapes
                       if sh.name in ('AutoShape 3', 'TextBox 8', 'TextBox 9')]
        self.page = 0

    @staticmethod
    def set_text(el, new):
        """Replace the text of a copied template text box, keeping its first run's style."""
        ns = {'a': 'http://schemas.openxmlformats.org/drawingml/2006/main'}
        paras = el.findall('.//a:p', ns)
        first_r = el.find('.//a:r', ns)
        rpr = copy.deepcopy(first_r.find('a:rPr', ns))
        for p in paras[1:]:
            p.getparent().remove(p)
        p0 = paras[0]
        for r in p0.findall('a:r', ns) + p0.findall('a:br', ns):
            p0.remove(r)
        r = etree.SubElement(p0, qn('a:r'))
        r.append(rpr)
        etree.SubElement(r, qn('a:t')).text = new

    def new(self, header=HEADER, number=True):
        s = self.prs.slides.add_slide(self.blank)
        for ph in list(s.placeholders):
            ph._element.getparent().remove(ph._element)
        tree = s.shapes._spTree
        for el in self.chrome:
            e = copy.deepcopy(el)
            name = e.find('.//' + qn('p:cNvPr')).get('name')
            if name == 'TextBox 8':
                self.set_text(e, TEAM)
            if name == 'TextBox 9':
                self.set_text(e, header)
                e.find('.//' + qn('a:pPr')).set('algn', 'r')
            tree.append(e)
        self.page += 1
        if number:
            text(s, 17.2, 10.45, 1.0, 0.35, str(self.page), size=14, color=MID, align='r')
        return s

    def from_template(self, idx, skip=('Group 2',)):
        """Copy a whole template page (minus photo groups) onto a new slide."""
        src = self.prs.slides[idx]
        s = self.prs.slides.add_slide(self.blank)
        for ph in list(s.placeholders):
            ph._element.getparent().remove(ph._element)
        tree = s.shapes._spTree
        for sh in src.shapes:
            if sh.name in skip:
                continue
            el = copy.deepcopy(sh._element)
            if el.xpath('.//a:blip'):                  # images are not carried over (relationships)
                continue
            tree.append(el)
        self.page += 1
        return s

    @staticmethod
    def find(slide, name):
        for sh in slide.shapes:
            if sh.name == name:
                return sh
        raise KeyError(name)

    def finish(self):
        sld = self.prs.slides._sldIdLst
        for sid in list(sld)[:self.n_template]:
            self.prs.part.drop_rel(sid.rId)
            sld.remove(sid)
        self.prs.save(OUT)
        print("wrote", OUT, "slides:", len(self.prs.slides))


# ---------------------------------------------------------------------------
# Slides
# ---------------------------------------------------------------------------
def build():
    S, H, E_, V = load("scaling_results.json"), load("havok_results.json"), load("equation_results.json"), load(
        "validation_results.json")
    OS = load("onestep_vs_free.json")
    FS = forecast_summary()
    fa = FS['a']
    mc = S['master_curve']
    v1, v2, v4, v5 = V['V1'], V['V2'], V['V4'], V['V5']
    co = E_['model_B']['g_coefficients']
    tl = E_['model_B']['tau_law']
    lof = E_['validation']
    ode_err = np.median([abs(float(r['B_lofo']['rel_t_arr_err'])) for r in lof.values()
                         if r['B_lofo']['rel_t_arr_err'] is not None and np.isfinite(float(r['B_lofo']['rel_t_arr_err']))])
    one_step = [v['teacher_forced_rmse'] for v in OS.values()]
    free = [v['free_running_rmse'] for v in OS.values()]
    D = Deck()

    # 1 ── title ──────────────────────────────────────────────────────────
    s = D.from_template(0)
    f = D.find
    D.set_text(f(s, 'TextBox 7')._element, "Decoding Dense Active Fluids Through Machine Learning")
    tb = f(s, 'TextBox 7')
    tb.width = E(10.6)
    for r in tb._element.iter(qn('a:rPr')):
        r.set('sz', '6000')
        r.set('spc', '-420')
    tb._element.find('.//' + qn('a:spcPts')).set('val', '6000')
    D.set_text(f(s, 'TextBox 9')._element, "Scaling, Koopman prediction and reduced equations")
    D.set_text(f(s, 'TextBox 8')._element, "for kinetic arrest in a dense active glass")
    f(s, 'TextBox 8').left = E(6.86)
    D.set_text(f(s, 'TextBox 10')._element, "Presented by :")
    D.set_text(f(s, 'TextBox 12')._element, TEAM)
    D.set_text(f(s, 'TextBox 11')._element, "Code & data :")
    D.set_text(f(s, 'TextBox 13')._element, REPO)
    for nm in ('TextBox 11', 'TextBox 13'):
        f(s, nm).left = E(13.18)
        f(s, nm).width = E(5.2)
    D.set_text(f(s, 'TextBox 14')._element, TEAM)
    D.set_text(f(s, 'TextBox 15')._element, "October 2026")
    text(s, X0, 9.77 + 0.42, 11.0, 0.4, MEMBERS, size=18)
    picture(s, "title_strip.png", 13.18, 1.95, w=5.01, frame=True)

    # 2 ── motivation ─────────────────────────────────────────────────────
    s = D.new()
    title(s, "Flow or freeze?", size=72, rule_y=None)
    rule(s, X0, 3.35, 11.0, 3.35)
    text(s, X0, 3.85, 4.4, 3.6,
         "Dense self-propelled particles each push themselves with an **active force f**. Packed tightly, every "
         "particle is caged by its neighbours.", size=20)
    text(s, 6.55, 3.85, 4.45, 3.6,
         "Strong pushing keeps the system **flowing**; weak pushing lets it **jam and freeze**. How the motion "
         "dies out near this transition has no closed theory.", size=20)
    card(s, X0, 7.25, 9.26, 2.75, None, "Our approach",
         "Learn the dynamics of the kinetic energy K(t) directly from simulations: scaling and master curves, "
         "Koopman-operator models, HAVOK and sparse equation discovery.", head_size=24, body_size=19)
    # hand-made schematic: particles in a cage
    fx, fy, fw, fh = 12.25, 2.0, 5.95, 7.25
    box(s, fx, fy, fw, fh)
    rng = np.random.default_rng(3)
    cx, cy = fx + fw / 2, fy + 3.3
    pts = []
    for row in range(-4, 5):
        for col in range(-4, 5):
            px = cx + (col + 0.5 * (row % 2)) * 0.92 + rng.normal(0, 0.06)
            py = cy + row * 0.8 + rng.normal(0, 0.06)
            if fx + 0.45 < px < fx + fw - 0.45 and fy + 0.45 < py < fy + 5.9:
                pts.append((px, py))
    for (px, py) in pts:
        big = rng.random() < 0.65
        r = 0.36 if big else 0.27
        centre = abs(px - cx) < 0.3 and abs(py - cy) < 0.3
        dot(s, px - r, py - r, 2 * r, ACCENT if centre else ("6E6E6E" if big else "BDBDBD"))
    box(s, cx - 1.25, cy - 1.15, 2.5, 2.3, line=INK, lw=1.25, shape=MSO_SHAPE.OVAL, dash='dash')
    rule(s, cx, cy, cx + 1.55, cy - 1.25, color=ACCENT, w=3, arrow=True)
    text(s, cx + 1.45, cy - 1.75, 0.6, 0.5, "f", size=26, font=BODY_B, color=ACCENT)
    text(s, fx + 0.3, fy + 6.05, fw - 0.6, 1.1,
         "{a}Self-propelled particle{/a} pushing against the **cage** formed by its neighbours (65 : 35 mixture).",
         size=16, font=BODY_I, spacing=1.15)

    # 3 ── system & data ──────────────────────────────────────────────────
    s = D.new()
    rule(s, 11.22, -0.16, 11.22, 11.4)
    text(s, X0, 1.85, 9.2, 1.2, "System & data", size=72, font=TITLE_FONT, spacing=0.9, title=True)
    rule(s, X0, 3.35, 10.5, 3.35)
    grid_table(s, X0, 3.8, [2.45, 6.3], [
        ["Model", "2D Kob–Andersen mixture, 65 : 35, Lennard-Jones"],
        ["Particles", "N = 1000, constant active force f on each"],
        ["Forces", "17 values, f = 0 … 2.0, one simulation each"],
        ["Signal", "mean kinetic energy K(t), Δt = 0.5, up to t ≈ 3×10^{4}"],
        ["We study", "log_{10}K: K falls by more than 20 decades"],
    ], size=19, row_h=0.95, fonts={(i, 0): BODY_B for i in range(5)}, header=False)
    picture(s, "data_overview.png", 11.55, 1.75, h=8.3)

    # 4 ── questions & pipeline ───────────────────────────────────────────
    s = D.new()
    title(s, "Two questions, one pipeline")
    card(s, X0, 3.65, 8.0, 2.45, "Q1", "Scaling",
         "Do all trajectories follow one master curve, and how does the time scale grow with f?",
         num_size=40, head_size=22, body_size=19)
    card(s, 10.2, 3.65, 8.0, 2.45, "Q2", "Prediction",
         "Can Koopman-operator models forecast a K(t) trajectory they have never seen?",
         num_size=40, head_size=22, body_size=19)
    steps = [("17 trajectories", "K(t; f)", ""), ("Master curve", "& critical fits", "Q1"),
             ("HAVOK", "intermittency", "Q1"), ("Weak SINDy", "reduced equation", "Q1 · Q2"),
             ("Koopman", "free-running forecasts", "Q2")]
    bw, gap, by = 2.92, 0.465, 6.75
    for i, (a, b, tag) in enumerate(steps):
        bx = X0 + i * (bw + gap)
        box(s, bx, by, bw, 1.55, line=ACCENT if i == 4 else INK)
        text(s, bx + 0.15, by + 0.28, bw - 0.3, 0.5, a, size=21, font=BODY_B, align='ctr')
        text(s, bx + 0.15, by + 0.82, bw - 0.3, 0.5, b, size=17, font=BODY_I, align='ctr')
        if tag:
            text(s, bx, by + 1.68, bw, 0.4, tag, size=16, color=MID, align='ctr')
        if i < 4:
            rule(s, bx + bw + 0.06, by + 0.78, bx + bw + gap - 0.06, by + 0.78, arrow=True)
    text(s, X0, 9.35, 16.4, 0.6,
         "{i}Every claim is checked with a test that could fail: baselines, held-out forces, burst-free surrogates "
         "and bootstrap intervals.{/i}", size=19, color=INK)

    # 5 ── anatomy of arrest ──────────────────────────────────────────────
    s = D.new()
    title(s, "Anatomy of an arrest")
    picture(s, "anatomy.png", X0 - 0.15, 3.5, w=10.3)
    picture(s, "arrest_times.png", 12.0, 3.45, w=6.2)
    f165 = D_tarr = S['taus']['t_arr'][S['forces'].index(1.65)]
    card(s, 12.25, 7.75, 5.95, 2.45, None, None,
         [f"f = 1.65 still arrests (t ≈ {D_tarr:,.0f}).".replace(",", " "),
          f"f = 1.8 is still decaying at the end of the run (p < 0.001)  →  {{a}}f_{{c}} > 1.8{{/a}}"],
         body_size=19)

    # 6 ── master curve ───────────────────────────────────────────────────
    s = D.new()
    title(s, "Q1 · One master curve")
    text(s, 9.2, 2.15, 9.0, 0.8, "log_{10}K(t, f) = a(f) + G( t / τ(f) )", size=30, align='r')
    picture(s, "collapse_raw.png", X0 - 0.1, 3.55, w=7.6)
    picture(s, "collapse_master.png", 10.45, 3.55, w=7.6)
    rule(s, 9.25, 6.0, 10.35, 6.0, w=3, arrow=True)
    text(s, 8.95, 6.25, 1.7, 0.5, "rescale", size=16, color=MID, align='ctr')
    text(s, X0 + 0.6, 8.75, 7.0, 0.8, f"spread {mc['cost_raw']:.2f}", size=40, font=TITLE_FONT, title=True)
    text(s, X0 + 0.6, 9.55, 7.0, 0.5, "variance of log K across forces, as simulated", size=17, font=BODY_I)
    text(s, 11.05, 8.75, 7.0, 0.8, f"spread {mc['cost_2param']:.3f}", size=40, font=TITLE_FONT, color=ACCENT,
         title=True)
    text(s, 11.05, 9.55, 7.0, 0.5, f"after t → t/τ(f) and a shift a(f)  ·  R^{{2}} = "
                                   f"{1 - mc['cost_2param'] / mc['cost_raw']:.3f}", size=17, font=BODY_I)

    # 7 ── is the collapse real? ──────────────────────────────────────────
    s = D.new()
    title(s, "Is the collapse real?")
    text(s, X0, 3.45, 16.4, 0.5, "{i}Two free numbers per curve is a lot of freedom, so we tested what the collapse "
                                 "never used.{/i}", size=19)
    picture(s, "tarr_vs_tauK.png", X0 - 0.1, 4.05, w=7.4)
    picture(s, "cliff.png", 10.75, 4.05, w=7.4)
    text(s, X0, 9.35, 7.6, 0.9, f"Arrest time follows from the slow decay alone: τ_{{K}} varies "
                                f"{v2['tau_K_range'][1] / v2['tau_K_range'][0]:.0f}×, the ratio t_{{arr}}/τ_{{K}} only "
                                f"{{a}}±{100 * v2['ratio_cv']:.0f} %{{/a}}.", size=18)
    text(s, 10.75, 9.35, 7.45, 0.9, f"After arrest every force collapses at the same rate: "
                                    f"{{a}}{-S['cliff_rate']['mean']:.2f} ± {S['cliff_rate']['std']:.2f}{{/a}} decades "
                                    f"per time unit.", size=18)

    # 8 ── critical slowing down ──────────────────────────────────────────
    s = D.new()
    title(s, "Critical slowing down")
    picture(s, "critical_loglog.png", X0 - 0.1, 3.4, w=7.6)
    picture(s, "critical_degeneracy.png", 10.55, 3.4, w=7.6)
    cw = (X1 - X0 - 2 * 0.4) / 3
    for i, (num, head, body) in enumerate([
            ("01.", "Arrhenius rejected", "τ grows faster than exponential: ΔAIC = 11–26 for four τ definitions"),
            ("02.", "f_{c} ≈ 1.8 – 1.95", "the critical force lies above every arresting run"),
            ("03.", "γ ≈ 1.4 – 2.1", "power law ≈ VFT; γ and f_{c} trade off against each other")]):
        card(s, X0 + i * (cw + 0.4), 8.75, cw, 1.6, None, f"{num}  {head}", body, head_size=21, body_size=16,
             accent=i == 1)

    # 9 ── HAVOK method ───────────────────────────────────────────────────
    s = D.new()
    title(s, "HAVOK: linear dynamics + rare kicks")
    bw, gap, by, bh = 3.68, 0.58, 3.65, 3.25
    labels = [("1  Signal", "fluctuation of log K"), ("2  Delay matrix", "stack 20 t.u. of history"),
              ("3  SVD", "a few dominant patterns  v_{1} … v_{r}"),
              ("4  Linear model", "dv/dt = A v + B v_{r}")]
    for i, (a, b) in enumerate(labels):
        bx = X0 + i * (bw + gap)
        box(s, bx, by, bw, bh, line=ACCENT if i == 3 else INK)
        text(s, bx + 0.25, by + 0.2, bw - 0.5, 0.5, a, size=21, font=BODY_B)
        text(s, bx + 0.25, by + 2.45, bw - 0.5, 0.7, b, size=16, font=BODY_I, spacing=1.1)
        if i < 3:
            rule(s, bx + bw + 0.08, by + bh / 2, bx + bw + gap - 0.08, by + bh / 2, arrow=True)
    picture(s, "signal_mini.png", X0 + 0.25, by + 0.8, w=3.2)
    gx, gy = X0 + bw + gap + 0.55, by + 0.85
    for rr in range(4):
        for cc in range(6):
            shade = ["5A5A5A", "7A7A7A", "9C9C9C", "BEBEBE"][(rr + cc) % 4]
            box(s, gx + cc * 0.42, gy + rr * 0.36, 0.36, 0.3, line=None, fill=shade)
    sx = X0 + 2 * (bw + gap) + 0.35
    for k, hgt in enumerate([1.25, 0.8, 0.5, 0.3, 0.2]):
        box(s, sx + k * 0.6, by + 2.2 - hgt, 0.42, hgt, line=None, fill=INK if k < 4 else ACCENT)
    text(s, X0 + 3 * (bw + gap) + 0.25, by + 0.9, bw - 0.5, 1.4,
         ["v_{r} quiet → linear", "{a}v_{r} bursts → something sudden{/a}"], size=17, spacing=1.15)
    picture(s, "lorenz.png", X0 - 0.1, 7.3, w=6.6)
    text(s, 8.9, 7.55, 9.3, 2.6, [
        "**Check 1 · Lorenz system:** the forcing fires at the lobe switches "
        f"({H['lorenz']['frac_switches_preceded_by_forcing'] / H['lorenz']['chance_level']:.1f}× chance).",
        "**Check 2 · PyDMD:** the open-source implementation gives the same eigenvalue (0.189i vs 0.188i).",
        "{m}Our own implementation of Brunton et al., Nat. Commun. 8, 19 (2017).{/m}"], size=18, spacing=1.2)

    # 10 ── HAVOK results ─────────────────────────────────────────────────
    s = D.new()
    title(s, "Bursts announce the arrest")
    picture(s, "havok_trace.png", X0 - 0.15, 3.45, w=10.0)
    picture(s, "havok_kurtosis.png", 11.8, 3.45, w=6.4)
    cw = (X1 - X0 - 2 * 0.4) / 3
    for i, (num, head, body) in enumerate([
            ("01.", "Bursty only when arresting", "kurtosis 3.5–32 for every arresting force; Gaussian at f = 2.0"),
            ("02.", "Real, not an artefact", "above all 19 burst-free surrogates; same for 3 window sizes"),
            ("03.", "An early warning", f"switches on in the last ~20 % and precedes sudden drops "
                                        f"{v4['40']['dip_hit_over_chance_median']:.1f}× more often than chance")]):
        card(s, X0 + i * (cw + 0.4), 8.15, cw, 2.15, num, head, body, num_size=28, head_size=21, body_size=16,
             accent=i == 2)

    # 11 ── equation ──────────────────────────────────────────────────────
    s = D.new()
    title(s, "A reduced equation for all forces")
    box(s, X0, 3.55, 16.46, 1.55, line=ACCENT)
    text(s, X0 + 0.4, 3.72, 15.6, 0.7, "τ(f) · dy/dt  =  g( y − a(f) ),      y = log_{10}K", size=30, align='ctr')
    text(s, X0 + 0.4, 4.45, 15.6, 0.5,
         f"g(u) = {co['u']:.2f}u + {co['u^2']:.2f}u^{{2}} + {co['u^3']:.3f}u^{{3}}      "
         f"τ(f) = {tl['A']:.0f} ({tl['fc']:.3f} − f)^{{−{tl['gamma']:.2f}}}      then collapse at "
         f"{-S['cliff_rate']['mean']:.2f} decades / t.u.", size=17, font=BODY_I, align='ctr')
    picture(s, "ode_lofo.png", X0 - 0.1, 5.25, h=4.75)
    picture(s, "g_of_u.png", 11.3, 5.25, h=4.75)
    text(s, X0, 10.1, 15.4, 0.45,
         f"{{m}}Weak-form SINDy · a generic polynomial library fails · leave-one-force-out arrest error "
         f"{100 * ode_err:.0f} % · the differential form of the master curve, not a first-principles law{{/m}}",
         size=15)

    # 12 ── Q2 set-up ─────────────────────────────────────────────────────
    s = D.new()
    title(s, "Q2 · Predicting runs never seen")
    text(s, X0, 3.55, 9.0, 0.5, "(a) Unseen force", size=21, font=BODY_B)
    box(s, X0, 4.15, 8.6, 0.62, line=ACCENT)
    text(s, X0, 4.27, 8.6, 0.4, "whole trajectory predicted from f alone", size=16, font=BODY_I, align='ctr')
    text(s, X0, 5.05, 9.0, 0.5, "(b) Given the first part", size=21, font=BODY_B)
    box(s, X0, 5.65, 2.3, 0.62, line=None, fill=LIGHT)
    text(s, X0, 5.77, 2.3, 0.4, "10 / 25 / 50 %", size=16, font=BODY_I, align='ctr')
    box(s, X0 + 2.3, 5.65, 6.3, 0.62, line=ACCENT)
    text(s, X0 + 2.3, 5.77, 6.3, 0.4, "forecast the rest", size=16, font=BODY_I, align='ctr')
    text(s, X0, 6.75, 9.0, 0.5, "Five models, all rebuilt without the test run", size=21, font=BODY_B)
    chips = ["Neighbour baseline", "Parametric HDMD", "Deep Koopman AE", "Master-curve Koopman", "Discovered equation"]
    cx_, cy_ = X0, 7.35
    for i, c in enumerate(chips):
        w_ = 0.12 * len(c) + 0.6
        if cx_ + w_ > X0 + 9.0:
            cx_, cy_ = X0, cy_ + 0.8
        box(s, cx_, cy_, w_, 0.6, line=ACCENT if i == 3 else INK, shape=MSO_SHAPE.ROUNDED_RECTANGLE)
        text(s, cx_, cy_ + 0.13, w_, 0.4, c, size=16, align='ctr', color=ACCENT if i == 3 else INK)
        cx_ += w_ + 0.25
    rule(s, 11.0, 3.55, 11.0, 10.2, color=LIGHT, w=1)
    picture(s, "onestep_vs_free.png", 11.25, 3.45, w=7.0)
    text(s, 11.45, 7.05, 6.7, 3.0, [
        "**One-step-ahead ≠ forecasting.**",
        f"The same model is {min(one_step):.2f} decades off when shown the true history every step, but "
        f"{min(free):.1f}–{max(free):.1f} decades off when running on its own.",
        "{a}All results here are free-running and leave-one-force-out.{/a}"], size=18, spacing=1.2)

    # 13 ── Q2 unseen force ───────────────────────────────────────────────
    s = D.new()
    title(s, "Q2 · Forecasting an unseen force")
    picture(s, "parity.png", X0 - 0.15, 3.45, h=6.75)
    lab = {'NeighbourBaseline': "Neighbour baseline", 'HDMD-param': "Parametric HDMD",
           'DeepKoopman': "Deep Koopman AE", 'MasterCurve': "Master-curve Koopman", 'ODE-B': "Discovered equation"}
    rows = [["model", "median error", "missed", "false"]]
    for m in ('NeighbourBaseline', 'HDMD-param', 'DeepKoopman', 'MasterCurve', 'ODE-B'):
        r = fa[m]
        rows.append([lab[m], f"{pct(r['med'])}  [{pct(r['lo'])}–{pct(r['hi'])}]", f"{r['missed']}/{r['n']}",
                     f"{r['false']}/{r['n_fluid']}"])
    colors = {(4, j): ACCENT for j in range(4)}
    fonts = {(4, j): BODY_B for j in range(4)}
    yy = grid_table(s, 9.85, 3.6, [3.35, 2.75, 1.1, 1.15], rows, size=17, row_h=0.6, colors=colors, fonts=fonts)
    text(s, 9.85, yy + 0.05, 8.3, 0.4, "{m}arrest-time error, 11 arresting forces; false arrests for f = 1.8, 2.0{/m}",
         size=14)
    text(s, 9.85, yy + 0.65, 8.35, 3.2, [
        "Linear Koopman operators cannot create the abrupt arrest; the deep Koopman AE drifts.",
        f"Scaling-informed Koopman: {{a}}{pct(fa['MasterCurve']['med'])}{{/a}}, no missed or false arrests.",
        f"A time-stretched neighbour copy is at least as good inside the range ({pct(fa['NeighbourBaseline']['med'])}) but "
        "fails at the edges."], size=18, spacing=1.15)

    # 14 ── Q2 prefix ─────────────────────────────────────────────────────
    s = D.new()
    title(s, "Q2 · Forecasting from the first part of a run")
    picture(s, "prefix_example.png", X0 - 0.15, 3.55, w=8.4)
    text(s, 10.6, 3.6, 7.6, 0.5, "Master-curve Koopman, median arrest-time error", size=19, font=BODY_B)
    cw = (X1 - 10.6 - 2 * 0.3) / 3
    for i, p in enumerate(FS['prefixes']):
        x_ = 10.6 + i * (cw + 0.3)
        box(s, x_, 4.3, cw, 2.35, line=ACCENT if i == 0 else INK)
        text(s, x_, 4.5, cw, 0.45, f"{int(round(100 * float(p)))} % given", size=17, font=BODY_I, align='ctr')
        text(s, x_, 5.15, cw, 1.0, pct(FS['b']['MasterCurve'][p]['med']), size=46, font=TITLE_FONT, color=ACCENT,
             align='ctr', title=True)
    hd = FS['b']['HDMD-prefix']
    dk = FS['b']['DeepKoopman']
    ob = FS['b']['ODE-B']
    text(s, 10.6, 7.05, 7.6, 3.1, [
        f"HDMD fitted on the prefix (≈ HAVOK without future kicks) and the deep Koopman AE miss most arrests "
        f"(up to {max(hd[p]['missed'] for p in FS['prefixes'])} and {max(dk[p]['missed'] for p in FS['prefixes'])} of 11).",
        f"Discovered equation: {pct(ob[FS['prefixes'][0]]['med'])} → {pct(ob[FS['prefixes'][-1]]['med'])}.",
        "{m}No missed arrests for the master-curve model at any prefix length.{/m}"], size=18, spacing=1.15)

    # 15 ── validation ────────────────────────────────────────────────────
    s = D.new()
    title(s, "How we tested every claim")
    rows = [["claim", "test that could have failed", "result", ""],
            ["f = 1.65 arrests; f = 1.8 still decaying", "block-bootstrap test of the slope", "p < 0.001", ""],
            ["one master curve for f = 1.0 – 1.8", "predict arrest from slow decay alone; held-out forces",
             f"R^{{2}} = {v2['r2']:.2f}, ratio ±{100 * v2['ratio_cv']:.0f} %", ""],
            ["τ diverges faster than exponential", "AIC against Arrhenius, four τ definitions", "ΔAIC = 11–26", ""],
            ["f_{c} ≈ 1.8 – 1.95,  γ ≈ 1.4 – 2.1", "profile, bootstrap, fit windows", "range only (degenerate)", ""],
            ["intermittent bursts are real", "19 burst-free surrogates; 3 window sizes", "above all surrogates", ""],
            ["reduced equation predicts unseen forces", "leave-one-out; synthetic recovery; bootstrap",
             f"{100 * ode_err:.0f} % error", ""],
            ["dK/dt = −AK + B f K^{n}", "fit n and the sign of B", "unphysical (B < 0)", ""],
            ["Koopman + master curve forecasts", "free-running, leave-one-out, vs baseline",
             f"{pct(fa['MasterCurve']['med'])}; ties baseline inside", ""]]
    verdict = [None, GREEN, GREEN, GREEN, AMBER, GREEN, GREEN, RED, GREEN]
    grid_table(s, X0, 3.5, [5.6, 5.9, 3.85, 1.1], rows, size=17, row_h=0.69)
    for i, c in enumerate(verdict):
        if c:
            dot(s, X0 + 5.6 + 5.9 + 3.85 + 0.4, 3.6 + i * 0.69 + 0.08, 0.3, c)
    for i, (c, l_) in enumerate([(GREEN, "supported"), (AMBER, "partly"), (RED, "rejected")]):
        dot(s, X0 + i * 2.6, 9.95, 0.26, c)
        text(s, X0 + i * 2.6 + 0.38, 9.88, 2.1, 0.4, l_, size=16)
    text(s, 9.6, 9.88, 8.6, 0.4, "{m}full list of 15 claims in the technical report: 10 supported · 4 partly · "
                                 "1 rejected{/m}", size=15, align='r')

    # 16 ── conclusions ───────────────────────────────────────────────────
    s = D.new()
    title(s, "Take-home messages")
    items = [("01.", "Q1 · Scaling", "One master curve  log_{10}K = a(f) + G(t/τ(f));  τ diverges near f_{c} ≈ 1.8–1.95."),
             ("02.", "Q2 · Prediction", f"Plain Koopman models fail when run freely; a scaling-informed Koopman model "
                                        f"predicts unseen runs ({pct(fa['MasterCurve']['med'])})."),
             ("03.", "Equation", "τ(f) · dy/dt = g(y − a(f)): compact, predictive, data-driven."),
             ("04.", "HAVOK", "Intermittent bursts switch on before arrest: an early-warning signal.")]
    for i, (num, head, body) in enumerate(items):
        yy = 3.65 + i * 1.6
        text(s, X0, yy, 1.2, 0.8, num, size=34, font=BODY_B, color=ACCENT)
        text(s, X0 + 1.35, yy + 0.02, 8.6, 0.5, head, size=22, font=BODY_B)
        text(s, X0 + 1.35, yy + 0.52, 8.6, 0.9, body, size=18, spacing=1.15)
    box(s, 12.3, 3.65, 5.9, 6.3)
    text(s, 12.65, 3.95, 5.3, 0.5, "Next steps", size=24, font=BODY_B)
    rule(s, 12.65, 4.65, 17.85, 4.65, w=1)
    text(s, 12.65, 4.9, 5.25, 4.9, [
        "Several runs per force → error bars",
        "Blind test: predict f = 1.7, 1.75, 1.85, 1.9 before simulating them",
        "A second system size and density",
        "Compare γ and G with active mode-coupling theory"], size=18, spacing=1.2, space_after=0.8)

    # 17 ── thank you ─────────────────────────────────────────────────────
    s = D.from_template(14)
    f = D.find
    D.set_text(f(s, 'TextBox 9')._element, "Decoding Dense Active Fluids Through Machine Learning")
    D.set_text(f(s, 'TextBox 8')._element, "Questions are very welcome")
    D.set_text(f(s, 'TextBox 10')._element, "Presented by :")
    D.set_text(f(s, 'TextBox 12')._element, TEAM)
    D.set_text(f(s, 'TextBox 11')._element, "Code & data :")
    D.set_text(f(s, 'TextBox 13')._element, REPO)
    for nm in ('TextBox 11', 'TextBox 13'):
        f(s, nm).left = E(13.18)
        f(s, nm).width = E(5.2)
    D.set_text(f(s, 'TextBox 14')._element, TEAM)
    D.set_text(f(s, 'TextBox 15')._element, "October 2026")
    text(s, X0, 9.77 + 0.42, 11.0, 0.4, MEMBERS, size=18)
    picture(s, "collapse_master.png", 12.3, 2.6, w=6.0)

    # ── backup slides ────────────────────────────────────────────────────
    s = D.new(header="Backup")
    title(s, "Backup · Critical fits", size=48)
    rows = [["τ definition", "best f_{c}", "best γ", "acceptable f_{c}  (ΔAIC < 2)", "acceptable γ", "ΔAIC Arrhenius",
             "ΔAIC VFT"]]
    names = {'t_arr': "arrest time", 't_-5': "time to K = 10^{−5}", 'tau_K': "slow-decay time τ_{K}",
             'tau_mc': "master-curve τ", 't_-4': "time to K = 10^{−4}  (biased)",
             't_-3': "time to K = 10^{−3}  (biased)", 'tau_HDMD': "HDMD on linear K  (biased)"}
    for k in ('t_arr', 't_-5', 'tau_K', 'tau_mc', 't_-4', 't_-3', 'tau_HDMD'):
        ft = S['fits'][k]
        pl, ad, a_ = ft['powerlaw'], ft['admissible'], ft['aic']
        rows.append([names[k], f"{pl['fc']:.2f}", f"{pl['gamma']:.2f}",
                     f"{ad['fc_range'][0]:.2f} – {ad['fc_range'][1]:.2f}",
                     f"{ad['gamma_range'][0]:.2f} – {ad['gamma_range'][1]:.2f}",
                     f"{a_['arrhenius'] - a_['powerlaw']:+.1f}", f"{a_['vft'] - a_['powerlaw']:+.1f}"])
    grid_table(s, X0, 3.6, [4.4, 1.6, 1.5, 3.2, 2.3, 1.9, 1.55], rows, size=17, row_h=0.7,
               colors={(i, j): MID for i in (5, 6, 7) for j in range(7)})
    text(s, X0, 9.55, 16.4, 0.8, "{m}τ = A (f_{c} − f)^{−γ} with f_{c} above every fitted force. Times to reach a fixed "
                                 "level mix the f-dependent offset a(f) with the time scale and are therefore biased."
                                 "{/m}", size=16)

    s = D.new(header="Backup")
    title(s, "Backup · HAVOK robustness", size=48)
    picture(s, "havok_pdfs.png", X0 - 0.1, 3.5, w=7.4)
    rows = [["delay window", "median kurtosis (arresting)", "f = 1.8", "f = 2.0", "drops: hit / chance"]]
    for q in ('20', '40', '80'):
        v = v4[q]
        rows.append([f"{int(q) // 2} t.u.", f"{v['kurtosis_jammed_median']:.1f}", f"{v['kurtosis_1p8']:.2f}",
                     f"{v['kurtosis_2p0']:.2f}", f"{v['dip_hit_over_chance_median']:.1f}×"])
    grid_table(s, 9.7, 3.7, [1.9, 2.6, 1.15, 1.15, 1.7], rows, size=17, row_h=0.75)
    text(s, 9.7, 7.1, 8.5, 2.8, [
        "Every arresting force has fat-tailed forcing at every window size; f = 2.0 stays Gaussian.",
        "Surrogates: phase-randomised copies with the same power spectrum (Theiler et al., Physica D 1992)."],
         size=17, spacing=1.15)

    s = D.new(header="Backup")
    title(s, "Backup · Equation details", size=48)
    b5 = v5['bootstrap']
    rows = [["term", "coefficient", "95 % bootstrap CI", "selected"]]
    for n, lo, hi, fs_ in zip(b5['names'], b5['ci95_low'], b5['ci95_high'], b5['frac_selected']):
        rows.append([n.replace('^2', '^{2}').replace('^3', '^{3}'), f"{co.get(n, 0.0):.3f}", f"[{lo:.2f}, {hi:.2f}]",
                     f"{100 * fs_:.0f} %"])
    yy = grid_table(s, X0, 3.6, [1.4, 2.0, 3.0, 1.6], rows, size=17, row_h=0.65)
    deg = v5['degree']
    text(s, X0, yy + 0.4, 8.0, 2.5, [
        f"Polynomial degree (leave-one-out arrest error): quadratic {pct(deg['2']['median_abs_t_arr_err'])}, "
        f"cubic {pct(deg['3']['median_abs_t_arr_err'])}, quartic {pct(deg['4']['median_abs_t_arr_err'])}.",
        "Synthetic test: the correct law is recovered up to twice the data's noise level."], size=17, spacing=1.15)
    picture(s, "langevin.png", 10.6, 3.5, w=7.4)
    L = E_['model_L']
    text(s, 10.6, 8.4, 7.6, 1.6, f"Langevin version: median rescaled arrest time {L['s_arrest_sim'][1]:.2f} vs "
                                 f"{np.median(L['s_arrest_obs']):.2f} observed; the spread is overestimated.",
         size=17, spacing=1.15)

    s = D.new(header="Backup")
    title(s, "Backup · Per-force forecast errors", size=48)
    rows = [["f", "Neighbour", "Param. HDMD", "Deep Koopman", "Master curve", "Equation"]]
    for k in FS['forces']:
        row = [k]
        for m in ('NeighbourBaseline', 'HDMD-param', 'DeepKoopman', 'MasterCurve', 'ODE-B'):
            r = fa[m]['per_force'][k]
            tp, tt = r['t_arr_pred'], r['t_arr_true']
            tt_f = tt is not None and np.isfinite(float(tt))
            tp_f = tp is not None and np.isfinite(float(tp))
            row.append((f"{100 * float(r['t_arr_rel_err']):+.0f} %" if tp_f else "no arrest") if tt_f
                       else ("correct" if not tp_f else "false arrest"))
        rows.append(row)
    grid_table(s, X0, 3.45, [1.4, 3.0, 3.0, 3.0, 3.0, 3.0], rows, size=15, row_h=0.47,
               colors={(i, 4): ACCENT for i in range(1, len(rows))})
    text(s, X0, 10.15, 16.4, 0.4, "{m}relative error of the predicted arrest time; f = 1.0 and f ≥ 1.65 require "
                                  "extrapolation in f{/m}", size=14)

    s = D.new(header="Backup")
    title(s, "Backup · Method checks", size=48)
    picture(s, "sindy_noise.png", X0 - 0.1, 3.5, w=6.8)
    rows = [["τ_{K} fit window", "γ", "f_{c}", "R^{2}"]]
    for k, v in V['V7'].items():
        a_, b_ = k.split('-')
        rows.append([f"t = {a_} … {float(b_):.2f} t_{{arr}}", f"{v['gamma']:.2f}", f"{v['fc']:.3f}", f"{v['r2']:.3f}"])
    grid_table(s, 9.9, 3.6, [3.6, 1.4, 1.6, 1.5], rows, size=17, row_h=0.62)
    text(s, X0, 8.4, 7.0, 1.6, "Weak SINDy recovers the right terms on synthetic data with the simulations' correlated "
                               "noise up to 0.2 decades (ours ≈ 0.1).", size=17, spacing=1.15)
    text(s, 9.9, 8.4, 8.3, 1.6, "γ and f_{c} barely move when the slow-stage window is changed.", size=17)

    D.finish()


if __name__ == "__main__":
    build()
