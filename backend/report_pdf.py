"""
report_pdf.py — generate a detailed, printable forensic PDF report.

The frontend posts the completed analysis result (plus file name and SHA-256)
to ``POST /api/report``; this module turns it into a multi-page A4 PDF.

Only reportlab + matplotlib are required.  Every value is taken from the
analysis result; nothing is invented.  Missing fields degrade gracefully so
that demo results can also be exported.
"""

from __future__ import annotations

import io
import logging
import threading
from datetime import datetime
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    HRFlowable,
    Image,
    KeepTogether,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.platypus.flowables import Flowable

logger = logging.getLogger(__name__)

# ─── Palette (flat; no gradients) ────────────────────────────────────────────
NAVY = colors.HexColor("#102a43")
BLUE = colors.HexColor("#2e74c0")
BLUE_DARK = colors.HexColor("#1f5c9e")
LIGHT = colors.HexColor("#eaf3fc")
LIGHT_2 = colors.HexColor("#f6f9fd")
BORDER = colors.HexColor("#c3d9ee")
MUTED = colors.HexColor("#5b7085")
INK = colors.HexColor("#243b53")
GREEN = colors.HexColor("#1f7a4d")
GREEN_BG = colors.HexColor("#e8f6ee")
RED = colors.HexColor("#b42318")
RED_BG = colors.HexColor("#fdecea")
AMBER = colors.HexColor("#a15c00")
AMBER_BG = colors.HexColor("#fdf6e7")
GREY = colors.HexColor("#8fa6bb")

# ─── Fonts ───────────────────────────────────────────────────────────────────
FONT = "Helvetica"
FONT_BOLD = "Helvetica-Bold"
FONT_MONO = "Courier"

try:  # Arial gives a cleaner, more formal report on Windows
    pdfmetrics.registerFont(TTFont("ReportSans", r"C:\Windows\Fonts\arial.ttf"))
    pdfmetrics.registerFont(TTFont("ReportSans-Bold", r"C:\Windows\Fonts\arialbd.ttf"))
    pdfmetrics.registerFont(TTFont("ReportMono", r"C:\Windows\Fonts\cour.ttf"))
    FONT, FONT_BOLD, FONT_MONO = "ReportSans", "ReportSans-Bold", "ReportMono"
except Exception:  # pragma: no cover - non-Windows fallback
    logger.warning("Arial/Courier fonts not found; using built-in Helvetica/Courier")

# ─── Text helpers ────────────────────────────────────────────────────────────
_UNICODE_MAP = {
    "\u2014": "-", "\u2013": "-", "\u2012": "-", "\u2015": "-",
    "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
    "\u2026": "...", "\u00d7": "x", "\u2265": ">=", "\u2264": "<=",
    "\u2022": "-", "\u2192": "->", "\u2190": "<-", "\u00b7": "-",
    "\u202f": " ", "\u2009": " ", "\u00a0": " ",
}


def _txt(value) -> str:
    """Plain text safe for the PDF fonts (Latin-1 / Arial)."""
    if value is None:
        return ""
    s = str(value)
    for k, v in _UNICODE_MAP.items():
        s = s.replace(k, v)
    s = s.replace("\r", " ")
    return "".join(ch for ch in s if ord(ch) < 256)


def _para(value, style) -> Paragraph:
    s = escape(_txt(value)).replace("\n", "<br/>")
    return Paragraph(s, style)


def _pct(value, digits: int = 1) -> str:
    try:
        return f"{float(value) * 100:.{digits}f}%"
    except (TypeError, ValueError):
        return "n/a"


def _num(value, digits: int = 3) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "n/a"


def _ms(value) -> str:
    try:
        v = float(value)
        return f"{v:.1f} s" if v >= 1000 else f"{v:.0f} ms"
    except (TypeError, ValueError):
        return "n/a"


def _safe_float(value, default=0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


# ─── Flowables ───────────────────────────────────────────────────────────────
class ProbabilityBar(Flowable):
    """A flat, single-colour probability bar."""

    def __init__(self, value: float, colour, width: float = 260, height: float = 9):
        super().__init__()
        self.value = max(0.0, min(1.0, _safe_float(value)))
        self.colour = colour
        self.width = width
        self.height = height

    def wrap(self, availWidth, availHeight):
        return self.width, self.height

    def draw(self):
        c = self.canv
        c.setFillColor(LIGHT)
        c.roundRect(0, 0, self.width, self.height, 2, stroke=0, fill=1)
        fill = self.width * self.value
        if fill > 0:
            c.setFillColor(self.colour)
            c.roundRect(0, 0, max(1.2, fill), self.height, 2, stroke=0, fill=1)


class Sparkline(Flowable):
    """Tiny line chart for the evidence-card mini series."""

    def __init__(self, data, colour=BLUE, width: float = 110, height: float = 24):
        super().__init__()
        self.data = [_safe_float(x) for x in (data or [])]
        self.colour = colour
        self.width = width
        self.height = height

    def wrap(self, availWidth, availHeight):
        return self.width, self.height

    def draw(self):
        if len(self.data) < 2:
            return
        c = self.canv
        lo, hi = min(self.data), max(self.data)
        span = (hi - lo) or 1.0
        step = self.width / (len(self.data) - 1)
        c.setStrokeColor(self.colour)
        c.setLineWidth(1.1)
        p = c.beginPath()
        for i, v in enumerate(self.data):
            x = i * step
            y = 2 + ((v - lo) / span) * (self.height - 4)
            p.moveTo(x, y) if i == 0 else p.lineTo(x, y)
        c.drawPath(p, stroke=1, fill=0)


class NumberedCanvas(canvas.Canvas):
    """Canvas that can print 'Page X of Y'."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        total = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self._draw_page_number(total)
            super().showPage()
        super().save()

    def _draw_page_number(self, total):
        self.saveState()
        self.setFont(FONT, 7.4)
        self.setFillColor(MUTED)
        self.drawRightString(A4[0] - 18 * mm, 10 * mm, f"Page {self._pageNumber} of {total}")
        self.restoreState()


# ─── Styles ──────────────────────────────────────────────────────────────────
def _styles():
    base = getSampleStyleSheet()
    return {
        "brand": ParagraphStyle(
            "brand", parent=base["Normal"], fontName=FONT_BOLD, fontSize=21,
            leading=24, textColor=BLUE, spaceAfter=1,
        ),
        "title": ParagraphStyle(
            "title", parent=base["Normal"], fontName=FONT_BOLD, fontSize=16,
            leading=19, textColor=NAVY, spaceAfter=2,
        ),
        "subtitle": ParagraphStyle(
            "subtitle", parent=base["Normal"], fontName=FONT, fontSize=8.2,
            leading=11, textColor=MUTED,
        ),
        "h2": ParagraphStyle(
            "h2", parent=base["Normal"], fontName=FONT_BOLD, fontSize=11.2,
            leading=14, textColor=NAVY, spaceBefore=2, spaceAfter=3,
        ),
        "h3": ParagraphStyle(
            "h3", parent=base["Normal"], fontName=FONT_BOLD, fontSize=9.3,
            leading=12, textColor=BLUE_DARK, spaceBefore=1, spaceAfter=1,
        ),
        "body": ParagraphStyle(
            "body", parent=base["Normal"], fontName=FONT, fontSize=8.8,
            leading=12.2, textColor=INK, spaceAfter=3,
        ),
        "small": ParagraphStyle(
            "small", parent=base["Normal"], fontName=FONT, fontSize=7.6,
            leading=10, textColor=INK,
        ),
        "tiny": ParagraphStyle(
            "tiny", parent=base["Normal"], fontName=FONT, fontSize=6.8,
            leading=8.8, textColor=MUTED,
        ),
        "mono": ParagraphStyle(
            "mono", parent=base["Normal"], fontName=FONT_MONO, fontSize=7.2,
            leading=9.5, textColor=INK,
        ),
        "table": ParagraphStyle(
            "table", parent=base["Normal"], fontName=FONT, fontSize=7.7,
            leading=9.8, textColor=INK,
        ),
        "table_bold": ParagraphStyle(
            "table_bold", parent=base["Normal"], fontName=FONT_BOLD, fontSize=7.7,
            leading=9.8, textColor=INK,
        ),
        "table_head": ParagraphStyle(
            "table_head", parent=base["Normal"], fontName=FONT_BOLD, fontSize=7.2,
            leading=9, textColor=NAVY,
        ),
        "verdict": ParagraphStyle(
            "verdict", parent=base["Normal"], fontName=FONT_BOLD, fontSize=21,
            leading=24, alignment=TA_LEFT,
        ),
        "verdict_sub": ParagraphStyle(
            "verdict_sub", parent=base["Normal"], fontName=FONT, fontSize=8.2,
            leading=11, textColor=MUTED,
        ),
        "center": ParagraphStyle(
            "center", parent=base["Normal"], fontName=FONT, fontSize=7.6,
            alignment=TA_CENTER, textColor=INK,
        ),
    }


def _table(data, col_widths, header_rows=1, zebra=True, pad=4):
    t = Table(data, colWidths=col_widths, repeatRows=header_rows)
    style = [
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("GRID", (0, 0), (-1, -1), 0.4, BORDER),
        ("LEFTPADDING", (0, 0), (-1, -1), pad),
        ("RIGHTPADDING", (0, 0), (-1, -1), pad),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]
    if header_rows:
        style += [
            ("BACKGROUND", (0, 0), (-1, header_rows - 1), LIGHT),
            ("TEXTCOLOR", (0, 0), (-1, header_rows - 1), NAVY),
        ]
    if zebra:
        for r in range(header_rows, len(data)):
            if (r - header_rows) % 2 == 1:
                style.append(("BACKGROUND", (0, r), (-1, r), LIGHT_2))
    t.setStyle(TableStyle(style))
    return t


# ─── Chart ───────────────────────────────────────────────────────────────────
_CHART_LOCK = threading.Lock()


def _sync_chart(result) -> io.BytesIO | None:
    sig = result.get("signals") or {}
    t = sig.get("time_points") or []
    a = sig.get("audio_energy") or []
    lip = sig.get("lip_mar") or []
    sync = sig.get("sync_curve") or []
    if len(t) < 2 or len(a) < 2:
        return None
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        logger.warning("matplotlib unavailable for report chart: %s", exc)
        return None

    with _CHART_LOCK:
        try:
            fig, ax = plt.subplots(figsize=(7.6, 2.75), dpi=170)
            # Shade suspicious intervals
            for iv in result.get("suspicious_intervals") or []:
                try:
                    ax.axvspan(
                        float(iv.get("start_sec", 0)), float(iv.get("end_sec", 0)),
                        color="#fdecea", alpha=0.85, linewidth=0, zorder=0,
                    )
                except Exception:
                    pass
            if len(a) == len(t):
                ax.plot(t, a, color="#2e74c0", linewidth=1.15, label="Audio energy", zorder=2)
            if len(lip) == len(t):
                ax.plot(t, lip, color="#102a43", linewidth=1.0, linestyle="--",
                        label="Lip opening (MAR)", zorder=2)
            if len(sync) == len(t):
                ax.plot(t, sync, color="#a15c00", linewidth=1.2, label="Sync curve", zorder=3)
            ax.set_xlim(float(t[0]), float(t[-1]) if float(t[-1]) > 0 else 1.0)
            ax.set_ylim(0.0, 1.0)
            ax.set_xlabel("Time (seconds)", fontsize=7.5, color="#243b53")
            ax.set_ylabel("Normalised value", fontsize=7.5, color="#243b53")
            ax.tick_params(labelsize=6.8, colors="#5b7085")
            ax.grid(True, color="#d9e7f5", linewidth=0.6, alpha=0.8)
            for spine in ("top", "right"):
                ax.spines[spine].set_visible(False)
            for spine in ("left", "bottom"):
                ax.spines[spine].set_color("#c3d9ee")
            ax.legend(loc="upper right", fontsize=6.6, frameon=False, ncol=3)
            fig.tight_layout(pad=0.4)
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=170, facecolor="white", bbox_inches="tight")
            buf.seek(0)
            return buf
        except Exception as exc:  # pragma: no cover
            logger.warning("Could not render sync chart: %s", exc)
            return None
        finally:
            try:
                plt.close("all")
            except Exception:
                pass


# ─── Section builders ────────────────────────────────────────────────────────
def _verdict_style(verdict: str):
    v = str(verdict or "").upper()
    if v == "DEEPFAKE":
        return RED, RED_BG, "DEEPFAKE - MANIPULATION DETECTED"
    if v == "REAL":
        return GREEN, GREEN_BG, "AUTHENTIC - NO MANIPULATION DETECTED"
    return AMBER, AMBER_BG, "INCONCLUSIVE - INSUFFICIENT EVIDENCE"


def _section_title(text, styles):
    return [
        Paragraph(_txt(text), styles["h2"]),
        HRFlowable(width="100%", thickness=0.7, color=BORDER, spaceBefore=0, spaceAfter=5),
    ]


def _kv_grid(rows, styles, content_width):
    """Two label/value pairs per row, compact metadata layout."""
    data = []
    for i in range(0, len(rows), 2):
        left = rows[i]
        right = rows[i + 1] if i + 1 < len(rows) else ("", "")
        data.append([
            Paragraph(_txt(left[0]), styles["table_bold"]),
            Paragraph(_txt(left[1]), styles["table"]),
            Paragraph(_txt(right[0]), styles["table_bold"]),
            Paragraph(_txt(right[1]), styles["table"]),
        ])
    widths = [0.17 * content_width, 0.33 * content_width,
              0.17 * content_width, 0.33 * content_width]
    return _table(data, widths, header_rows=0, pad=4)


def _prob_row(label, value, colour, styles, content_width):
    return Table(
        [[
            Paragraph(_txt(label), styles["table"]),
            ProbabilityBar(value, colour, width=content_width * 0.55),
            Paragraph(_pct(value), styles["table_bold"]),
        ]],
        colWidths=[content_width * 0.27, content_width * 0.60, content_width * 0.13],
        style=TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING", (0, 0), (-1, -1), 0),
            ("RIGHTPADDING", (0, 0), (-1, -1), 3),
            ("TOPPADDING", (0, 0), (-1, -1), 2),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
        ]),
    )


# ─── Main builder ────────────────────────────────────────────────────────────
def build_report_pdf(result: dict, filename: str = None, sha256: str = None,
                     elapsed_s: float = None, job_id: str = None) -> bytes:
    """Return the complete report as PDF bytes."""
    styles = _styles()
    result = result or {}
    meta = result.get("video_meta") or {}
    verdict = str(result.get("verdict") or "INCONCLUSIVE").upper()
    v_colour, v_bg, v_label = _verdict_style(verdict)
    is_demo = str(result.get("model_used") or "").lower() == "demo"

    now = datetime.now()
    shorthash = (sha256 or "")[:10]
    report_id = "DG-" + now.strftime("%Y%m%d-%H%M%S") + ("-" + shorthash if shorthash else "")
    generated = now.strftime("%d %b %Y, %H:%M:%S")

    filename = filename or meta.get("filename") or "video"
    sha256 = sha256 or "not available"
    elapsed_s = _safe_float(elapsed_s if elapsed_s is not None else result.get("_elapsed_s"), 0.0)

    content_w = A4[0] - 36 * mm  # 36 mm total horizontal margin

    story = []

    # ── Header ──────────────────────────────────────────────────────────────
    story.append(Paragraph("RAVIS", styles["brand"]))
    story.append(Paragraph("Forensic Deepfake Analysis Report", styles["title"]))
    story.append(Paragraph(
        f"Report ID: <b>{_txt(report_id)}</b> &nbsp;&nbsp;|&nbsp;&nbsp; "
        f"Generated: {_txt(generated)} &nbsp;&nbsp;|&nbsp;&nbsp; "
        f"Classification: CONFIDENTIAL - FORENSIC USE",
        styles["subtitle"],
    ))
    story.append(Spacer(1, 5))
    story.append(HRFlowable(width="100%", thickness=1.3, color=BLUE, spaceAfter=8))

    if is_demo:
        story.append(Table(
            [[Paragraph(
                "DEMO MODE - The data in this report is illustrative and was not produced "
                "by a real forensic analysis.",
                styles["table_bold"],
            )]],
            colWidths=[content_w],
            style=TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), AMBER_BG),
                ("BOX", (0, 0), (-1, -1), 0.8, AMBER),
                ("LEFTPADDING", (0, 0), (-1, -1), 7),
                ("RIGHTPADDING", (0, 0), (-1, -1), 7),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]),
        ))
        story.append(Spacer(1, 8))

    # ── Verdict panel ───────────────────────────────────────────────────────
    confidence = _safe_float(result.get("confidence"), 0.5)
    reliability = result.get("reliability_grade") or "n/a"
    category = result.get("category") or (
        "Deepfake Manipulated" if verdict == "DEEPFAKE"
        else "Authentic Human" if verdict == "REAL"
        else "Inconclusive — Insufficient or Conflicting Evidence"
    )
    vs = ParagraphStyle(
        "v", parent=styles["verdict"], textColor=v_colour, fontSize=20, leading=23,
    )
    verdict_left = [
        Paragraph("VERDICT", styles["table_head"]),
        Paragraph(_txt(verdict if verdict != "INCONCLUSIVE" else "INCONCLUSIVE"), vs),
    ]
    conf_text = "n/a (verdict withheld)" if verdict == "INCONCLUSIVE" else _pct(confidence)
    verdict_right = [
        Paragraph(f"Category: <b>{_txt(category)}</b>", styles["body"]),
        Paragraph(
            f"Confidence: <b>{_txt(conf_text)}</b> &nbsp;&nbsp; "
            f"Reliability: <b>{_txt(reliability)}</b> &nbsp;&nbsp; "
            f"Model: <b>{_txt(result.get('model_used') or 'n/a')}</b>",
            styles["verdict_sub"],
        ),
    ]
    verdict_box = Table(
        [[verdict_left, verdict_right]],
        colWidths=[content_w * 0.43, content_w * 0.57],
        style=TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), v_bg),
            ("BOX", (0, 0), (-1, -1), 1.0, v_colour),
            ("LINEAFTER", (0, 0), (0, 0), 0.6, v_colour),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING", (0, 0), (-1, -1), 9),
            ("RIGHTPADDING", (0, 0), (-1, -1), 9),
            ("TOPPADDING", (0, 0), (-1, -1), 7),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ]),
    )
    story.append(verdict_box)
    story.append(Spacer(1, 6))

    # Probability block
    model_p = result.get("model_probability")
    physics_p = (result.get("physics_evidence") or {}).get("fake_probability")
    fused_p = _safe_float(result.get("deepfake_probability"), 0.5)
    story.append(_prob_row("Fused manipulation probability", fused_p, v_colour, styles, content_w))
    if model_p is not None:
        story.append(_prob_row("Random Forest probability", _safe_float(model_p), BLUE, styles, content_w))
    if physics_p is not None:
        story.append(_prob_row("Physics-based evidence probability", _safe_float(physics_p), AMBER, styles, content_w))
    story.append(Spacer(1, 7))

    # Decision reason
    decision_reason = result.get("decision_reason")
    if decision_reason:
        story.append(Paragraph("<b>Decision basis.</b> " + _txt(decision_reason), styles["body"]))

    # Summary
    explanation = result.get("explanation") or {}
    summary = explanation.get("summary") if isinstance(explanation, dict) else None
    if summary:
        story.append(Paragraph("<b>Executive summary.</b> " + _txt(summary), styles["body"]))
    story.append(Spacer(1, 2))

    # ── Video thumbnails ────────────────────────────────────────────────────
    thumb_list = result.get("thumbnail_b64_list") or []
    if thumb_list:
        import base64 as _b64
        story += _section_title("Video Frames", styles)
        story.append(Paragraph(
            "Representative frames extracted from the analysed video "
            "(approximately 10%, 40% and 70% through the clip).",
            styles["body"],
        ))
        story.append(Spacer(1, 4))

        # Build a single row of up to 3 thumbnail images
        thumb_cells = []
        thumb_w = (content_w - 6 * mm) / min(len(thumb_list), 3)   # share width equally
        thumb_h = thumb_w * (9 / 16)                                 # 16:9 aspect

        for idx, b64str in enumerate(thumb_list[:3]):
            try:
                raw = _b64.b64decode(b64str)
                img = Image(io.BytesIO(raw), width=thumb_w, height=thumb_h)
                label = Paragraph(
                    f"Frame {idx + 1} (~{(idx * 30) + 10}% of clip)",
                    styles["tiny"],
                )
                thumb_cells.append([img, label])
            except Exception as exc:
                logger.warning("Could not decode thumbnail %d: %s", idx, exc)

        if thumb_cells:
            # Each cell stacks [image, caption] vertically; cells are placed
            # side-by-side in a single-row table.
            col_w = (content_w - (len(thumb_cells) - 1) * 3 * mm) / len(thumb_cells)
            row_data = [[cell[0] for cell in thumb_cells]]
            cap_data = [[cell[1] for cell in thumb_cells]]
            col_widths = [col_w] * len(thumb_cells)

            img_table = Table(
                row_data,
                colWidths=col_widths,
                style=TableStyle([
                    ("VALIGN",       (0, 0), (-1, -1), "MIDDLE"),
                    ("ALIGN",        (0, 0), (-1, -1), "CENTER"),
                    ("LEFTPADDING",  (0, 0), (-1, -1), 2),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 2),
                    ("TOPPADDING",   (0, 0), (-1, -1), 2),
                    ("BOTTOMPADDING",(0, 0), (-1, -1), 2),
                    ("BOX",          (0, 0), (-1, -1), 0.5, BORDER),
                    ("INNERGRID",    (0, 0), (-1, -1), 0.4, BORDER),
                ]),
            )
            cap_table = Table(
                cap_data,
                colWidths=col_widths,
                style=TableStyle([
                    ("ALIGN",        (0, 0), (-1, -1), "CENTER"),
                    ("LEFTPADDING",  (0, 0), (-1, -1), 2),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 2),
                    ("TOPPADDING",   (0, 0), (-1, -1), 1),
                    ("BOTTOMPADDING",(0, 0), (-1, -1), 3),
                ]),
            )
            story.append(KeepTogether([img_table, cap_table]))
            story.append(Spacer(1, 6))

    # ── File & acquisition ──────────────────────────────────────────────────
    story += _section_title("1. File and Acquisition Details", styles)
    file_size = meta.get("file_size_mb")
    file_size_txt = f"{_safe_float(file_size):.2f} MB" if file_size is not None else "n/a"
    resolution = "n/a"
    if meta.get("width") and meta.get("height"):
        resolution = f"{int(meta['width'])} x {int(meta['height'])} px"
    duration = meta.get("duration")
    analyzed_duration = meta.get("analyzed_duration")
    rows = [
        ("File name", filename),
        ("SHA-256", sha256),
        ("File size", file_size_txt),
        ("Container duration", f"{_num(duration, 2)} s" if duration is not None else "n/a"),
        ("Resolution", resolution),
        ("Container frame rate", f"{_num(meta.get('fps'), 2)} fps" if meta.get("fps") else "n/a"),
        ("Analysis sample rate", f"{_num(meta.get('analysis_fps'), 2)} fps" if meta.get("analysis_fps") else "n/a"),
        ("Frames analysed", str(meta.get("analyzed_frames") or "n/a")),
        ("Analysed span", f"{_num(analyzed_duration, 2)} s" if analyzed_duration is not None else "n/a"),
        ("Audio duration", f"{_num(meta.get('audio_duration'), 2)} s" if meta.get("audio_duration") is not None else "n/a"),
        ("Analysis elapsed", f"{elapsed_s:.1f} s" if elapsed_s else "n/a"),
        ("Server job ID", job_id or "n/a"),
        ("Report timestamp", result.get("timestamp_generated") or generated),
        ("Report ID", report_id),
    ]
    story.append(_kv_grid(rows, styles, content_w))
    story.append(Spacer(1, 6))

    # ── Evidence sufficiency ────────────────────────────────────────────────
    story += _section_title("2. Evidence Sufficiency", styles)
    eq = result.get("evidence_quality") or {}
    if eq:
        informative = bool(eq.get("informative"))
        eq_colour = GREEN if informative else AMBER
        eq_bg = GREEN_BG if informative else AMBER_BG
        eq_table = Table(
            [[
                Paragraph(
                    "SUFFICIENT" if informative else "INSUFFICIENT",
                    ParagraphStyle("eq", parent=styles["table_bold"], textColor=eq_colour),
                ),
                Paragraph(
                    f"Evidence quality score: <b>{_pct(eq.get('score'), 1)}</b> &nbsp; "
                    f"Active speech: <b>{_txt((eq.get('metrics') or {}).get('active_speech_seconds', 'n/a'))} s</b>",
                    styles["table"],
                ),
            ]],
            colWidths=[content_w * 0.22, content_w * 0.78],
            style=TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), eq_bg),
                ("BOX", (0, 0), (-1, -1), 0.7, eq_colour),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 7),
                ("RIGHTPADDING", (0, 0), (-1, -1), 7),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]),
        )
        story.append(eq_table)
        flags = eq.get("flags") or []
        if flags:
            story.append(Spacer(1, 4))
            story.append(Paragraph("Evidence limitations", styles["h3"]))
            for f in flags:
                story.append(Paragraph(_txt(f), styles["body"], bulletText="-"))
        metrics = eq.get("metrics") or {}
        if metrics:
            mrows = []
            for k, label in [
                ("duration_s", "Clip duration (s)"),
                ("face_rate", "Face detection rate"),
                ("speech_activity", "Speech activity"),
                ("lip_activity", "Lip activity ratio"),
                ("mar_range", "Mouth aspect range"),
                ("active_speech_seconds", "Active speech (s)"),
                ("xcorr_peak", "Near-zero xcorr"),
                ("lag_ms", "Measured offset (ms)"),
            ]:
                if k in metrics:
                    value = metrics[k]
                    if k in ("face_rate", "speech_activity", "lip_activity"):
                        value = _pct(value)
                    mrows.append((label, value))
            if mrows:
                story.append(Spacer(1, 3))
                story.append(_kv_grid(mrows, styles, content_w))
    else:
        story.append(Paragraph(
            "Evidence-sufficiency metadata is not available for this report "
            "(demo/legacy result).",
            styles["body"],
        ))
    story.append(Spacer(1, 6))

    # ── Key metrics ─────────────────────────────────────────────────────────
    story += _section_title("3. Key Forensic Metrics", styles)
    metrics_rows = [[
        Paragraph("Metric", styles["table_head"]),
        Paragraph("Value", styles["table_head"]),
        Paragraph("Interpretation", styles["table_head"]),
    ]]
    sync_score = _safe_float(result.get("sync_score"), 0.5)
    xcorr = _safe_float(result.get("xcorr_peak"), 0.0)
    lag_ms = _safe_float(result.get("audio_lip_delay_ms"), 0.0)
    dtw = _safe_float(result.get("dtw_distance"), 0.0)
    face_rate = _safe_float(result.get("face_detection_rate"), 0.0)
    suspicious_frac = _safe_float((result.get("evidence_quality") or {}).get("metrics", {}).get("suspicious_fraction"),
                                  _safe_float(result.get("suspicious_window_fraction"), 0.0))
    active_sec = (result.get("evidence_quality") or {}).get("metrics", {}).get("active_speech_seconds", "n/a")

    def metric(label, value, note):
        metrics_rows.append([
            Paragraph(_txt(label), styles["table_bold"]),
            Paragraph(_txt(value), styles["table"]),
            Paragraph(_txt(note), styles["table"]),
        ])

    metric("Audio-visual sync score", _num(sync_score, 3),
           "Higher is better. Natural speech is typically above 0.50-0.55.")
    metric("Near-zero cross-correlation", _num(xcorr, 3),
           "Correlation at the in-sync position; <0.15 is weak.")
    metric("Audio-lip offset", _ms(lag_ms),
           "Natural co-articulation stays within roughly 150 ms.")
    metric("DTW distance", _num(dtw, 4),
           "Lower is better; high values mean even time-warping cannot align the signals.")
    metric("Face detection rate", _pct(face_rate),
           "Share of sampled frames in which a face was tracked.")
    metric("Suspicious window fraction", _pct(suspicious_frac),
           "Share of active speech windows scoring below 0.35.")
    metric("Active speech duration", f"{_txt(active_sec)} s",
           "Only these seconds carry usable lip-sync evidence.")
    metric("Fused manipulation probability", _pct(fused_p),
           "Combined Random Forest + physics evidence (45% / 55%).")
    metric("Random Forest probability", _pct(model_p) if model_p is not None else "n/a",
           "Learned classifier output (200 trees, 74 features).")
    metric("Physics evidence probability", _pct(physics_p) if physics_p is not None else "n/a",
           "Calibrated A/V-consistency score for this clip.")
    metric("Reliability", _txt(reliability),
           "High / Medium / Low based on face, audio and duration coverage.")
    story.append(_table(metrics_rows, [content_w * 0.28, content_w * 0.18, content_w * 0.54]))
    story.append(Spacer(1, 6))

    # ── Sync chart ──────────────────────────────────────────────────────────
    # Keep the heading, introduction and chart on one page so the section
    # never starts with an orphaned title at the bottom of a page.
    chart_block = _section_title("4. Audio-Visual Synchronisation Analysis", styles)
    chart_block.append(Paragraph(
        "The chart overlays the audio energy envelope, the normalised mouth-opening "
        "signal (MAR) and the per-sample synchronisation curve. Shaded regions are "
        "seconds classified as suspicious. Both signals were resampled to a shared "
        "25 Hz timebase before correlation.",
        styles["body"],
    ))
    chart = _sync_chart(result)
    if chart is not None:
        chart_block.append(Image(chart, width=content_w, height=content_w * 0.36))
        chart_block.append(Paragraph(
            "Figure 1. Audio energy (blue), lip opening (dashed navy) and sync curve "
            "(amber) over the analysed span.",
            styles["tiny"],
        ))
    else:
        chart_block.append(Paragraph("Signal chart unavailable for this report.", styles["body"]))
    chart_block.append(Spacer(1, 5))
    story.append(KeepTogether(chart_block))

    # Timeline summary
    timeline = result.get("timeline") or []
    if timeline:
        good = sum(1 for t in timeline if t.get("status") == "good")
        warn = sum(1 for t in timeline if t.get("status") == "warn")
        bad = sum(1 for t in timeline if t.get("status") == "bad")
        total = max(1, len(timeline))
        story.append(Paragraph(
            f"<b>Per-second classification:</b> {good}/{total} seconds in sync "
            f"({_pct(good / total)}), {warn} marginal, {bad} desynchronised.",
            styles["body"],
        ))

    # Suspicious intervals
    intervals = result.get("suspicious_intervals") or []
    if intervals:
        story.append(Paragraph("Detected suspicious intervals", styles["h3"]))
        irows = [[
            Paragraph("Start", styles["table_head"]),
            Paragraph("End", styles["table_head"]),
            Paragraph("Severity", styles["table_head"]),
            Paragraph("Reason", styles["table_head"]),
            Paragraph("Min sync", styles["table_head"]),
        ]]
        for iv in intervals[:25]:
            irows.append([
                Paragraph(_txt(iv.get("start_timestamp", iv.get("start_sec", ""))), styles["table"]),
                Paragraph(_txt(iv.get("end_timestamp", iv.get("end_sec", ""))), styles["table"]),
                Paragraph(_txt(iv.get("severity", "")), styles["table_bold"]),
                Paragraph(_txt(iv.get("reason", "")), styles["table"]),
                Paragraph(_num(iv.get("min_score"), 3), styles["table"]),
            ])
        story.append(_table(irows, [content_w * 0.12, content_w * 0.12,
                                    content_w * 0.16, content_w * 0.44, content_w * 0.16]))
        if len(intervals) > 25:
            story.append(Paragraph(
                f"{len(intervals) - 25} additional intervals are present in the digital "
                "result and were omitted from this printed table.",
                styles["tiny"],
            ))
    else:
        story.append(Paragraph("No sustained suspicious interval was detected.", styles["body"]))
    story.append(Spacer(1, 5))

    # ── Evidence board ──────────────────────────────────────────────────────
    story.append(PageBreak())
    story += _section_title("5. Evidence Board", styles)
    story.append(Paragraph(
        "Each card summarises one family of forensic measurements. Status reflects the "
        "final verdict; scores are risk indicators on a 0-100 scale.",
        styles["body"],
    ))
    cards = result.get("evidence_cards") or []
    if not cards:
        story.append(Paragraph(
            "Detailed evidence cards are not available for this result "
            "(demo/legacy analysis).",
            styles["body"],
        ))
    for card in cards:
        status = str(card.get("status") or "INCONCLUSIVE").upper()
        if status == "CRITICAL":
            sc, sbg = RED, RED_BG
        elif status == "SUSPICIOUS":
            sc, sbg = AMBER, AMBER_BG
        elif status == "INCONCLUSIVE":
            sc, sbg = AMBER, AMBER_BG
        else:
            sc, sbg = GREEN, GREEN_BG

        header = Table(
            [[
                Paragraph(f"<b>{_txt(card.get('title') or card.get('id'))}</b>", styles["h3"]),
                Paragraph(
                    f"<b>{_txt(status)}</b>",
                    ParagraphStyle("pill", parent=styles["table_bold"], textColor=sc, alignment=TA_CENTER),
                ),
                Paragraph(
                    ("Risk n/a &nbsp; Confidence n/a" if status == "INCONCLUSIVE" else
                     f"Risk {_num(card.get('score'), 1)} / 100 &nbsp; "
                     f"Confidence {_pct(card.get('confidence'))}"),
                    styles["table"],
                ),
            ]],
            colWidths=[content_w * 0.40, content_w * 0.16, content_w * 0.44],
            style=TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), sbg),
                ("BOX", (0, 0), (-1, -1), 0.6, sc),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]),
        )
        body_bits = [
            Paragraph(
                f"<b>{_txt(card.get('headline') or '')}</b>",
                ParagraphStyle("hl", parent=styles["table"], fontName=FONT_BOLD, fontSize=8.2),
            ),
            Paragraph(_txt(card.get("description") or ""), styles["small"]),
            Spacer(1, 2),
        ]
        m = card.get("metrics") or []
        if m:
            mdata = [[
                Paragraph(_txt(x.get("label", "")), styles["table_head"]),
                Paragraph(_txt(x.get("value", "")), styles["table"]),
            ] for x in m]
            body_bits.append(_table(mdata, [content_w * 0.32, content_w * 0.68], header_rows=0, pad=3))
        chart_data = card.get("chart_data") or []
        if chart_data:
            body_bits.append(Spacer(1, 3))
            body_bits.append(Sparkline(chart_data, colour=sc, width=content_w * 0.55, height=22))
        block = Table(
            [[header], [body_bits]],
            colWidths=[content_w],
            style=TableStyle([
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
                ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                ("TOPPADDING", (0, 0), (-1, -1), 0),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
            ]),
        )
        story.append(KeepTogether([block, Spacer(1, 7)]))

    # ── Reasoning ───────────────────────────────────────────────────────────
    story += _section_title("6. Reasoning Trace", styles)
    reasons = explanation.get("reasons") if isinstance(explanation, dict) else None
    warnings = explanation.get("warnings") if isinstance(explanation, dict) else None
    if reasons:
        for r in reasons:
            story.append(Paragraph(
                f"<b>{_txt(r.get('title') or 'Finding')}.</b> {_txt(r.get('detail') or '')}",
                styles["body"], bulletText="-",
            ))
    else:
        story.append(Paragraph("No structured reasoning trace was supplied.", styles["body"]))
    if warnings:
        story.append(Paragraph("Warnings and limitations", styles["h3"]))
        for w in warnings:
            story.append(Paragraph(
                f"<b>{_txt(w.get('title') or 'Warning')}.</b> {_txt(w.get('detail') or '')}",
                styles["body"], bulletText="-",
            ))
    if result.get("attack_chips"):
        story.append(Paragraph("Detected indicators", styles["h3"]))
        chips = ", ".join(_txt(c.get("tag")) for c in result["attack_chips"] if c.get("tag"))
        story.append(Paragraph(chips or "n/a", styles["body"]))

    # ── Model & methodology ─────────────────────────────────────────────────
    story += _section_title("7. Model, Method and Limitations", styles)
    story.append(Paragraph(
        "<b>Pipeline.</b> The uploaded file is validated, audio is decoded to 16 kHz mono, "
        "and frames are sampled evenly across the whole clip (up to 1,200 frames). "
        "MediaPipe Face Mesh tracks 468 facial landmarks per frame and derives the mouth "
        "aspect ratio (MAR). Audio RMS energy and MAR are resampled onto a shared 25 Hz "
        "grid; normalised cross-correlation, dynamic time warping and per-second "
        "correlation are then computed. Only seconds with active speech contribute to "
        "the local sync statistics.",
        styles["body"],
    ))
    story.append(Paragraph(
        "<b>Classifier.</b> A Random Forest with 200 trees and 74 features produces a "
        "manipulation probability. Because that model is not calibrated for every camera "
        "or codec, its output is fused with a transparent physics score derived from the "
        "measured correlation, offset, DTW and suspicious-window values (45% model / 55% "
        "physics). A verdict is only issued when the fused probability is outside the "
        "uncertainty band; otherwise the result is INCONCLUSIVE.",
        styles["body"],
    ))
    thresholds = result.get("decision_thresholds") or {}
    if thresholds:
        trows = [[
            Paragraph("Decision rule", styles["table_head"]),
            Paragraph("Value", styles["table_head"]),
        ]]
        trows.append([Paragraph("DEEPFAKE threshold", styles["table"]),
                      Paragraph(f">= {_pct(thresholds.get('fake'))}", styles["table_bold"])])
        trows.append([Paragraph("REAL threshold", styles["table"]),
                      Paragraph(f"<= {_pct(thresholds.get('real'))}", styles["table_bold"])])
        if thresholds.get("model_weight") is not None:
            trows.append([Paragraph("Random Forest weight", styles["table"]),
                          Paragraph(_pct(thresholds.get("model_weight")), styles["table"])])
            trows.append([Paragraph("Physics evidence weight", styles["table"]),
                          Paragraph(_pct(1 - _safe_float(thresholds.get("model_weight"), 0.45)), styles["table"])])
        story.append(_table(trows, [content_w * 0.55, content_w * 0.45], pad=4))
        story.append(Spacer(1, 4))

    top_features = result.get("top_features") or []
    if top_features:
        story.append(Paragraph("Top model features", styles["h3"]))
        frows = [[
            Paragraph("#", styles["table_head"]),
            Paragraph("Feature", styles["table_head"]),
            Paragraph("Importance", styles["table_head"]),
        ]]
        for i, item in enumerate(top_features[:10], 1):
            try:
                name, importance = item[0], item[1]
            except Exception:
                name, importance = str(item), 0
            frows.append([
                Paragraph(str(i), styles["table"]),
                Paragraph(_txt(name), styles["table"]),
                Paragraph(_num(importance, 4), styles["table"]),
            ])
        story.append(_table(frows, [content_w * 0.08, content_w * 0.62, content_w * 0.30], pad=4))
        story.append(Spacer(1, 4))

    story.append(Paragraph("Limitations", styles["h3"]))
    for limitation in [
        "The analysis uses audio-visual synchronisation and facial-motion cues. It does not "
        "perform pixel-level generative-artifact detection or speaker verification.",
        "Music, voice-over, dubbing, translation, multi-speaker scenes, heavy occlusion or a "
        "face that is not the speaker can reduce the available evidence; such clips may be "
        "reported as INCONCLUSIVE rather than REAL or DEEPFAKE.",
        "A constant container-level audio offset is treated as a reliability limitation, not "
        "automatically as manipulation.",
        "A DEEPFAKE verdict is statistical evidence, not legal proof. Use it as decision "
        "support alongside expert review, source provenance and chain-of-custody checks.",
        "The Random Forest was trained outside this deployment; its probabilities are fused "
        "with the measured physics evidence to reduce unsupported binary calls.",
    ]:
        story.append(Paragraph(_txt(limitation), styles["body"], bulletText="-"))
    story.append(Spacer(1, 5))

    # ── Chain of custody ────────────────────────────────────────────────────
    story += _section_title("8. Chain of Custody and Integrity", styles)
    cust_rows = [
        ("Evidence file", filename),
        ("SHA-256 digest", sha256),
        ("Digest source", "Computed in the analyst's browser from the uploaded bytes"),
        ("Analysis engine", "RAVIS forensic pipeline (Flask)"),
        ("Server job ID", job_id or "n/a"),
        ("Result timestamp", result.get("timestamp_generated") or "n/a"),
        ("Report generated", generated),
        ("Operator", "Automated - RAVIS"),
    ]
    story.append(_kv_grid(cust_rows, styles, content_w))
    story.append(Spacer(1, 12))
    sig = Table(
        [[
            Paragraph("Analyst signature: ______________________________", styles["body"]),
            Paragraph("Date: ________________", styles["body"]),
        ], [
            Paragraph("Reviewer signature: _____________________________", styles["body"]),
            Paragraph("Date: ________________", styles["body"]),
        ]],
        colWidths=[content_w * 0.68, content_w * 0.32],
        style=TableStyle([
            ("LEFTPADDING", (0, 0), (-1, -1), 0),
            ("RIGHTPADDING", (0, 0), (-1, -1), 0),
            ("TOPPADDING", (0, 0), (-1, -1), 12),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
        ]),
    )
    story.append(sig)
    story.append(Spacer(1, 10))

    # ── Disclaimer ──────────────────────────────────────────────────────────
    story.append(Paragraph(
        "<b>Disclaimer.</b> This report was produced by an automated decision-support "
        "system. It does not by itself establish that a person, organisation or device "
        "created or distributed manipulated media. Findings must be interpreted by a "
        "qualified examiner in the context of the investigation, and the original evidence "
        "must be preserved unaltered. A REAL verdict does not prove authenticity; an "
        "INCONCLUSIVE verdict does not prove that the media is authentic or manipulated.",
        styles["small"],
    ))
    story.append(Spacer(1, 10))
    story.append(HRFlowable(width="100%", thickness=0.6, color=BORDER, spaceAfter=4))
    story.append(Paragraph(
        "End of report. This page and every preceding page are part of the same report ID.",
        styles["tiny"],
    ))

    # ── Build ───────────────────────────────────────────────────────────────
    buf = io.BytesIO()
    doc = BaseDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=17 * mm,
        bottomMargin=15 * mm,
        title=f"RAVIS Forensic Report {report_id}",
        author="RAVIS",
        subject="Deepfake forensic analysis report",
        creator="RAVIS forensic pipeline",
    )
    frame = Frame(
        doc.leftMargin, doc.bottomMargin, doc.width, doc.height,
        leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0, id="main",
    )

    def _on_page(canv, doc_obj):
        canv.saveState()
        # Top rule + running header
        canv.setStrokeColor(BORDER)
        canv.setLineWidth(0.6)
        canv.line(18 * mm, A4[1] - 12 * mm, A4[0] - 18 * mm, A4[1] - 12 * mm)
        canv.setFont(FONT, 7.0)
        canv.setFillColor(MUTED)
        canv.drawString(18 * mm, A4[1] - 10.5 * mm, "RAVIS  |  Forensic Deepfake Analysis Report")
        canv.drawRightString(A4[0] - 18 * mm, A4[1] - 10.5 * mm, report_id)
        # Footer
        canv.setStrokeColor(BORDER)
        canv.line(18 * mm, 14 * mm, A4[0] - 18 * mm, 14 * mm)
        canv.setFont(FONT, 6.8)
        canv.drawString(18 * mm, 10 * mm, "CONFIDENTIAL - FORENSIC USE ONLY")
        canv.drawCentredString(A4[0] / 2, 10 * mm, "Generated by RAVIS")
        canv.restoreState()

    doc.addPageTemplates([PageTemplate(id="main", frames=[frame], onPage=_on_page)])
    doc.build(story, canvasmaker=NumberedCanvas)
    return buf.getvalue()
