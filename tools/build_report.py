#!/usr/bin/env python3
"""Build the Vector2Mars task-2 PDF from the Markdown report.

Usage:
    python3 -m pip install -r requirements-report.txt
    python3 tools/build_report.py
"""

from __future__ import annotations

import html
import re
from pathlib import Path

import markdown
from bs4 import BeautifulSoup, NavigableString, Tag
from weasyprint import HTML

ROOT = Path(__file__).resolve().parent.parent
MD_PATH = ROOT / "vector-to-mars.md"
OUT_PATH = ROOT / "vector-to-mars.pdf"

text = MD_PATH.read_text(encoding="utf-8")
lines = text.splitlines()
body_start = next(i for i, line in enumerate(lines) if line.startswith("## "))

title = lines[0].lstrip("# ").strip()
subtitle = re.sub(r"^\*\*|\*\*$", "", lines[2].strip())
intro_md = "\n".join(lines[4:body_start]).strip()
body_md = "\n".join(lines[body_start:]).strip()


def normalize_tables(source: str) -> str:
    """Python-Markdown needs blank lines around pipe tables."""
    result: list[str] = []
    in_fence = False
    for line in source.splitlines():
        if line.strip().startswith("```"):
            in_fence = not in_fence
            result.append(line)
            continue
        if not in_fence and line.startswith("|"):
            if result and result[-1].strip() and not result[-1].startswith("|"):
                result.append("")
        elif not in_fence and result and result[-1].startswith("|") and line.strip():
            result.append("")
        result.append(line)
    return "\n".join(result)


body_md = normalize_tables(body_md)

md = markdown.Markdown(extensions=["tables", "fenced_code", "sane_lists"])
body_html = md.convert(body_md)
intro_html = markdown.markdown(intro_md, extensions=["sane_lists"])

soup = BeautifulSoup(f'<div id="report-root">{body_html}</div>', "html.parser")
container = soup.find("div", id="report-root")
assert container is not None


def is_blank(node: object) -> bool:
    return isinstance(node, NavigableString) and not node.strip()


def image_paragraph(node: object) -> bool:
    if not isinstance(node, Tag) or node.name != "p":
        return False
    kids = [c for c in node.children if not is_blank(c)]
    if not kids:
        return False
    if not any(isinstance(c, Tag) and c.name == "img" for c in kids):
        return False
    return all(isinstance(c, Tag) and c.name in {"img", "br"} for c in kids)


def caption_paragraph(node: object) -> bool:
    if not isinstance(node, Tag) or node.name != "p":
        return False
    return node.get_text(" ", strip=True).startswith("Рис.")


def build_figure(images: list[Tag], caption: Tag | None) -> Tag:
    cols = 3 if len(images) >= 6 else (2 if len(images) >= 2 else 1)
    figure = soup.new_tag("figure")
    figure["class"] = f"fig cols-{cols}"
    if any("field_route" in (image.get("src") or "") for image in images):
        figure["class"] += " tall"
    if any("flight-working" in (image.get("src") or "") for image in images):
        figure["class"] += " raw"
    if any("pipeline" in (image.get("src") or "") for image in images):
        figure["class"] += " wide"

    table = soup.new_tag("table")
    table["class"] = "img-grid"
    tbody = soup.new_tag("tbody")
    for start in range(0, len(images), cols):
        row = soup.new_tag("tr")
        for image in images[start : start + cols]:
            cell = soup.new_tag("td")
            cell["style"] = f"width: {100 / cols:.4f}%"
            cell.append(image.extract())
            row.append(cell)
        tbody.append(row)
    table.append(tbody)
    figure.append(table)

    if caption is not None:
        figcaption = soup.new_tag("figcaption")
        figcaption.string = caption.get_text(" ", strip=True)
        figure.append(figcaption)

    return figure


def group_figures(root: Tag) -> None:
    children = list(root.children)
    rebuilt: list[Tag | NavigableString] = []
    i = 0
    while i < len(children):
        node = children[i]
        if image_paragraph(node):
            images: list[Tag] = []
            j = i
            while j < len(children):
                if image_paragraph(children[j]):
                    images.extend(children[j].find_all("img"))
                    j += 1
                elif is_blank(children[j]):
                    j += 1
                else:
                    break
            k = j
            caption = (
                children[k]
                if k < len(children) and caption_paragraph(children[k])
                else None
            )
            rebuilt.append(build_figure(images, caption))
            i = k + 1 if caption is not None else j
            continue
        rebuilt.append(node)
        i += 1
    root.clear()
    for node in rebuilt:
        root.append(node)


group_figures(container)

for table in container.find_all("table"):
    classes = table.get("class") or []
    if "img-grid" not in classes:
        table["class"] = classes + ["data"]


def take_section(root: Tag, title: str) -> list[Tag]:
    """Detach one h2 section from the body so it can open the report."""
    heading = next(
        h for h in root.find_all("h2") if h.get_text(" ", strip=True) == title
    )
    nodes: list[Tag] = [heading]
    for sibling in list(heading.next_siblings):
        if isinstance(sibling, Tag) and sibling.name == "h2":
            break
        nodes.append(sibling)
    for node in nodes:
        node.extract()
    return nodes


front_nodes = take_section(container, "Конвейер") + take_section(
    container, "Почему не машинное обучение"
)
front_html = "".join(str(node) for node in front_nodes)
front_soup = BeautifulSoup(f"<div>{front_html}</div>", "html.parser")
front_root = front_soup.find("div")
assert front_root is not None

# Give every chapter and subsection a deterministic anchor and collect the TOC.
toc: list[tuple[str, str, str]] = []
section = 0
subsection = 0
for heading in front_root.find_all(["h2", "h3"]) + container.find_all(["h2", "h3"]):
    label = heading.get_text(" ", strip=True)
    if heading.name == "h2":
        section += 1
        subsection = 0
        anchor = f"sec-{section}"
    else:
        subsection += 1
        anchor = f"sec-{section}-{subsection}"
    heading["id"] = anchor
    toc.append((heading.name, anchor, label))

NEW_PAGE_SECTIONS = {
    "Задача и исходные данные",
    "Что проверено на тестовых данных",
    "Бортовая программа",
}
for heading in container.find_all("h2"):
    if heading.get_text(" ", strip=True) in NEW_PAGE_SECTIONS:
        heading["class"] = (heading.get("class") or []) + ["newpage"]

toc_items = []
for kind, anchor, label in toc:
    css_class = "toc-h2" if kind == "h2" else "toc-h3"
    toc_items.append(
        f'<li class="{css_class}"><a href="#{anchor}">{html.escape(label)}</a></li>'
    )
toc_html = '<ol class="toc-list">' + "".join(toc_items) + "</ol>"

body_html_final = container.decode_contents()
front_html = front_root.decode_contents()

CSS = r"""
@page {
  size: A4;
  margin: 21mm 17mm 18mm 17mm;
  @top-left {
    content: "Вектор на Марс · Задание 2";
    font-family: "DejaVu Sans";
    font-size: 7.1pt;
    color: #7b8798;
    vertical-align: bottom;
    padding-bottom: 2.4mm;
  }
  @top-right {
    content: string(chapter, first);
    font-family: "DejaVu Sans";
    font-size: 7.1pt;
    color: #7b8798;
    vertical-align: bottom;
    padding-bottom: 2.4mm;
  }
  @bottom-left {
    content: "Распознавание маршрутов ровера";
    font-family: "DejaVu Sans";
    font-size: 6.6pt;
    color: #a5b0bf;
    vertical-align: top;
    padding-top: 2.4mm;
  }
  @bottom-right {
    content: counter(page);
    font-family: "DejaVu Sans";
    font-size: 7.8pt;
    color: #475569;
    vertical-align: top;
    padding-top: 2mm;
  }
}

@page cover {
  margin: 0;
  @top-left { content: none; }
  @top-right { content: none; }
  @bottom-left { content: none; }
  @bottom-right { content: none; }
}

@page summary {
  @top-left { content: none; }
  @top-right { content: none; }
}

@page toc {
  @top-left { content: none; }
  @top-right { content: none; }
}

html {
  font-family: "DejaVu Sans", sans-serif;
  color: #1f2937;
}

body {
  margin: 0;
}

.cover {
  page: cover;
  height: 297mm;
  box-sizing: border-box;
  padding: 27mm 24mm 20mm;
  color: #ffffff;
  background: linear-gradient(155deg, #070d18 0%, #0d1b31 45%, #173253 100%);
  position: relative;
  overflow: hidden;
  font-family: "Carlito", "DejaVu Sans", sans-serif;
  break-after: page;
}

.cover::before {
  content: "";
  position: absolute;
  right: -55mm;
  top: -55mm;
  width: 160mm;
  height: 160mm;
  border-radius: 50%;
  background: radial-gradient(circle, rgba(234, 88, 12, 0.38) 0%, rgba(234, 88, 12, 0) 68%);
}

.cover::after {
  content: "";
  position: absolute;
  left: 0;
  bottom: 0;
  width: 210mm;
  height: 78mm;
  background: linear-gradient(0deg, rgba(234, 88, 12, 0.10), rgba(234, 88, 12, 0));
}

.cover > * {
  position: relative;
  z-index: 1;
}

.cover-kicker {
  font-size: 9.2pt;
  letter-spacing: 3pt;
  text-transform: uppercase;
  color: #fb923c;
  font-weight: 700;
}

.cover h1 {
  font-size: 40pt;
  line-height: 1.03;
  margin: 12mm 0 6mm;
  letter-spacing: -0.6pt;
  font-weight: 800;
  max-width: 165mm;
  bookmark-level: none;
}

.cover-sub {
  font-size: 12.2pt;
  line-height: 1.38;
  color: #cbd5e1;
  max-width: 150mm;
  margin: 0;
}

.cover-rule {
  width: 28mm;
  height: 1.6mm;
  background: #ea580c;
  margin: 10mm 0 9mm;
  border-radius: 1mm;
}

.cover-answer {
  width: 128mm;
  border-left: 1.5mm solid #ea580c;
  background: rgba(255, 255, 255, 0.06);
  padding: 6mm 7mm;
  border-radius: 0 2mm 2mm 0;
}

.cover-answer .label {
  font-size: 8.3pt;
  letter-spacing: 1.5pt;
  text-transform: uppercase;
  color: #fdba74;
}

.cover-answer .big {
  font-size: 46pt;
  line-height: 1.0;
  font-weight: 800;
  margin: 2mm 0;
  color: #ffffff;
}

.cover-answer .meta {
  font-size: 11.8pt;
  color: #e2e8f0;
  font-weight: 600;
}

.cover-answer .note {
  font-size: 8.4pt;
  color: #94a3b8;
  margin-top: 2mm;
}

.cover-note {
  margin-top: 7mm;
  width: 128mm;
  font-size: 9pt;
  line-height: 1.5;
  color: #94a3b8;
}

.cover-bottom {
  position: absolute;
  left: 24mm;
  right: 24mm;
  bottom: 20mm;
  display: flex;
  justify-content: space-between;
  font-size: 8.4pt;
  color: #94a3b8;
  border-top: 0.5pt solid rgba(255, 255, 255, 0.15);
  padding-top: 4mm;
}

.summary {
  page: summary;
  break-before: page;
}

.summary h2,
.toc-title {
  font-family: "Carlito", "DejaVu Sans", sans-serif;
  font-size: 19pt;
  line-height: 1.15;
  color: #0f2a4a;
  margin: 0 0 5mm;
  padding-bottom: 2.5mm;
  border-bottom: 1.2pt solid #e2e8f0;
  bookmark-level: none;
}

.summary .lead {
  font-size: 10.2pt;
  line-height: 1.55;
  color: #334155;
}

.summary .lead p {
  margin: 0 0 3.4mm;
  text-align: justify;
}

.summary .scope-note {
  font-size: 9.1pt;
  line-height: 1.45;
  color: #475569;
  border-left: 1mm solid #e2e8f0;
  padding-left: 4mm;
  margin: 4mm 0 0;
}

.answer-box {
  margin: 7mm 0 6mm;
  padding: 5mm 6mm;
  background: #fff7ed;
  border-left: 1.4mm solid #ea580c;
  border-radius: 0 2mm 2mm 0;
}

.answer-box .small {
  font-size: 8pt;
  letter-spacing: 1.3pt;
  text-transform: uppercase;
  color: #9a3412;
}

.answer-box .value {
  font-family: "Carlito", "DejaVu Sans", sans-serif;
  font-size: 21pt;
  font-weight: 800;
  color: #9a3412;
  margin: 1mm 0;
}

.answer-box .sub {
  font-size: 9.7pt;
  color: #7c2d12;
}

.stat-row {
  display: flex;
  gap: 3.5mm;
  margin: 7mm 0 7mm;
}

.stat {
  flex: 1;
  border: 0.6pt solid #e2e8f0;
  background: #f8fafc;
  border-radius: 2mm;
  padding: 4mm;
}

.stat b {
  display: block;
  font-family: "Carlito", "DejaVu Sans", sans-serif;
  font-size: 19pt;
  color: #1e3a5f;
  line-height: 1.1;
}

.stat span {
  display: block;
  font-size: 7.8pt;
  color: #64748b;
  margin-top: 1mm;
}

.summary h3 {
  font-family: "Carlito", "DejaVu Sans", sans-serif;
  font-size: 12pt;
  color: #1e3a5f;
  margin: 7mm 0 2.5mm;
}

.summary ul {
  margin: 0 0 0 5mm;
  padding: 0;
}

.summary li {
  font-size: 9.4pt;
  line-height: 1.45;
  margin-bottom: 1.4mm;
}

.summary-body {
  font-size: 9.3pt;
  line-height: 1.45;
  color: #1f2937;
}

.summary-body h2 {
  font-family: "Carlito", "DejaVu Sans", sans-serif;
  font-size: 14.5pt;
  line-height: 1.2;
  color: #0f2a4a;
  margin: 6.5mm 0 3mm;
  padding-bottom: 2mm;
  border-bottom: 1pt solid #e2e8f0;
}

.summary-body h2::before {
  content: "";
  display: block;
  width: 10mm;
  height: 1.4mm;
  background: #ea580c;
  margin-bottom: 2.4mm;
  border-radius: 1mm;
}

.summary-body h3 {
  font-family: "Carlito", "DejaVu Sans", sans-serif;
  font-size: 11pt;
  color: #1e3a5f;
  margin: 4mm 0 1.6mm;
}

.summary-body p {
  margin: 0 0 2.6mm;
  text-align: justify;
}

.summary-body ul {
  margin: 0 0 2.6mm 5mm;
  padding: 0;
}

.summary-body li {
  margin-bottom: 1.2mm;
}

.summary-body strong {
  color: #0f172a;
}

.summary-body code {
  font-family: "DejaVu Sans Mono", monospace;
  font-size: 7.9pt;
  background: #f1f5f9;
  color: #9a3412;
  padding: 0.2mm 1mm;
  border-radius: 1mm;
}

.summary-body table.data {
  width: 100%;
  border-collapse: collapse;
  margin: 2mm 0 3mm;
  font-size: 7.9pt;
  line-height: 1.32;
  break-inside: avoid;
}

.summary-body table.data th {
  background: #1e3a5f;
  color: #ffffff;
  font-weight: 600;
  text-align: left;
  padding: 1.5mm 2mm;
  border: 0.5pt solid #1e3a5f;
}

.summary-body table.data td {
  padding: 1.3mm 2mm;
  border-bottom: 0.5pt solid #e2e8f0;
  vertical-align: top;
}

.summary-body table.data tr:nth-child(even) td {
  background: #f8fafc;
}

.summary-body table.data tr {
  break-inside: avoid;
}

.summary-body figure.fig {
  margin: 3mm 0 3mm;
}

.method-table {
  width: 100%;
  border-collapse: collapse;
  margin-top: 2mm;
}

.method-table td {
  width: 33.33%;
  vertical-align: top;
  background: #f8fafc;
  border: 0.6pt solid #e2e8f0;
  padding: 3mm;
}

.method-table b {
  display: block;
  font-family: "Carlito", "DejaVu Sans", sans-serif;
  font-size: 10pt;
  color: #1e3a5f;
  margin-bottom: 1mm;
}

.method-table span {
  display: block;
  font-size: 8.2pt;
  line-height: 1.4;
  color: #475569;
}

.toc-page {
  page: toc;
  break-before: page;
}

.toc-list {
  list-style: none;
  margin: 8mm 0 0;
  padding: 0;
}

.toc-list li {
  margin: 0;
}

.toc-h2 {
  font-size: 9.8pt;
  font-weight: 700;
  color: #0f2a4a;
  margin-top: 3.1mm !important;
}

.toc-h3 {
  font-size: 8.7pt;
  font-weight: 400;
  color: #475569;
  margin-left: 6mm !important;
  margin-top: 1.25mm !important;
}

.toc-list a {
  color: inherit;
  text-decoration: none;
}

.toc-list a::after {
  content: leader('.') target-counter(attr(href), page);
  color: #94a3b8;
  font-weight: 400;
}

#report-body {
  font-size: 9.55pt;
  line-height: 1.48;
  color: #1f2937;
}

#report-body h2 {
  string-set: chapter content();
  font-family: "Carlito", "DejaVu Sans", sans-serif;
  font-size: 18.5pt;
  line-height: 1.16;
  color: #0f2a4a;
  margin: 9mm 0 5mm;
  padding-bottom: 2.5mm;
  border-bottom: 1.2pt solid #e2e8f0;
  break-after: avoid;
}

#report-body h2.newpage {
  break-before: page;
  margin-top: 0;
}

#report-body h2::before {
  content: "";
  display: block;
  width: 12mm;
  height: 1.6mm;
  background: #ea580c;
  margin-bottom: 3mm;
  border-radius: 1mm;
}

#report-body h3 {
  font-family: "Carlito", "DejaVu Sans", sans-serif;
  font-size: 12.3pt;
  line-height: 1.25;
  color: #1e3a5f;
  margin: 6mm 0 2mm;
  break-after: avoid;
}

#report-body p {
  margin: 0 0 3.2mm;
  text-align: justify;
  orphans: 2;
  widows: 2;
}

#report-body strong {
  color: #0f172a;
}

#report-body code {
  font-family: "DejaVu Sans Mono", monospace;
  font-size: 8.1pt;
  background: #f1f5f9;
  color: #9a3412;
  padding: 0.2mm 1mm;
  border-radius: 1mm;
}

#report-body pre {
  background: #f8fafc;
  border: 0.6pt solid #e2e8f0;
  border-left: 1.4mm solid #ea580c;
  border-radius: 1.5mm;
  padding: 3mm 4mm;
  margin: 4mm 0 5mm;
  white-space: pre-wrap;
  word-wrap: break-word;
}

#report-body pre code {
  background: none;
  color: #1e293b;
  padding: 0;
  font-size: 8pt;
  line-height: 1.38;
}

#report-body ul,
#report-body ol {
  margin: 0 0 4mm 5mm;
  padding: 0;
}

#report-body li {
  margin-bottom: 1.2mm;
}

#report-body table.data {
  width: 100%;
  border-collapse: collapse;
  margin: 3mm 0 5mm;
  font-size: 8.25pt;
  line-height: 1.35;
}

#report-body table.data thead {
  display: table-header-group;
}

#report-body table.data th {
  background: #1e3a5f;
  color: #ffffff;
  font-weight: 600;
  text-align: left;
  padding: 2mm 2.2mm;
  border: 0.5pt solid #1e3a5f;
}

#report-body table.data td {
  padding: 1.65mm 2.2mm;
  border-bottom: 0.5pt solid #e2e8f0;
  vertical-align: top;
}

#report-body table.data tr:nth-child(even) td {
  background: #f8fafc;
}

#report-body table.data tr {
  break-inside: avoid;
}

figure.fig {
  margin: 4mm 0 5mm;
  break-inside: avoid;
}

figure.fig table.img-grid {
  width: 100%;
  table-layout: fixed;
  border-collapse: separate;
  border-spacing: 0;
  margin: 0;
}

figure.fig td {
  padding: 1.25mm;
  vertical-align: top;
}

figure.fig img {
  width: 100%;
  height: auto;
  border: 0.5pt solid #dbe3ee;
  border-radius: 1.5mm;
  display: block;
}

figure.fig.cols-1 img {
  width: auto;
  max-width: 100%;
  max-height: 145mm;
  margin: 0 auto;
}

figure.fig.tall img {
  max-height: 202mm;
}

figure.fig.raw img {
  max-height: 92mm;
}

figure.fig.cols-2 img {
  width: auto;
  max-width: 100%;
  max-height: 88mm;
  margin: 0 auto;
}

figure.fig.cols-3 img {
  width: auto;
  max-width: 100%;
  max-height: 60mm;
  margin: 0 auto;
}

figure.fig.wide img {
  width: 100%;
  max-width: 100%;
  max-height: none;
}

figure.fig figcaption {
  font-family: "DejaVu Sans", sans-serif;
  font-size: 7.9pt;
  line-height: 1.4;
  color: #64748b;
  text-align: center;
  margin-top: 1mm;
  padding: 0 4mm;
}
"""

document = f"""<!doctype html>
<html lang="ru">
<head>
<meta charset="utf-8">
<title>{html.escape(title)} — Вектор на Марс, задание 2</title>
<meta name="author" content="Команда Vector2Mars">
<meta name="description" content="Технический отчёт: распознавание маршрутов ровера, детектор стрелок, бортовая программа и сшивка кадров.">
<meta name="keywords" content="Вектор на Марс, OpenCV, ArUco, Pioneer SDK 2, БПЛА, маршруты">
<style>{CSS}</style>
</head>
<body>

<section class="cover">
  <div class="cover-kicker">Вектор на Марс · Задание 2</div>
  <h1>{html.escape(title)}</h1>
  <p class="cover-sub">Технический отчёт: детектор стрелок, бортовая программа и сшивка кадров</p>
  <div class="cover-rule"></div>
  <div class="cover-answer">
    <div class="label">Состав решения</div>
    <div class="big">КОД</div>
    <div class="meta">детектор · полёт · сшивка</div>
    <div class="note">кадры — примеры работы, не финальный ответ</div>
  </div>
  <div class="cover-note">Маршруты и длины зависят от сцены: приложенные кадры показывают работу конвейера, а не эталонный ответ.</div>
  <div class="cover-bottom">
    <span>20 ноября 2026</span>
    <span>Python · OpenCV · Pioneer SDK 2</span>
  </div>
</section>

<section class="summary">
  <h2>Кратко</h2>
  <div class="lead">{intro_html}</div>
  <p class="scope-note">Числа на конкретных кадрах — пример. Решение пересчитывает маршруты для новой сцены; разбор приложенного полевого кадра — в разделе «Выбор маршрута на приложенном кадре».</p>
  <div class="stat-row">
    <div class="stat"><b>3</b><span>слоя: зрение, полёт, сшивка кадров</span></div>
    <div class="stat"><b>2</b><span>рабочих кадра в приложенном примере</span></div>
    <div class="stat"><b>111</b><span>тестов проходят офлайн, включая полёт и сшивку</span></div>
  </div>
  <div class="summary-body">{front_html}</div>
</section>

<section class="toc-page">
  <h2 class="toc-title">Содержание</h2>
  {toc_html}
</section>

<main id="report-body">
{body_html_final}
</main>

</body>
</html>
"""

OUT_PATH.write_bytes(
    HTML(string=document, base_url=str(ROOT)).write_pdf()
)
print(f"written {OUT_PATH} ({OUT_PATH.stat().st_size / 1024 / 1024:.1f} MiB)")
