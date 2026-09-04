#!/usr/bin/env python3
"""Create Chinese-explained copies of every C1 PDF figure.

The source PDFs are immutable experiment evidence.  This helper mirrors them
under ``figures/annotated-zh/`` and extends every page with a footer containing
the figure's purpose, reading guidance, scientific conclusion, and evidence
status.  Historical figures remain clearly labelled as historical.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import NamedTuple

import pymupdf


SCRIPT_DIR = Path(__file__).resolve().parent
C1_DIR = SCRIPT_DIR.parent
FIGURES_DIR = C1_DIR / "figures"
OUTPUT_DIR = FIGURES_DIR / "annotated-zh"
MANIFEST_PATH = OUTPUT_DIR / "manifest.json"
README_PATH = OUTPUT_DIR / "README.md"
FOOTER_HEIGHT = 250.0
OUTPUT_VERSION = 1


class Caption(NamedTuple):
    family: str
    purpose: str
    reading: str
    conclusion: str


CAPTIONS: dict[str, Caption] = {
    "c1_combined_motivation.pdf": Caption(
        "combined_motivation",
        "把 E1 物理扩展、E2 路由扇出、E3 单分片图搜索工作量和 E4 查询总工作量放在同一页，展示散播—聚合瓶颈的完整因果链。",
        "按行比较 SIFT1M 与 GloVe-200-angular，按列从左到右观察归一化 QPS、实际扇出、单分片工作量以及扇出乘以局部工作量；同时比较 Random、K-Means 和理想参考线。",
        "物理吞吐扩展显著次线性；K-Means 可以在部分数据上保持很小扇出；局部工作量缓慢下降并非所有配置都成立；总工作量恒等式成立但不能通过跨数据集物理归因门槛，因此完整 C1 结论为 INSUFFICIENT。",
    ),
    "c1_fig1_physical_scaleout.pdf": Caption(
        "physical_scaleout",
        "检验增加真实物理工作节点时，端到端查询吞吐量是否接近线性增长。",
        "横轴是实际物理节点数 M，纵轴是相对 M=1 的归一化实测 QPS；灰色理想线为 y=M。实测曲线离理想线越远，扩展效率越低。误差条表示独立重复测量的不确定性。",
        "M=4 时四条实测曲线仅达到约 0.394–1.137 倍基线，远低于理想的 4 倍，因此 C1-a“物理扩展显著次线性”得到支持。这里只包含 M=1、2、4 的真实物理测量。",
    ),
    "fig_c1_physical_scaleout.pdf": Caption(
        "physical_scaleout",
        "检验增加真实物理工作节点时，端到端查询吞吐量是否接近线性增长。",
        "横轴是实际物理节点数 M，纵轴是相对 M=1 的归一化实测 QPS；灰色理想线为 y=M。实测曲线离理想线越远，扩展效率越低。误差条表示独立重复测量的不确定性。",
        "M=4 时四条实测曲线仅达到约 0.394–1.137 倍基线，远低于理想的 4 倍，因此 C1-a“物理扩展显著次线性”得到支持。这里只包含 M=1、2、4 的真实物理测量。",
    ),
    "fig_c1_scaling_efficiency.pdf": Caption(
        "scaling_efficiency",
        "把吞吐量增益除以物理节点数，直接显示每增加一个节点所保留下来的扩展效率。",
        "横轴是物理节点数，纵轴 E(M)=归一化 QPS/M；理想效率恒为 1。曲线越接近 0，说明新增节点带来的收益越少，甚至可能因为散播、聚合或客户端开销而退化。",
        "M=4 时效率约为 0.098–0.284，明显低于理想值 1；两个数据集和两种分区方法均表现出严重效率损失，与 C1-a 的次线性扩展结论一致。",
    ),
    "c1_fig2_fanout_vs_logical_shards.pdf": Caption(
        "fanout_vs_shards",
        "比较逻辑分片数增长时，实际服务路由访问的分片数与逐查询 oracle 最小分片数。",
        "横轴 M 是逻辑分片数而非物理机器数；纵轴是平均扇出。actual 表示实验实际选择的固定服务扇出，oracle 表示达到目标召回率所需的逐查询最小扇出；两者间距表示可避免的过度散播。",
        "Random 在两数据集上实际扇出等于 M；但 SIFT1M K-Means 在 M=1、2、4、8、16、32 时仅为 1、1、1、2、3、3，因此“高扇出可推广到空间分区”的 C1-b 被反驳。GloVe K-Means 的 1、2、3、4、9、8 还显示非单调性。",
    ),
    "fig_c1_fanout_vs_shards.pdf": Caption(
        "fanout_vs_shards",
        "比较逻辑分片数增长时，实际服务路由访问的分片数与逐查询 oracle 最小分片数。",
        "横轴 M 是逻辑分片数而非物理机器数；纵轴是平均扇出。actual 表示实验实际选择的固定服务扇出，oracle 表示达到目标召回率所需的逐查询最小扇出；两者间距表示可避免的过度散播。",
        "Random 在两数据集上实际扇出等于 M；但 SIFT1M K-Means 在 M=1、2、4、8、16、32 时仅为 1、1、1、2、3、3，因此“高扇出可推广到空间分区”的 C1-b 被反驳。GloVe K-Means 的 1、2、3、4、9、8 还显示非单调性。",
    ),
    "c1_fig3_required_fanout_cdf.pdf": Caption(
        "oracle_fanout_cdf",
        "展示每条查询达到目标召回率时所需的 oracle 最小扇出分布，而不只看平均值。",
        "横轴是最少需要访问的分片数，纵轴经验 CDF 表示“所需扇出不超过该值”的查询比例。曲线越靠左、越快升到 1，说明多数查询只需少量分片；不同 M 的曲线可判断需求是否随分片数同步增长。",
        "oracle 分布通常远小于 M，K-Means 尤其明显；这说明很多查询理论上无需广播。结合实际扇出结果，广义高扇出假设不能覆盖所有分区和数据集，C1-b 最终为 CONTRADICTED。",
    ),
    "fig_c1_oracle_fanout_cdf.pdf": Caption(
        "oracle_fanout_cdf",
        "展示每条查询达到目标召回率时所需的 oracle 最小扇出分布，而不只看平均值。",
        "横轴是最少需要访问的分片数，纵轴经验 CDF 表示“所需扇出不超过该值”的查询比例。曲线越靠左、越快升到 1，说明多数查询只需少量分片；不同 M 的曲线可判断需求是否随分片数同步增长。",
        "oracle 分布通常远小于 M，K-Means 尤其明显；这说明很多查询理论上无需广播。结合实际扇出结果，广义高扇出假设不能覆盖所有分区和数据集，C1-b 最终为 CONTRADICTED。",
    ),
    "fig_c1_actual_fanout_cdf.pdf": Caption(
        "actual_fanout_cdf",
        "展示实际服务配置对每条查询访问多少分片，并与 oracle CDF 区分开来。",
        "横轴是实际访问分片数，纵轴是经验 CDF。由于每个配置采用固定 selected fan-out，曲线通常呈阶跃；跃升位置就是该配置的实际服务扇出。将它与 oracle CDF 对照，可识别路由保守程度。",
        "Random 的阶跃位于 M，等同广播；SIFT1M K-Means 在 M=4、16、32 仅访问 1、3、3 个分片，GloVe K-Means 为 3、9、8。实际策略差异很大，不能把高扇出概括为所有空间分区的普遍规律。",
    ),
    "c1_fig4_local_search_work.pdf": Caption(
        "local_search_work",
        "同时考察每个被搜索分片上的距离计算数、图节点访问数和 worker CPU 时间如何随逻辑分片数变化。",
        "每项均相对 M=1 归一化。1/M 线代表工作量与分片大小完全成比例下降，log(N/M) 是较慢下降的参考；实测线高于 1/M 表示局部工作下降较慢，等于或低于它则构成反例。",
        "距离计算和节点访问的 40 个非基线比较中，34 个高于理想 1/M，但仍有 6 个等于或低于它，包含 GloVe Random 的 M=2、4 等反例。因此“局部图搜索工作普遍缓慢下降”的广义 C1-c 为 CONTRADICTED。",
    ),
    "fig_c1_local_search_work.pdf": Caption(
        "local_search_work",
        "同时考察每个被搜索分片上的距离计算数、图节点访问数和 worker CPU 时间如何随逻辑分片数变化。",
        "每项均相对 M=1 归一化。1/M 线代表工作量与分片大小完全成比例下降，log(N/M) 是较慢下降的参考；实测线高于 1/M 表示局部工作下降较慢，等于或低于它则构成反例。",
        "距离计算和节点访问的 40 个非基线比较中，34 个高于理想 1/M，但仍有 6 个等于或低于它，包含 GloVe Random 的 M=2、4 等反例。因此“局部图搜索工作普遍缓慢下降”的广义 C1-c 为 CONTRADICTED。",
    ),
    "fig_c1_local_distance_computations.pdf": Caption(
        "local_distance_computations",
        "用距离计算次数衡量每个被搜索分片内部的主要图搜索工作量。",
        "横轴是逻辑分片数，纵轴是相对 M=1 的单分片距离计算量；与 1/M 和 log(N/M) 参考线比较。高于 1/M 表示下降不及数据规模缩小速度，等于或低于 1/M 表示局部成本下降较快。",
        "多数配置的距离计算下降慢于 1/M，但存在足以否定全称命题的反例；结合节点访问证据，40 个比较中有 6 个不高于 1/M，因此广义 C1-c 最终为 CONTRADICTED，而不是“所有配置均缓慢下降”。",
    ),
    "fig_c1_local_nodes_visited.pdf": Caption(
        "local_nodes_visited",
        "用 HNSW 图节点访问数交叉验证单分片搜索工作量，而不只依赖距离计算计数器。",
        "横轴是逻辑分片数，纵轴是相对 M=1 的每个被搜索分片节点访问数；实测线与 1/M 参考线的上下关系表示局部搜索下降慢于或不慢于理想比例。",
        "节点访问趋势与距离计算的总体判断一致：多数点下降较慢，但并非所有数据集、分区方法和 M 都成立。存在反例使广义 C1-c 被反驳；该图不能被解读为普遍支持。",
    ),
    "fig_c1_local_cpu_time.pdf": Caption(
        "local_cpu_time",
        "从实际 worker CPU 时间角度检查局部图搜索成本，补充距离计算和节点访问两个硬件计数器。",
        "横轴是逻辑分片数，纵轴是相对 M=1 的每个被搜索分片 CPU 时间；与 1/M 比较时，要注意 CPU 时间还包含固定开销、调度和计时噪声，因此它是辅助指标而非单独的机制判据。",
        "CPU 时间没有呈现跨数据集、跨分区的一致 1/M 规律，说明固定开销和实现因素不可忽略；最终 C1-c 仍以距离计算与节点访问的确定性证据判为 CONTRADICTED。",
    ),
    "c1_fig5_aggregate_work_decomposition.pdf": Caption(
        "aggregate_work",
        "验证单次查询的总图搜索工作量能否由“实际扇出 × 每个被搜索分片的平均局部工作量”解释。",
        "横轴是逻辑分片数，纵轴是归一化总距离计算量。observed 是逐查询事件求和，model 是扇出乘以局部工作量；两条线重合说明分解恒等式成立。1/M 线仅是理想总成本下降参考。",
        "所有配置中 model 与 observed 精确一致，支持 C1-d 的聚合成本恒等式；但总工作量常随 M 增长，且工作量投影无法解释跨数据集的实测 QPS，所以物理吞吐归因仍为 INSUFFICIENT。",
    ),
    "fig_c1_aggregate_work.pdf": Caption(
        "aggregate_work",
        "验证单次查询的总图搜索工作量能否由“实际扇出 × 每个被搜索分片的平均局部工作量”解释。",
        "横轴是逻辑分片数，纵轴是归一化总距离计算量。observed 是逐查询事件求和，model 是扇出乘以局部工作量；两条线重合说明分解恒等式成立。1/M 线仅是理想总成本下降参考。",
        "所有配置中 model 与 observed 精确一致，支持 C1-d 的聚合成本恒等式；但总工作量常随 M 增长，且工作量投影无法解释跨数据集的实测 QPS，所以物理吞吐归因仍为 INSUFFICIENT。",
    ),
    "fig_c1_model_vs_observed.pdf": Caption(
        "model_vs_observed",
        "用散点图直接检验聚合工作量模型与逐查询实测总工作量是否一致。",
        "横轴是 observed，纵轴是 fan-out × local-work model；点落在对角线 y=x 上表示精确吻合。颜色区分分区方法，标注区分 M；应同时观察两个数据集面板。",
        "四个数据集/分区组合的 Pearson 和 Spearman 相关系数均为 1，平均及最大相对误差均为 0。它证明成本分解恒等式，但不等价于证明 QPS 因果关系；物理归因仍为 INSUFFICIENT。",
    ),
    "c1_fig6_projected_logical_scaling.pdf": Caption(
        "projected_scaling",
        "把 M=1、2、4 的真实物理 E1 吞吐量与基于聚合图搜索工作量的候选投影放在一起，并执行发布门槛。",
        "实测点只对应真实物理节点；M=8、16、32 是逻辑分片机制数据。图中“withheld”表示跨数据集趋势校验失败，缺失的高 M 曲线不是零值，也不是已经测得的物理吞吐量。",
        "SIFT Random 的工作量投影与 M=1、2、4 实测 E1 趋势反相关，跨数据集门槛失败；因此所有 M>4 吞吐量投影均被扣留，C1-d 物理归因为 INSUFFICIENT。",
    ),
    "fig_c1_projected_scaling.pdf": Caption(
        "projected_scaling",
        "把 M=1、2、4 的真实物理 E1 吞吐量与基于聚合图搜索工作量的候选投影放在一起，并执行发布门槛。",
        "实测点只对应真实物理节点；M=8、16、32 是逻辑分片机制数据。图中“withheld”表示跨数据集趋势校验失败，缺失的高 M 曲线不是零值，也不是已经测得的物理吞吐量。",
        "SIFT Random 的工作量投影与 M=1、2、4 实测 E1 趋势反相关，跨数据集门槛失败；因此所有 M>4 吞吐量投影均被扣留，C1-d 物理归因为 INSUFFICIENT。",
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_pdfs() -> list[Path]:
    return sorted(
        path
        for path in FIGURES_DIR.rglob("*.pdf")
        if OUTPUT_DIR.name not in path.relative_to(FIGURES_DIR).parts
    )


def caption_for(path: Path) -> Caption:
    matches = [
        (suffix, caption)
        for suffix, caption in CAPTIONS.items()
        if path.name.endswith(suffix)
    ]
    if not matches:
        raise ValueError(f"No Chinese caption mapping for {path}")
    return max(matches, key=lambda item: len(item[0]))[1]


def evidence_status(path: Path) -> str:
    relative = path.relative_to(FIGURES_DIR)
    label = relative.as_posix()
    if "history" in relative.parts:
        return (
            "历史目录中的阶段快照，仅用于复现、差异审计和结论演变追溯；"
            "最终引用请使用 figures 顶层无 stage 前缀的 Stage 12 权威图。"
        )
    if label.startswith("stage3-"):
        return (
            "Stage 3 单数据集 E2 阶段快照；当前确定性 Stage 12 E2 已取代其最终解释，"
            "请以顶层无 stage 前缀图为准。"
        )
    if label.startswith("stage4-original-"):
        return (
            "Stage 4 原始 E1 历史证据，受串行客户端限制；其物理解读已被纠正后的 Stage 6 E1 取代，"
            "不可作为最终扩展曲线。"
        )
    if label.startswith("stage5-original-"):
        return (
            "Stage 5 基于原始 Stage 4 曲线的历史 E4 证据；纠正后的 Stage 7 及确定性 Stage 12 结果"
            "已取代其物理解释。"
        )
    if label.startswith("stage6-corrected-sift1m-"):
        return (
            "纠正后的 SIFT1M E1 数据集快照，是最终 SIFT 物理曲线的数据来源；跨数据集引用请使用"
            "顶层无 stage 前缀图。"
        )
    if label.startswith("stage7-corrected-sift1m-"):
        return (
            "纠正后的 SIFT1M 阶段性 E4 快照；确定性 Stage 12 E2–E4 已取代其最终机制解释，"
            "保留本图仅用于追溯。"
        )
    if label.startswith("stage8-corrected-glove-200-angular-"):
        return (
            "纠正后的 GloVe E1 数据集快照，是最终 GloVe 物理曲线的数据来源；跨数据集引用请使用"
            "顶层无 stage 前缀图。"
        )
    if label.startswith("stage9-glove-200-angular-"):
        return (
            "Stage 9 GloVe 阶段性 E4 快照；确定性 Stage 12 E2–E4 已取代其最终机制解释，"
            "保留本图仅用于追溯。"
        )
    return (
        "figures 顶层无 stage 前缀的 Stage 12 当前权威图；最终口径与 deterministic final record、"
        "RESULTS.md 和 22/22 完整性审计一致。"
    )


def insert_footer_text(
    page: pymupdf.Page,
    original_height: float,
    page_width: float,
    caption: Caption,
    status: str,
    font_buffer: bytes,
) -> None:
    page.draw_line(
        pymupdf.Point(24, original_height + 14),
        pymupdf.Point(page_width - 24, original_height + 14),
        color=(0.72, 0.76, 0.82),
        width=0.8,
    )
    page.insert_font(fontname="c1-cjk", fontbuffer=font_buffer)
    title_rect = pymupdf.Rect(
        28,
        original_height + 24,
        page_width - 28,
        original_height + 48,
    )
    title_result = page.insert_textbox(
        title_rect,
        "中文图解（当前科学口径）",
        fontname="c1-cjk",
        fontsize=13.0,
        lineheight=1.0,
        color=(0.08, 0.24, 0.47),
    )
    if title_result < 0:
        raise RuntimeError("Chinese footer title does not fit")

    body = (
        f"用途：{caption.purpose}\n\n"
        f"如何理解：{caption.reading}\n\n"
        f"最终结论：{caption.conclusion}\n\n"
        f"证据状态：{status}"
    )
    body_rect = pymupdf.Rect(
        28,
        original_height + 54,
        page_width - 28,
        original_height + FOOTER_HEIGHT - 16,
    )
    for fontsize in (11.2, 10.8, 10.4, 10.0, 9.6):
        result = page.insert_textbox(
            body_rect,
            body,
            fontname="c1-cjk",
            fontsize=fontsize,
            lineheight=1.22,
            color=(0.08, 0.08, 0.09),
        )
        if result >= 0:
            return
    raise RuntimeError("Chinese footer body does not fit")


def annotate_pdf(source: Path, destination: Path, caption: Caption, status: str) -> int:
    source_doc = pymupdf.open(source)
    if source_doc.page_count < 1:
        source_doc.close()
        raise ValueError(f"PDF has no pages: {source}")

    output_doc = pymupdf.open()
    cjk_font = pymupdf.Font(fontname="china-s")
    font_buffer = cjk_font.buffer
    for page_number, source_page in enumerate(source_doc):
        source_rect = source_page.rect
        output_page = output_doc.new_page(
            width=source_rect.width,
            height=source_rect.height + FOOTER_HEIGHT,
        )
        output_page.show_pdf_page(
            pymupdf.Rect(0, 0, source_rect.width, source_rect.height),
            source_doc,
            page_number,
            keep_proportion=False,
        )
        insert_footer_text(
            output_page,
            source_rect.height,
            source_rect.width,
            caption,
            status,
            font_buffer,
        )

    page_count = source_doc.page_count
    source_doc.close()
    output_doc.subset_fonts()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    output_doc.save(temporary, garbage=4, deflate=True)
    output_doc.close()
    temporary.replace(destination)
    return page_count


def write_readme(entries: list[dict[str, object]]) -> None:
    family_counts = Counter(str(entry["family"]) for entry in entries)
    top_level_count = sum("/" not in str(entry["source"]) for entry in entries)
    historical_count = len(entries) - top_level_count
    rows = "\n".join(
        f"| `{family}` | {count} |" for family, count in sorted(family_counts.items())
    )
    text = f"""# C1 PDF 中文说明版

本目录包含 `{len(entries)}` 张 C1 PDF 的中文说明副本：顶层当前/阶段图 `{top_level_count}` 张，
`history/` 历史图 `{historical_count}` 张。原始 PDF 未被修改。

每一页图表下方均包含：

- 用途：该图回答什么实验问题；
- 如何理解：横纵轴、参考线、CDF 或模型线应如何阅读；
- 最终结论：采用 Stage 12 确定性最终记录的科学结论；
- 证据状态：区分当前权威图、数据集快照、已被取代图和历史审计副本。

顶层无 `stage` 前缀的图是当前引用版本。`stage*` 及 `history/` 文件仅用于复现和结论演变追溯。

| 图表类型 | PDF 数量 |
|---|---:|
{rows}

生成与检查：

```bash
/users/dry/orion-distributed/venv/bin/python experiments/c1/scripts/c1_annotate_figures_zh.py
/users/dry/orion-distributed/venv/bin/python experiments/c1/scripts/c1_annotate_figures_zh.py --check
```
"""
    README_PATH.write_text(text, encoding="utf-8")


def generate() -> int:
    sources = source_pdfs()
    if not sources:
        raise RuntimeError(f"No source PDFs found under {FIGURES_DIR}")

    entries: list[dict[str, object]] = []
    for index, source in enumerate(sources, start=1):
        relative = source.relative_to(FIGURES_DIR)
        destination = OUTPUT_DIR / relative
        caption = caption_for(source)
        status = evidence_status(source)
        pages = annotate_pdf(source, destination, caption, status)
        entries.append(
            {
                "source": relative.as_posix(),
                "annotated": destination.relative_to(OUTPUT_DIR).as_posix(),
                "family": caption.family,
                "pages": pages,
                "source_sha256": sha256(source),
                "annotated_sha256": sha256(destination),
            }
        )
        print(f"[{index:03d}/{len(sources):03d}] {relative}")

    manifest = {
        "version": OUTPUT_VERSION,
        "source_root": str(FIGURES_DIR),
        "output_root": str(OUTPUT_DIR),
        "pdf_count": len(entries),
        "footer_height_points": FOOTER_HEIGHT,
        "entries": entries,
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_readme(entries)
    print(f"Generated {len(entries)} annotated PDFs under {OUTPUT_DIR}")
    return 0


def check() -> int:
    if not MANIFEST_PATH.exists():
        raise RuntimeError(f"Missing manifest: {MANIFEST_PATH}")
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    entries = manifest.get("entries", [])
    sources = source_pdfs()
    if len(entries) != len(sources):
        raise RuntimeError(
            f"Manifest/source count mismatch: {len(entries)} != {len(sources)}"
        )

    source_names = {path.relative_to(FIGURES_DIR).as_posix() for path in sources}
    manifest_names = {str(entry["source"]) for entry in entries}
    if source_names != manifest_names:
        missing = sorted(source_names - manifest_names)
        stale = sorted(manifest_names - source_names)
        raise RuntimeError(f"Manifest mismatch: missing={missing}, stale={stale}")

    required_labels = ("用途：", "如何理解：", "最终结论：", "证据状态：")
    for index, entry in enumerate(entries, start=1):
        source = FIGURES_DIR / str(entry["source"])
        annotated = OUTPUT_DIR / str(entry["annotated"])
        if not annotated.exists():
            raise RuntimeError(f"Missing annotated PDF: {annotated}")
        if sha256(source) != entry["source_sha256"]:
            raise RuntimeError(f"Source changed after generation: {source}")
        if sha256(annotated) != entry["annotated_sha256"]:
            raise RuntimeError(f"Annotated PDF hash mismatch: {annotated}")

        source_doc = pymupdf.open(source)
        annotated_doc = pymupdf.open(annotated)
        if source_doc.page_count != annotated_doc.page_count:
            raise RuntimeError(f"Page-count mismatch: {annotated}")
        for page_number in range(source_doc.page_count):
            source_page = source_doc[page_number]
            annotated_page = annotated_doc[page_number]
            height_delta = annotated_page.rect.height - source_page.rect.height
            if abs(height_delta - FOOTER_HEIGHT) > 0.2:
                raise RuntimeError(
                    f"Footer-height mismatch in {annotated}, page {page_number + 1}"
                )
            text = annotated_page.get_text()
            missing_labels = [label for label in required_labels if label not in text]
            if missing_labels:
                raise RuntimeError(
                    f"Missing Chinese labels {missing_labels} in {annotated}, "
                    f"page {page_number + 1}"
                )
        source_doc.close()
        annotated_doc.close()
        print(f"[{index:03d}/{len(entries):03d}] PASS {entry['source']}")

    print(f"PASS: {len(entries)}/{len(entries)} annotated PDFs verified")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify existing annotated PDFs instead of regenerating them",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return check() if args.check else generate()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
