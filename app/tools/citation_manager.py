"""Citation Manager：纯代码逻辑，不调用LLM（CLAUDE.md第4条硬性约束）。

职责：
1. 来源去重 + 生成稳定的引用编号（S1/S2/...），原始SourceRef以URL为key，
   这里转成人类可读、Writer可以直接引用的编号。
2. URL可达性校验，用于识别SubAgent可能编造的URL——这是CLAUDE.md第2/3条
   约束（无来源不确定性断言 + 幻觉防护）在Citation Manager层面的落地。
3. 把findings的source_ids从原始URL替换成引用编号。
4. 校验Writer生成报告里的引用编号是否都在映射表里，查不到视为幻觉引用。
"""
from __future__ import annotations

import asyncio
import re

import httpx

from app.graph.state import Finding, SourceRef

CITATION_PATTERN = re.compile(r"\[(S\d+)\]")


async def _probe_url(url: str, timeout_seconds: float) -> bool:
    """探测URL是否可达。只探测一次，不重试——这里的"失败"本身就是一个
    有效结论（"当前连不上/可能是编造的"），重试并不会改变这个判断的用途，
    且要遵守CLAUDE.md"硬上限、不允许无限循环"的约束，没有必要为一次性
    的探测引入重试循环。
    """
    try:
        async with httpx.AsyncClient(timeout=timeout_seconds, follow_redirects=True) as client:
            resp = await client.head(url)
            if resp.status_code >= 400:
                resp = await client.get(url)
            return resp.status_code < 400
    except Exception:
        return False


class CitationManager:
    def __init__(self) -> None:
        self._citation_ids: dict[str, str] = {}  # url -> "S1"
        self._sources: dict[str, SourceRef] = {}  # url -> SourceRef

    def register_sources(self, sources: dict[str, SourceRef]) -> None:
        """去重合并来源，编号按首次注册顺序分配，同一个URL不会拿到两个编号。"""
        for url, ref in sources.items():
            if url not in self._sources:
                self._sources[url] = ref
                self._citation_ids[url] = f"S{len(self._citation_ids) + 1}"

    async def verify_reachability(self, timeout_seconds: float = 8.0, concurrency: int = 5) -> None:
        """对所有已注册URL做一次可达性探测，更新SourceRef.reachable。"""
        semaphore = asyncio.Semaphore(concurrency)

        async def check(url: str) -> None:
            async with semaphore:
                reachable = await _probe_url(url, timeout_seconds)
            self._sources[url] = self._sources[url].model_copy(update={"reachable": reachable})

        await asyncio.gather(*(check(url) for url in self._sources))

    def citation_map(self) -> dict[str, SourceRef]:
        """id -> SourceRef，供Writer生成参考文献列表/引用校验使用。"""
        return {self._citation_ids[url]: ref for url, ref in self._sources.items()}

    def remap_findings(self, findings: list[Finding]) -> list[Finding]:
        """把findings的source_ids从URL替换成引用编号。如果一条finding的
        来源经过可达性校验、且全部不可达，代码层强制把verified降级为
        unverified——不信任模型自报的status，这是"避免LLM编造网址"要求
        的具体落地，而不是仅仅停留在prompt层面的口头约束。
        """
        remapped: list[Finding] = []
        for f in findings:
            ids: list[str] = []
            any_reachable = False
            any_checked = False
            for url in f.source_ids:
                cid = self._citation_ids.get(url)
                if cid is None:
                    continue  # 未注册的URL（正常流程不应发生），丢弃这个引用而不是保留一个野指针
                ids.append(cid)
                ref = self._sources[url]
                if ref.reachable is not None:
                    any_checked = True
                    if ref.reachable:
                        any_reachable = True

            status = f.status
            if ids and any_checked and not any_reachable and status == "verified":
                status = "unverified"

            remapped.append(f.model_copy(update={"source_ids": ids, "status": status}))
        return remapped


def validate_report_citations(report_text: str, citation_map: dict[str, SourceRef]) -> list[str]:
    """扫描report里所有[S数字]引用标记，返回不在citation_map里的编号列表
    （即幻觉引用）。纯正则+集合运算，不用LLM。"""
    cited = set(CITATION_PATTERN.findall(report_text))
    valid = set(citation_map.keys())
    invalid = cited - valid
    return sorted(invalid, key=lambda s: int(s[1:]))


def sanitize_invalid_citations(report_text: str, citation_map: dict[str, SourceRef]) -> str:
    """代码层兜底：把报告里所有不在citation_map中的引用标记替换成
    [引用待核实]。用于Writer多次重写仍无法消除幻觉引用时的最终降级，
    保证系统绝不会把带幻觉引用的报告原样输出给用户。"""
    valid = set(citation_map.keys())

    def _replace(match: re.Match[str]) -> str:
        cid = match.group(1)
        return match.group(0) if cid in valid else "[引用待核实]"

    return CITATION_PATTERN.sub(_replace, report_text)
