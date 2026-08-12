"""app/tools/citation_manager.py 单元测试：纯代码逻辑，不打真实API。

网络探测部分用mock（真实网络可达性验证放在
tests/integration/test_adversarial_m3.py 的对抗测试里，那边故意用一个
真实不存在的域名做端到端验证）。
"""
from unittest.mock import AsyncMock, patch

import pytest

from app.graph.state import Finding, SourceRef
from app.tools.citation_manager import (
    CitationManager,
    sanitize_invalid_citations,
    validate_report_citations,
)


def _ref(url: str) -> SourceRef:
    return SourceRef(url=url, title=url, fetched_at="now")


def test_register_sources_deduplicates_and_assigns_stable_ids():
    cm = CitationManager()
    cm.register_sources({"https://a.com": _ref("https://a.com")})
    cm.register_sources({"https://a.com": _ref("https://a.com"), "https://b.com": _ref("https://b.com")})

    citation_map = cm.citation_map()
    assert len(citation_map) == 2
    assert set(citation_map.keys()) == {"S1", "S2"}


@pytest.mark.asyncio
async def test_verify_reachability_marks_sources():
    cm = CitationManager()
    cm.register_sources({"https://good.com": _ref("https://good.com"), "https://bad.com": _ref("https://bad.com")})

    async def fake_probe(url, timeout_seconds):
        return url == "https://good.com"

    with patch("app.tools.citation_manager._probe_url", new=AsyncMock(side_effect=fake_probe)):
        await cm.verify_reachability()

    citation_map = cm.citation_map()
    reachable_by_url = {ref.url: ref.reachable for ref in citation_map.values()}
    assert reachable_by_url["https://good.com"] is True
    assert reachable_by_url["https://bad.com"] is False


@pytest.mark.asyncio
async def test_remap_findings_replaces_urls_with_citation_ids():
    cm = CitationManager()
    cm.register_sources({"https://a.com": _ref("https://a.com")})
    finding = Finding(
        subtask_id="sub-1", claim="c", source_ids=["https://a.com"], confidence=0.8, status="unverified"
    )

    remapped = cm.remap_findings([finding])

    assert remapped[0].source_ids == ["S1"]


@pytest.mark.asyncio
async def test_remap_findings_downgrades_verified_when_all_sources_unreachable():
    """CLAUDE.md硬性约束落地：来源全部探测为不可达时，即使SubAgent自己
    标了verified，Citation Manager也要代码层强制降级，不信任模型自报。"""
    cm = CitationManager()
    cm.register_sources({"https://fake.com": _ref("https://fake.com")})

    async def fake_probe(url, timeout_seconds):
        return False

    with patch("app.tools.citation_manager._probe_url", new=AsyncMock(side_effect=fake_probe)):
        await cm.verify_reachability()

    finding = Finding(
        subtask_id="sub-1", claim="c", source_ids=["https://fake.com"], confidence=0.9, status="verified"
    )
    remapped = cm.remap_findings([finding])

    assert remapped[0].status == "unverified"


def test_remap_findings_drops_unregistered_urls():
    cm = CitationManager()
    finding = Finding(
        subtask_id="sub-1", claim="c", source_ids=["https://never-registered.com"], confidence=0.5, status="unverified"
    )
    remapped = cm.remap_findings([finding])
    assert remapped[0].source_ids == []


def test_validate_report_citations_finds_hallucinated_ids():
    citation_map = {"S1": _ref("https://a.com")}
    report = "第一个论点[S1]。第二个论点[S2]。"

    invalid = validate_report_citations(report, citation_map)

    assert invalid == ["S2"]


def test_validate_report_citations_passes_when_all_ids_valid():
    citation_map = {"S1": _ref("https://a.com"), "S2": _ref("https://b.com")}
    report = "论点A[S1]，论点B[S2]。"

    assert validate_report_citations(report, citation_map) == []


def test_sanitize_invalid_citations_replaces_only_invalid_markers():
    citation_map = {"S1": _ref("https://a.com")}
    report = "合法引用[S1]，幻觉引用[S99]。"

    sanitized = sanitize_invalid_citations(report, citation_map)

    assert "[S1]" in sanitized
    assert "[S99]" not in sanitized
    assert "[引用待核实]" in sanitized
