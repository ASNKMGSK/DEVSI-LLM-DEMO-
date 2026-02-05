"""
청킹 테스트: 불릿 항목이 개별 청크로 분리되는지 확인
"""
import re
import sys
sys.path.insert(0, r"c:\Users\AKS\Desktop\데브시스터즈 프로젝트\backend 리팩토링 시작")

# service.py에서 청킹 관련 함수들 import
from rag.service import (
    _extract_bullet_blocks,
    _is_bullet_line,
    BULLET_REGEX,
    SECTION_TITLE_PATTERN,
)

OUTPUT_FILE = r"C:\Users\AKS\Desktop\데브시스터즈 프로젝트\backend 리팩토링 시작\chunking_result.txt"

# 파싱된 텍스트 (이상적인 형태)
PARSED_TEXT = """15. 왕국 레벨
왕국 경험치는 계정의 경험치를 뜻하며, 다양한 방법을 통해 상승시킨다.
- 왕국 활동: 물품 생산, 건물 건설 및 업그레이드, 영토 확장을 완료하면 반드시 주어진다. 후반부 영토 확장은 경험치를 매우 많이 얻을 수 있다.
- 퀘스트 완료: 초반 경험치 상승의 핵심.
- 곰젤리 열차 보상: 곰젤리 열차 납품 완료 후 칸별 보상으로 가끔 주어진다.
- 소원나무: 소원나무에 물품을 납품하면 자원의 가치에 따라 비례한 경험치를 얻는다.
- 곰젤리 열기구: 원정을 끝내고 돌아오면 반드시 주어진다.
왕국 레벨이 상승하면 쿠키 최대레벨, 스태미너 젤리 최대 충전개수가 증가하고 스태미너 젤리를 받을 수 있다.
왕국레벨 21부터 쿠키 최대레벨, 스태미너 젤리 최대 개수가 2씩 증가하여, 4월 8일 패치 기준으로 왕국 레벨 40에 쿠키의 레벨을 60까지 올릴 수 있게 된다. 41 이상부터는 스태미너 젤리 최대 충전개수만 증가한다.
2023-01-27 현재는 왕국 레벨 41에 쿠키 레벨 65, 왕국 레벨 42에 쿠키 레벨 70, 왕국 레벨 45에 쿠키 레벨 80으로 상한선이 해방된다.
2025-09-28 쿠키런 킹덤 전 서버 최초로 유튜버 ND러너가 100렙을 달성했다"""


def main():
    output_lines = []

    def log(line=""):
        output_lines.append(line)

    log("=" * 70)
    log("청킹 테스트: 불릿 항목이 개별 청크로 분리되는지 확인")
    log("=" * 70)

    # 1. 불릿 라인 감지 테스트
    log("\n" + "=" * 70)
    log("1) 불릿 라인 감지 테스트")
    log("=" * 70)

    test_lines = [
        "15. 왕국 레벨",  # 섹션 제목 - 불릿 아님
        "- 왕국 활동: 물품 생산...",  # 불릿
        "- 퀘스트 완료: 초반...",  # 불릿
        "왕국 경험치는 계정의 경험치를...",  # 일반 텍스트 - 불릿 아님
        "2023-01-27 현재는...",  # 날짜 - 불릿 아님
    ]

    for line in test_lines:
        is_bullet = _is_bullet_line(line)
        is_section = bool(SECTION_TITLE_PATTERN.match(line.strip()))
        status = "[BULLET]" if is_bullet else "[NOT BULLET]"
        log(f"  {status} {line[:50]}...")
        if is_section:
            log(f"           -> 섹션 제목으로 감지됨")

    # 2. 블록 추출 테스트
    log("\n" + "=" * 70)
    log("2) 블록 추출 테스트 (_extract_bullet_blocks)")
    log("=" * 70)

    blocks = _extract_bullet_blocks(PARSED_TEXT)

    for i, block in enumerate(blocks, 1):
        block_type = block.get("type", "unknown")
        log(f"\n[Block {i}] Type: {block_type}")

        if block_type == "bullet":
            header = block.get("header", "")
            items = block.get("items", [])
            log(f"  Header: {header}")
            log(f"  Items: {len(items)}개")
            for j, item in enumerate(items, 1):
                if isinstance(item, dict):
                    title = item.get("title", "")[:40]
                    desc = item.get("description", "")[:40]
                    log(f"    [{j}] title: {title}")
                    log(f"        desc: {desc}...")
                else:
                    log(f"    [{j}] {item[:50]}...")

        elif block_type == "prose":
            content = block.get("content", "")[:100]
            log(f"  Content: {content}...")

    # 3. 체크리스트
    log("\n" + "=" * 70)
    log("3) 체크리스트")
    log("=" * 70)

    bullet_blocks = [b for b in blocks if b.get("type") == "bullet"]
    total_items = sum(len(b.get("items", [])) for b in bullet_blocks)

    checks = [
        ("불릿 블록 1개 이상 존재", len(bullet_blocks) >= 1),
        ("불릿 항목 5개 존재", total_items == 5),
        ("'왕국 활동' 항목 존재", any(
            "왕국 활동" in str(item.get("title", "") if isinstance(item, dict) else item)
            for b in bullet_blocks for item in b.get("items", [])
        )),
        ("'퀘스트 완료' 항목 존재", any(
            "퀘스트 완료" in str(item.get("title", "") if isinstance(item, dict) else item)
            for b in bullet_blocks for item in b.get("items", [])
        )),
        ("prose 블록 존재 (일반 텍스트)", any(b.get("type") == "prose" for b in blocks)),
    ]

    for desc, passed in checks:
        status = "[OK]" if passed else "[FAIL]"
        log(f"  {status} {desc}")

    # 4. 예상 Child 청크 형태
    log("\n" + "=" * 70)
    log("4) 예상 Child 청크 형태 (1항목 = 1청크)")
    log("=" * 70)

    expected_chunks = [
        {"type": "list_item", "key": "왕국 활동", "text": "왕국 활동: 물품 생산..."},
        {"type": "list_item", "key": "퀘스트 완료", "text": "퀘스트 완료: 초반..."},
        {"type": "list_item", "key": "곰젤리 열차 보상", "text": "곰젤리 열차 보상: ..."},
        {"type": "list_item", "key": "소원나무", "text": "소원나무: ..."},
        {"type": "list_item", "key": "곰젤리 열기구", "text": "곰젤리 열기구: ..."},
    ]

    for chunk in expected_chunks:
        log(f"  [CHILD] section_title: 15. 왕국 레벨")
        log(f"          type: {chunk['type']}")
        log(f"          key: {chunk['key']}")
        log(f"          text: {chunk['text']}")
        log("")

    # 파일로 저장
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write('\n'.join(output_lines))

    print(f"Result saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
