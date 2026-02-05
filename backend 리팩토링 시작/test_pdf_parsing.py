"""
PDF 파싱 테스트 스크립트 v3
"15. 왕국 레벨" 섹션이 제대로 파싱되는지 확인
"""
import re
import pdfplumber
import warnings

# PDF 경고 숨기기
warnings.filterwarnings('ignore')
import logging
logging.getLogger("pdfminer").setLevel(logging.ERROR)

# PDF 경로
PDF_PATH = r"C:\Users\AKS\Desktop\쿠키런 pdf\쿠키런_ 킹덤_게임 요소 - 나무위키.pdf"
OUTPUT_FILE = r"C:\Users\AKS\Desktop\데브시스터즈 프로젝트\backend 리팩토링 시작\parsing_result.txt"


def clean_namuwiki_pdf_noise(txt: str) -> str:
    """나무위키 PDF 노이즈 제거 + 구조화"""
    if not txt:
        return ""

    # 1. 각주 번호 제거
    txt = re.sub(r'\[\d+\]', '', txt)

    # 2. 광고 제거
    ad_patterns = [
        r'포토샵\s*포함.*?다운로드\s*받으세요\s*\.?',
        r'구매하기',
        r'어도비\s*\d+\s*캘린더.*?\.?',
        r'Creative\s*Cloud.*?완성해보세요\.?',
        r'타이포그래피도.*?더보기',
        r'굿즈\s*디자인까지',
    ]
    for pattern in ad_patterns:
        txt = re.sub(pattern, '', txt, flags=re.IGNORECASE | re.DOTALL)

    # 3. UI 문구 제거
    ui_patterns = [
        r'\[\s*펼치기\s*[·•]\s*접기\s*\]',
        r'\[\s*확률표\s*펼치기\s*[·•]\s*접기\s*\]',
        r'최근\s*수정\s*시각:\s*[\d\-:\s]+',
        r'분류:[가-힣A-Za-z0-9\s/:]+',
        r'상위\s*문서:\s*[가-힣A-Za-z0-9:\s]+',
    ]
    for pattern in ui_patterns:
        txt = re.sub(pattern, '', txt, flags=re.IGNORECASE | re.MULTILINE)

    # 4. 페이지 마커 제거
    txt = re.sub(r'<IMAGE\s+FOR\s+PAGE[^>]*>', '', txt)
    txt = re.sub(r'<PARSED\s+TEXT[^>]*>', '', txt)

    # 5. ★ 섹션 제목 정규화: "15.왕국" → "15. 왕국"
    txt = re.sub(r'(\d+)\.([가-힣])', r'\1. \2', txt)

    # 6. ★ 불릿 항목 표준화 (줄 시작에서 "항목명 - 설명" 패턴)
    def convert_bullet_line(match):
        item_name = match.group(1).strip()
        description = match.group(2).strip()
        return f"- {item_name}: {description}"

    txt = re.sub(
        r'^([가-힣]+(?:\s[가-힣]+)?(?:\s보상)?)\s*-\s*(.+)$',
        convert_bullet_line,
        txt,
        flags=re.MULTILINE
    )

    # 7. 연속 공백/줄바꿈 정리
    txt = re.sub(r'[ \t]+', ' ', txt)
    txt = re.sub(r'\n{3,}', '\n\n', txt)
    txt = re.sub(r'^\s+', '', txt, flags=re.MULTILINE)

    return txt.strip()


def extract_kingdom_level_section_from_pages(pdf) -> str:
    """페이지별로 순회하면서 본문의 '15.왕국 레벨' 섹션 찾기"""
    for i, page in enumerate(pdf.pages):
        page_text = page.extract_text() or ""

        # 본문 패턴: "15.왕국 레벨" 바로 뒤에 "왕국 경험치"가 오는 경우
        # 목차는 "15. 왕국 레벨\n16."처럼 바로 다음 번호가 옴
        match = re.search(r'15\.왕국\s*레벨\n', page_text)
        if match and "왕국 경험치" in page_text[match.end():match.end()+100]:
            # 본문 발견!
            start = match.start()
            # 16.뽑기 까지
            next_section = re.search(r'16\.뽑기', page_text[start:])
            if next_section:
                end = start + next_section.start()
            else:
                end = len(page_text)

            return page_text[start:end]

    return "섹션을 찾을 수 없습니다"


def main():
    output_lines = []

    def log(line=""):
        output_lines.append(line)

    log("=" * 70)
    log("PDF Parsing Test v3: 15. 왕국 레벨 섹션 (본문 추출)")
    log("=" * 70)

    # PDF 열기
    with pdfplumber.open(PDF_PATH) as pdf:
        log(f"\nTotal pages: {len(pdf.pages)}")

        # 본문에서 "15. 왕국 레벨" 섹션 추출
        section_raw = extract_kingdom_level_section_from_pages(pdf)

    log("\n" + "=" * 70)
    log("1) RAW 추출 결과 (노이즈 제거 전)")
    log("=" * 70)
    log(section_raw)

    # 노이즈 제거 + 구조화 적용
    section_cleaned = clean_namuwiki_pdf_noise(section_raw)

    log("\n" + "=" * 70)
    log("2) 노이즈 제거 + 구조화 후")
    log("=" * 70)
    log(section_cleaned)

    # 체크리스트
    log("\n" + "=" * 70)
    log("3) 체크리스트")
    log("=" * 70)

    checks = [
        ("15. 왕국 레벨 헤더 존재 (공백 포함)", r'15\.\s+왕국\s*레벨', 0),
        ("불릿 형태: '- 왕국 활동:' 존재", r'-\s*왕국\s*활동:', 0),
        ("불릿 형태: '- 퀘스트 완료:' 존재", r'-\s*퀘스트\s*완료:', 0),
        ("불릿 형태: '- 곰젤리 열차 보상:' 존재", r'-\s*곰젤리\s*열차\s*보상:', 0),
        ("불릿 형태: '- 소원나무:' 존재", r'-\s*소원나무:', 0),
        ("불릿 형태: '- 곰젤리 열기구:' 존재", r'-\s*곰젤리\s*열기구:', 0),
        ("날짜 '2023-01-27' 존재", r'2023-01-27', 0),
        ("날짜 '2025-09-28' 존재", r'2025-09-28', 0),
    ]

    for desc, pattern, flags in checks:
        found = bool(re.search(pattern, section_cleaned, flags))
        status = "[OK]" if found else "[FAIL]"
        log(f"  {status} {desc}")

    # 불릿 개수 세기
    bullet_count = len(re.findall(r'^-\s+[가-힣]', section_cleaned, re.MULTILINE))
    status = "[OK]" if bullet_count >= 5 else "[FAIL]"
    log(f"  {status} 불릿 항목 5개 이상 (현재: {bullet_count}개)")

    # 정답 형태와 비교
    log("\n" + "=" * 70)
    log("4) 실제 출력 결과")
    log("=" * 70)
    log(section_cleaned)

    # 파일로 저장
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write('\n'.join(output_lines))

    print(f"Result saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
