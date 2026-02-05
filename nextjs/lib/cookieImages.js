// lib/cookieImages.js
// CookieRun Kingdom 캐릭터 이미지 URL 매핑
// Source: https://sugargnome.com

// 주요 쿠키 이미지 URL (sugargnome.com에서 가져옴)
export const COOKIE_IMAGES = {
  // 에이션트/레전더리 쿠키
  'Pure Vanilla Cookie': 'https://sugar-gnome-warehouse-img.org/images/1749382015289-s8fc2g.webp',
  'Dark Cacao Cookie': 'https://sugar-gnome-warehouse-img.org/images/1749378670603-daif5t.webp',
  'Hollyberry Cookie': 'https://sugar-gnome-warehouse-img.org/images/1749376562048-gc9m7m.webp',

  // 에픽 쿠키
  'Espresso Cookie': 'https://sugar-gnome-warehouse-img.org/images/1749377303300-moxk85.webp',
  'Madeleine Cookie': 'https://sugar-gnome-warehouse-img.org/images/1749377573443-7k1dc5.webp',

  // 기본 쿠키들
  'GingerBrave': 'https://sugar-gnome-warehouse-img.org/images/1749377308451-qgo2ui.webp',
  'Strawberry Cookie': 'https://sugar-gnome-warehouse-img.org/images/1749377482757-p5vek5.webp',
};

// 한국어 이름 -> 영어 이름 매핑
export const COOKIE_NAME_MAP = {
  '용감한 쿠키': 'GingerBrave',
  '딸기맛 쿠키': 'Strawberry Cookie',
  '퓨어바닐라 쿠키': 'Pure Vanilla Cookie',
  '다크카카오 쿠키': 'Dark Cacao Cookie',
  '홀리베리 쿠키': 'Hollyberry Cookie',
  '에스프레소맛 쿠키': 'Espresso Cookie',
  '마들렌맛 쿠키': 'Madeleine Cookie',
};

// 쿠키 이름으로 이미지 URL 가져오기
export function getCookieImage(cookieName) {
  // 직접 매칭
  if (COOKIE_IMAGES[cookieName]) {
    return COOKIE_IMAGES[cookieName];
  }

  // 한국어 이름으로 시도
  const englishName = COOKIE_NAME_MAP[cookieName];
  if (englishName && COOKIE_IMAGES[englishName]) {
    return COOKIE_IMAGES[englishName];
  }

  return null;
}

// 대표 쿠키들 (배경 장식용)
export const FEATURED_COOKIES = [
  {
    name: 'Madeleine Cookie',
    nameKr: '마들렌맛 쿠키',
    image: COOKIE_IMAGES['Madeleine Cookie'],
    grade: 'Epic',
  },
  {
    name: 'Pure Vanilla Cookie',
    nameKr: '퓨어바닐라 쿠키',
    image: COOKIE_IMAGES['Pure Vanilla Cookie'],
    grade: 'Ancient',
  },
  {
    name: 'Dark Cacao Cookie',
    nameKr: '다크카카오 쿠키',
    image: COOKIE_IMAGES['Dark Cacao Cookie'],
    grade: 'Ancient',
  },
  {
    name: 'Hollyberry Cookie',
    nameKr: '홀리베리 쿠키',
    image: COOKIE_IMAGES['Hollyberry Cookie'],
    grade: 'Ancient',
  },
  {
    name: 'Espresso Cookie',
    nameKr: '에스프레소맛 쿠키',
    image: COOKIE_IMAGES['Espresso Cookie'],
    grade: 'Epic',
  },
];

export default COOKIE_IMAGES;
