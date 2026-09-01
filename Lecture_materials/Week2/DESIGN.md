# Kami Academic · 생성형 AI

- Surface: parchment `#F5F4ED`, ivory panels `#FAF9F5`
- Accent: ink blue `#1B365D`만 사용하며 Outline과 Summary만 어두운 장표로 구성한다.
- Type: 제목은 `Noto Serif KR` 500, 본문은 `Pretendard` 또는 `Noto Sans KR`, 수식·코드는 고정폭 글꼴을 사용한다.
- Canvas: 1920×1080, 배포용 Markdown을 장표 단위로 대응하되 사용자 피드백으로 제외한 장표는 HTML에서 생략한다.
- Hierarchy: 장표 제목 58–68px, 본문 28–34px, 보조 정보 20–24px로 투사 가독성을 유지한다.
- Layout: 두 개의 `###`는 두 패널로, 수식은 설명과 수식 카드를 나란히, 표는 비교 기준과 표를 같은 장에 배치한다.
- Density: 큰 표·수식과 긴 설명을 동시에 쌓지 않고, 화면 경계를 넘으면 글자를 줄이기보다 간격과 보조 문구를 먼저 조정한다.
- Interaction: 방향키·PageUp/PageDown·Space·Home/End, 휠, 터치, ESC 개요, 현재 장 번호, localStorage, print CSS를 제공한다.
- Student view: 제작 방식·템플릿명·버전·경로 같은 메타데이터를 장표 화면에 표시하지 않는다.

시각 언어는 로컬 `kami-deck` 템플릿의 종이 질감, 잉크색, 절제된 타이포그래피 원칙을 한국어 대학 강의에 맞게 적용한다.
