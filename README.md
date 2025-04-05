# 곧 폐쇄 합니다~~처남 => 새롭게 Sean-Seongho-Ong/Chatbot 을 사용해~~

# QLoRA LLM 챗봇 웹 인터페이스

이 프로젝트는 QLoRA 방식으로 미세 조정된 LLM(Large Language Model) 기반 챗봇을 위한 웹 인터페이스를 제공합니다.

## 프로젝트 구조

```
ChatBot_interface_Web/
├── backend/           # FastAPI 기반 백엔드 서버
├── frontend/          # React 기반 프론트엔드
│   ├── public/        # 정적 파일
│   └── src/           # 소스 코드
│       ├── components/  # React 컴포넌트
│       └── ...        
├── manage.bat         # 프로젝트 관리 도구
├── run_chatbot.bat    # 전체 애플리케이션 실행
├── start_backend.bat  # 백엔드만 실행
└── start_frontend.bat # 프론트엔드만 실행
```

## 주요 기능

- 사용자 친화적인 채팅 인터페이스
- 마크다운 형식 지원 (코드 블록, 표, 목록 등)
- 백엔드 API 상태 모니터링
- 응답 생성 중 로딩 표시
- 오류 처리 및 알림

## 시작하기

### 사전 요구사항

- Node.js (v14 이상)
- Python (v3.8 이상)
- pip (Python 패키지 관리자)

### 설치 및 실행

1. 프로젝트를 클론 또는 다운로드합니다.
2. 명령 프롬프트 또는 PowerShell에서 프로젝트 루트 디렉토리로 이동합니다.
3. 다음 방법 중 하나로 애플리케이션을 실행할 수 있습니다:

#### 방법 1: 관리 도구 사용

`manage.bat` 파일을 실행합니다:

```
.\manage.bat
```

메뉴에서 원하는 작업을 선택합니다:
- `[1]` 전체 애플리케이션 실행 (백엔드 + 프론트엔드)
- `[2]` 백엔드만 실행
- `[3]` 프론트엔드만 실행
- `[4]` 프론트엔드 패키지 설치
- `[5]` 프록시 패키지 설치
- `[6]` React 앱 초기화 (주의: 기존 코드 삭제됨)
- `[7]` 실행 방법 안내
- `[8]` 종료

#### 방법 2: 직접 실행 (권장)

간단하게 전체 애플리케이션을 한 번에 실행하려면:

```
.\run_chatbot.bat
```

이 스크립트는 백엔드 서버를 먼저 시작하고, 30초 후에 자동으로 프론트엔드를 실행합니다.

#### 방법 3: 서버 별도 실행

백엔드 서버만 실행하려면:

```
.\start_backend.bat
```

프론트엔드만 별도로 실행하려면 (백엔드 서버가 이미 실행 중이어야 함):

```
.\start_frontend.bat
```

4. 브라우저에서 `http://localhost:3000`으로 접속하여 챗봇 인터페이스를 사용합니다.

> **주의**: PowerShell에서는 현재 디렉토리의 스크립트 실행 시 반드시 `.\` 접두사를 사용해야 합니다.

## 백엔드 API 엔드포인트

- `/api/health` - 서버 상태 확인
- `/api/test` - API 연결 테스트
- `/api/chat` - 채팅 메시지 처리
- `/api/status` - API 초기화 상태 확인

## 기술 스택

### 프론트엔드
- React.js
- Material-UI
- Axios
- React Markdown

### 백엔드
- FastAPI
- Transformers
- Hugging Face 모델
- QLoRA 미세조정

## 개발 정보

- 프론트엔드 개발 서버: http://localhost:3000
- 백엔드 API 서버: http://localhost:8000
- API 문서: http://localhost:8000/docs

## 문제 해결

### 백엔드 서버에 연결할 수 없음
- 백엔드 서버가 실행 중인지 확인하세요
- 포트 8000이 다른 프로세스에 의해 사용 중인지 확인하세요
- 백엔드 모델 로딩 시간이 오래 걸릴 수 있습니다. `run_chatbot.bat`를 사용하면 자동으로 대기합니다.

### 프론트엔드 실행 오류
- Node.js 버전이 호환되는지 확인하세요
- 필요한 패키지가 모두 설치되었는지 확인하세요 (메뉴 옵션 4, 5 사용)

### PowerShell에서 스크립트 실행 오류
- PowerShell에서는 현재 디렉토리의 스크립트 실행 시 `.\` 접두사를 반드시 사용해야 합니다.
- 예: `.\run_chatbot.bat`, `.\manage.bat`

## 라이선스

MIT 
