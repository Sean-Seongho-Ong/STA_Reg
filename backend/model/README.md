---
base_model: meta-llama/Llama-2-13b-chat-hf
library_name: peft
---

# QLoRA LLM 챗봇 API

FCC 규제 관련 문서를 기반으로 한 QLoRA 파인튜닝된 Llama2 챗봇 API 서버입니다.

## 주요 기능

- **QLoRA 파인튜닝된 LLM**: Llama2 13B 모델을 기반으로 FCC 규제 문서에 특화된 응답 생성
- **RAG (Retrieval-Augmented Generation)**: Qdrant 벡터 데이터베이스를 활용한 문서 검색 및 응답 생성
- **스트리밍 응답**: 실시간 응답 생성 지원
- **요약 기능**: 긴 응답에 대한 자동 요약 제공

## 시스템 요구사항

- Python 3.9 이상
- CUDA 지원 GPU (최소 16GB VRAM 권장)
- 최소 32GB RAM
- Qdrant 서버 (벡터 데이터베이스)

## 설치 방법

1. 저장소 클론:
```bash
git clone [repository_url]
cd ChatBot_interface_Web/backend
```

2. 가상환경 생성 및 활성화:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\\Scripts\\activate   # Windows
```

3. 의존성 설치:
```bash
pip install -r requirements.txt
```

4. 환경 변수 설정:
`.env` 파일을 생성하고 다음 변수들을 설정:

```env
# 환경 설정
ENV=development  # development 또는 production

# 모델 경로 설정
LOCAL_MODEL_PATH="path/to/model/directory"
LOCAL_BASE_MODEL_PATH="path/to/model/llama2"
LOCAL_ADAPTER_PATH="path/to/model/adapter"

# Qdrant 설정
QDRANT_HOST="localhost"
QDRANT_PORT=6333
COLLECTION_NAME="fcc_kdb_docs"

# Hugging Face 설정
HF_TOKEN="your_token_here"  # Hugging Face 토큰
```

## 실행 방법

1. Qdrant 서버 시작:
```bash
docker run -p 6333:6333 qdrant/qdrant
```

2. API 서버 시작:
```bash
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## API 엔드포인트

### 1. 채팅 API
- **엔드포인트**: `/api/chat`
- **메서드**: POST
- **요청 형식**:
```json
{
    "messages": [
        {"role": "user", "content": "질문 내용"}
    ],
    "max_tokens": 2048,
    "temperature": 0.7,
    "stream": false
}
```

### 2. 상태 확인
- **엔드포인트**: `/api/status`
- **메서드**: GET

### 3. 건강 체크
- **엔드포인트**: `/api/health`
- **메서드**: GET

## 주요 구성 요소

1. **LLM 모듈**
   - QLoRA 파인튜닝된 Llama2 모델
   - 4비트 양자화를 통한 메모리 최적화
   - GPU 오프로딩 지원

2. **RAG 시스템**
   - Qdrant 벡터 데이터베이스 연동
   - 문서 임베딩 및 검색
   - 컨텍스트 기반 응답 생성

3. **파이프라인**
   - 문서 검색 (Retrieval)
   - 응답 생성 (Generation)
   - 요약 (Summarization)

## 개발 가이드

### 디렉토리 구조
```
backend/
├── app/
│   ├── models/      # 모델 관련 로직
│   ├── core/        # 핵심 기능
│   ├── api/         # API 엔드포인트
│   └── utils/       # 유틸리티 함수
├── model/          # 모델 파일 저장
└── main.py         # 앱 진입점
```

### 환경 변수 설명

- `ENV`: 개발/배포 환경 설정
- `LOCAL_MODEL_PATH`: 모델 파일 저장 경로
- `LOCAL_BASE_MODEL_PATH`: Llama2 기본 모델 경로
- `LOCAL_ADAPTER_PATH`: QLoRA 어댑터 경로
- `QDRANT_HOST`: Qdrant 서버 주소
- `QDRANT_PORT`: Qdrant 서버 포트
- `COLLECTION_NAME`: Qdrant 컬렉션 이름

## 라이선스

[라이선스 정보 추가]

## Model Card for Model ID

<!-- Provide a quick summary of what the model is/does. -->



## Model Details

### Model Description

<!-- Provide a longer summary of what this model is. -->



- **Developed by:** [More Information Needed]
- **Funded by [optional]:** [More Information Needed]
- **Shared by [optional]:** [More Information Needed]
- **Model type:** [More Information Needed]
- **Language(s) (NLP):** [More Information Needed]
- **License:** [More Information Needed]
- **Finetuned from model [optional]:** [More Information Needed]

### Model Sources [optional]

<!-- Provide the basic links for the model. -->

- **Repository:** [More Information Needed]
- **Paper [optional]:** [More Information Needed]
- **Demo [optional]:** [More Information Needed]

## Uses

<!-- Address questions around how the model is intended to be used, including the foreseeable users of the model and those affected by the model. -->

### Direct Use

<!-- This section is for the model use without fine-tuning or plugging into a larger ecosystem/app. -->

[More Information Needed]

### Downstream Use [optional]

<!-- This section is for the model use when fine-tuned for a task, or when plugged into a larger ecosystem/app -->

[More Information Needed]

### Out-of-Scope Use

<!-- This section addresses misuse, malicious use, and uses that the model will not work well for. -->

[More Information Needed]

## Bias, Risks, and Limitations

<!-- This section is meant to convey both technical and sociotechnical limitations. -->

[More Information Needed]

### Recommendations

<!-- This section is meant to convey recommendations with respect to the bias, risk, and technical limitations. -->

Users (both direct and downstream) should be made aware of the risks, biases and limitations of the model. More information needed for further recommendations.

## How to Get Started with the Model

Use the code below to get started with the model.

[More Information Needed]

## Training Details

### Training Data

<!-- This should link to a Dataset Card, perhaps with a short stub of information on what the training data is all about as well as documentation related to data pre-processing or additional filtering. -->

[More Information Needed]

### Training Procedure

<!-- This relates heavily to the Technical Specifications. Content here should link to that section when it is relevant to the training procedure. -->

#### Preprocessing [optional]

[More Information Needed]


#### Training Hyperparameters

- **Training regime:** [More Information Needed] <!--fp32, fp16 mixed precision, bf16 mixed precision, bf16 non-mixed precision, fp16 non-mixed precision, fp8 mixed precision -->

#### Speeds, Sizes, Times [optional]

<!-- This section provides information about throughput, start/end time, checkpoint size if relevant, etc. -->

[More Information Needed]

## Evaluation

<!-- This section describes the evaluation protocols and provides the results. -->

### Testing Data, Factors & Metrics

#### Testing Data

<!-- This should link to a Dataset Card if possible. -->

[More Information Needed]

#### Factors

<!-- These are the things the evaluation is disaggregating by, e.g., subpopulations or domains. -->

[More Information Needed]

#### Metrics

<!-- These are the evaluation metrics being used, ideally with a description of why. -->

[More Information Needed]

### Results

[More Information Needed]

#### Summary



## Model Examination [optional]

<!-- Relevant interpretability work for the model goes here -->

[More Information Needed]

## Environmental Impact

<!-- Total emissions (in grams of CO2eq) and additional considerations, such as electricity usage, go here. Edit the suggested text below accordingly -->

Carbon emissions can be estimated using the [Machine Learning Impact calculator](https://mlco2.github.io/impact#compute) presented in [Lacoste et al. (2019)](https://arxiv.org/abs/1910.09700).

- **Hardware Type:** [More Information Needed]
- **Hours used:** [More Information Needed]
- **Cloud Provider:** [More Information Needed]
- **Compute Region:** [More Information Needed]
- **Carbon Emitted:** [More Information Needed]

## Technical Specifications [optional]

### Model Architecture and Objective

[More Information Needed]

### Compute Infrastructure

[More Information Needed]

#### Hardware

[More Information Needed]

#### Software

[More Information Needed]

## Citation [optional]

<!-- If there is a paper or blog post introducing the model, the APA and Bibtex information for that should go in this section. -->

**BibTeX:**

[More Information Needed]

**APA:**

[More Information Needed]

## Glossary [optional]

<!-- If relevant, include terms and calculations in this section that can help readers understand the model or model card. -->

[More Information Needed]

## More Information [optional]

[More Information Needed]

## Model Card Authors [optional]

[More Information Needed]

## Model Card Contact

[More Information Needed]
### Framework versions

- PEFT 0.14.0