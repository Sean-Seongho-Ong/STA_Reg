from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, TypedDict, Annotated, Sequence
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
from peft import PeftModel, PeftConfig
import logging
import time
import uuid
import asyncio
from fastapi.responses import StreamingResponse
import sys
from langchain_community.llms import HuggingFacePipeline.
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models
from huggingface_hub import login, snapshot_download
import qdrant_client
import traceback
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import StateGraph, END
from langchain_core.documents import Document
from dotenv import load_dotenv
import shutil

# GraphState 정의
class GraphState(TypedDict):
    """RAG 및 요약 파이프라인 상태"""
    question: str  # 사용자 원본 질문
    documents: Sequence[Document]  # 검색된 문서 리스트
    initial_answer: str  # RAG 파이프라인이 생성한 초기 답변
    final_answer: str  # 최종 요약된 답변
    error: Optional[str]  # 오류 발생 시 메시지 저장

load_dotenv()

login(token=os.getenv("HF_TOKEN"))

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="QLoRA LLM 챗봇 API")

# CORS 설정 - 프론트엔드 연결을 위해 필요
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],  # 프론트엔드 오리진
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)

# 환경 설정
ENV = os.getenv("ENV", "development")
LOCAL_BASE_MODEL_PATH = os.getenv("LOCAL_BASE_MODEL_PATH")
LOCAL_ADAPTER_PATH = os.getenv("LOCAL_ADAPTER_PATH")

# Qdrant 설정
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "regulatory_docs")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")

# 모델 설정
BASE_MODEL = "meta-llama/Llama-2-13b-chat-hf"
ADAPTER_REPO = "Sean-Ong/STA_Reg"
OFFLOAD_DIR = "./offload"
MODEL_CACHE_DIR = "./model_cache"

# 전역 변수 선언
model = None
tokenizer = None
base_model = None
local_llm = None
summarization_llm = None
is_initialized = False
qdrant_client_instance = None
embedding_model = None
vector_store = None
langgraph_app = None

# 디바이스 설정
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {device}")

def initialize_model_and_tokenizer():
    """모델과 토크나이저 초기화 (한 번만 실행)"""
    global model, tokenizer, base_model, local_llm, summarization_llm, is_initialized
    
    if is_initialized:
        logger.info("모델이 이미 초기화되어 있습니다.")
        return True
        
    try:
        if ENV == "development":
            logger.info("개발 환경: 로컬 모델 사용")
            
            # 4비트 양자화 설정
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                llm_int8_enable_fp32_cpu_offload=True
            )
            
            # 토크나이저 로드
            tokenizer = AutoTokenizer.from_pretrained(
                LOCAL_BASE_MODEL_PATH,
                use_fast=True,
                padding_side="left"
            )
            
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # 디바이스 매핑 설정
            device_map = {
                "model.embed_tokens": 0,
                "model.norm": 0,
                "lm_head": 0,
                "model.layers.0": 0,
                "model.layers.1": 0,
                "model.layers.2": 0,
                "model.layers.3": 0,
                "model.layers.4": 0,
                "model.layers.5": 0,
                "model.layers.6": 0,
                "model.layers.7": 0,
                "model.layers.8": 0,
                "model.layers.9": 0,
                "model.layers.10": 0,
                "model.layers.11": 0,
                "model.layers.12": 0,
                "model.layers.13": 0,
                "model.layers.14": 0,
                "model.layers.15": 0,
                "model.layers.16": 0,
                "model.layers.17": 0,
                "model.layers.18": 0,
                "model.layers.19": 0,
                "model.layers.20": 0,
                "model.layers.21": 0,
                "model.layers.22": 0,
                "model.layers.23": 0,
                "model.layers.24": 0,
                "model.layers.25": 0,
                "model.layers.26": 0,
                "model.layers.27": 0,
                "model.layers.28": 0,
                "model.layers.29": 0,
                "model.layers.30": 0,
                "model.layers.31": 0,
                "model.layers.32": 0,
                "model.layers.33": 0,
                "model.layers.34": 0,
                "model.layers.35": 0,
                "model.layers.36": 0,
                "model.layers.37": 0,
                "model.layers.38": 0,
                "model.layers.39": 0
            }
            
            # 기본 모델 로드
            base_model = AutoModelForCausalLM.from_pretrained(
                LOCAL_BASE_MODEL_PATH,
                device_map=device_map,
                quantization_config=quantization_config,
                torch_dtype=torch.float16,
                offload_folder=OFFLOAD_DIR
            )
            
            # QLoRA 어댑터 로드
            logger.info(f"로컬 어댑터 로드: {LOCAL_ADAPTER_PATH}")
            model = PeftModel.from_pretrained(
                base_model,
                LOCAL_ADAPTER_PATH,
                device_map=device_map,
                offload_folder=OFFLOAD_DIR
            )
            
            # 모델을 평가 모드로 설정
            model.eval()
            
            # 기본 파이프라인 생성
            pipe = pipeline(
                "text-generation",
                model=model.base_model,
                tokenizer=tokenizer,
                max_new_tokens=512,
                temperature=0.7,
                do_sample=True,
                device_map=device_map,
                torch_dtype=torch.float16
            )
            local_llm = HuggingFacePipeline(pipeline=pipe)
            logger.info("기본 LLM 파이프라인 생성 완료")
            
            # 요약용 파이프라인 생성
            summarization_pipe = pipeline(
                "text-generation",
                model=model.base_model,
                tokenizer=tokenizer,
                max_new_tokens=1024,
                temperature=0.3,
                do_sample=True,
                device_map=device_map,
                torch_dtype=torch.float16
            )
            summarization_llm = HuggingFacePipeline(pipeline=summarization_pipe)
            logger.info("요약용 파이프라인 생성 완료")
            
            is_initialized = True
            return True
        else:
            logger.info("프로덕션 환경: 원격 모델 사용")
            return False
            
    except Exception as e:
        logger.error(f"모델 초기화 중 오류 발생: {e}")
        logger.error(traceback.format_exc())
        return False

# 4비트 양자화 설정
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    llm_int8_enable_fp32_cpu_offload=True
)

@app.on_event("startup")
async def startup_event():
    """서버 시작 시 초기화"""
    global qdrant_client_instance, embedding_model, vector_store, langgraph_app
    
    try:
        # 모델 초기화
        if not initialize_model_and_tokenizer():
            logger.error("모델 초기화 실패")
            return
            
        # Qdrant 클라이언트 초기화
        qdrant_client_instance = QdrantClient(
            host=QDRANT_HOST,
            port=QDRANT_PORT
        )
        
        # 컬렉션 존재 여부 확인 및 생성
        try:
            collection_info = qdrant_client_instance.get_collection(collection_name=COLLECTION_NAME)
            logger.info(f"Qdrant 컬렉션 '{COLLECTION_NAME}' 확인 완료")
        except Exception as q_err:
            logger.warning(f"Qdrant 컬렉션 '{COLLECTION_NAME}'이 존재하지 않습니다. 새로 생성합니다.")
            try:
                # 임베딩 모델 초기화
                embedding_model = HuggingFaceEmbeddings(
                    model_name=EMBEDDING_MODEL_NAME,
                    model_kwargs={'device': device}
                )
                
                # 컬렉션 생성
                qdrant_client_instance.create_collection(
                    collection_name=COLLECTION_NAME,
                    vectors_config=qdrant_models.VectorParams(
                        size=384,  # MiniLM-L6-v2 임베딩 크기
                        distance=qdrant_models.Distance.COSINE
                    )
                )
                logger.info(f"Qdrant 컬렉션 '{COLLECTION_NAME}' 생성 완료")
            except Exception as create_err:
                logger.error(f"Qdrant 컬렉션 생성 실패: {create_err}")
                return
        
        # 임베딩 모델 초기화 (아직 초기화되지 않은 경우)
        if embedding_model is None:
            embedding_model = HuggingFaceEmbeddings(
                model_name=EMBEDDING_MODEL_NAME,
                model_kwargs={'device': device}
            )
        logger.info("임베딩 모델 로드 완료")
        
        # 벡터 저장소 초기화
        vector_store = Qdrant(
            client=qdrant_client_instance,
            collection_name=COLLECTION_NAME,
            embeddings=embedding_model
        )
        logger.info("벡터 저장소 초기화 완료")
        
        # LangGraph 워크플로우 빌드
        try:
            workflow = StateGraph(GraphState)
            workflow.add_node("retrieve", retrieve_node)
            workflow.add_node("generate_rag_answer", generate_rag_answer_node)
            workflow.add_node("summarize", summarize_answer_node)
            workflow.set_entry_point("retrieve")
            workflow.add_edge("retrieve", "generate_rag_answer")
            workflow.add_edge("generate_rag_answer", "summarize")
            workflow.add_edge("summarize", END)
            langgraph_app = workflow.compile()
            logger.info("LangGraph 워크플로우 컴파일 완료")
        except Exception as graph_err:
            logger.error(f"LangGraph 워크플로우 컴파일 실패: {graph_err}")
            logger.error(traceback.format_exc())
            langgraph_app = None
        
    except Exception as e:
        logger.error(f"startup_event 오류: {e}")
        logger.error(traceback.format_exc())
        langgraph_app = None

# 기본 LLM 파이프라인 생성
if model is not None and tokenizer is not None:
    try:
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=512, # 기본 답변 생성 시 토큰 수
            temperature=0.7,    # 기본 temperature
            do_sample=True,     # 기본 샘플링 활성화
            device_map="auto"
        )
        local_llm = HuggingFacePipeline(pipeline=pipe)
        logger.info("HuggingFacePipeline (로컬 LLM 래퍼) 생성 완료.")
    except Exception as e:
        logger.error(f"HuggingFacePipeline 생성 중 오류: {str(e)}")
        local_llm = None

# --- [요약용 수정] 요약 전용 LLM 파이프라인 생성 ---
if local_llm: # 기본 LLM 생성 성공 시에만 시도
     try:
         logger.info("Creating dedicated pipeline for summarization...")
         summarization_pipe = pipeline(
             "text-generation",
             model=model, # 동일 모델 사용
             tokenizer=tokenizer, # 동일 토크나이저 사용
             max_new_tokens=1024, # <<< 요약 최대 토큰 수 제한 (필요시 조절)
             temperature=0.3,    # <<< 낮은 temperature 설정
             do_sample=True,        # <<< 샘플링 활성화 (UserWarning 해결 시도)
             top_p=0.9,
             repetition_penalty=1.4,
             device_map="auto"   # device_map 설정 유지
         )
         summarization_llm = HuggingFacePipeline(pipeline=summarization_pipe)
         logger.info("HuggingFacePipeline for Summarization created.")
     except Exception as e:
         logger.error(f"Failed to create summarization pipeline: {e}")
         summarization_llm = None # 실패 시 None 처리
# --- 수정 끝 ---

torch.cuda.empty_cache()  # GPU 메모리 캐시 정리

# 기존 offload 디렉토리 삭제 및 재생성
OFFLOAD_DIR = "./offload"
if os.path.exists(OFFLOAD_DIR):
    shutil.rmtree(OFFLOAD_DIR)
os.makedirs(OFFLOAD_DIR, exist_ok=True)

# Pydantic 모델
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    max_tokens: Optional[int] = 2048
    temperature: Optional[float] = 0.7
    stream: Optional[bool] = False

class ChatResponse(BaseModel):
    response: str
    message_id: str

# 채팅 응답 생성 함수
def generate_response(messages, max_tokens=2048, temperature=0.7):
    # 모델이 로드되지 않은 경우 기본 응답 반환
    if model is None or tokenizer is None:
        return get_fallback_response("") # type: ignore
    
    # 사용자 메시지 로깅
    last_user_message = ""
    for message in messages:
        if message.role == "user":
            last_user_message = message.content.strip().lower()
    
    logger.info(f"사용자 메시지: '{last_user_message}'")
    
    # Llama2 채팅 형식으로 변환 - 표준 형식 사용
    prompt = ""
    for message in messages:
        if message.role == "user":
            prompt += f"<s>[INST] {message.content} [/INST]"
        elif message.role == "assistant":
            prompt += f" {message.content} </s>"
        elif message.role == "system" and len(prompt) == 0:
            # 시스템 메시지는 첫 번째 사용자 메시지 앞에 추가
            prompt += f"<s>[INST] <<SYS>>\n{message.content}\n<</SYS>>\n\n"
    
    # 마지막 메시지가 사용자인 경우, 응답 시작부 형식 정의
    if messages[-1].role == "user":
        prompt += " " # 프롬프트와 응답 사이 경계
    
    logger.info(f"프롬프트 길이: {len(prompt)}, 끝부분 5자: '{prompt[-5:]}'")
    
    # 토크나이저 설정 - padding 관련 오류 방지
    # 패딩 토큰 다시 확인
    if tokenizer.pad_token is None:
        logger.warning("패딩 토큰이 없습니다. EOS 토큰을 패딩 토큰으로 설정합니다.")
        tokenizer.pad_token = tokenizer.eos_token
    
    inputs = tokenizer(
        prompt, 
        return_tensors="pt",
        truncation=True,
        max_length=4096 - max_tokens
    ).to(model.device)
    
    # 생성 파라미터 설정
    gen_config = {
        "max_new_tokens": max_tokens, 
        "temperature": temperature,
        "do_sample": temperature > 0,
        "top_p": 0.95,
        "top_k": 50,
        "repetition_penalty": 1.1,
        "pad_token_id": tokenizer.eos_token_id
    }
    
    try:
        # 텍스트 생성
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                **gen_config
            )
        
        # 전체 응답 디코딩
        full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        logger.info(f"전체 응답 길이: {len(full_response)}, 처음 50자: '{full_response[:50]}'")
        
        # --- 수정된 응답 분리 로직 ---
        last_inst_tag = "[/INST]"
        inst_tag_pos = full_response.rfind(last_inst_tag) # 마지막 [/INST] 태그 위치 찾기

        if inst_tag_pos != -1:
            # [/INST] 태그 이후의 텍스트를 응답으로 간주
            response = full_response[inst_tag_pos + len(last_inst_tag):].strip()
            logger.info(f"마지막 '[/INST]' 태그 위치: {inst_tag_pos}, 태그 이후부터 응답 추출")
        else:
            # [/INST] 태그를 찾지 못한 경우, 이전 fallback 로직 사용 (하지만 개선된 로직으로 인해 거의 발생하지 않아야 함)
            logger.warning("응답에서 '[/INST]' 태그를 찾지 못했습니다. 이전 fallback 로직을 사용합니다.")
            # 기존 prompt 변수를 사용하기보다, 입력 메시지 마지막 [/INST]를 기준으로 시도
            prompt_end_marker_in_input = prompt.rfind(last_inst_tag)
            if prompt_end_marker_in_input != -1:
                 # 입력 프롬프트의 마지막 [/INST] 다음부터 비교 시작점을 찾아 잘라내기 시도
                 potential_response_start = full_response[prompt_end_marker_in_input + len(last_inst_tag):]
                 # 그래도 안전하게 첫글자가 날아가지 않도록 확인
                 if full_response.startswith(prompt[:prompt_end_marker_in_input + len(last_inst_tag)]):
                     response = potential_response_start.strip()
                     logger.info("Fallback: 입력 프롬프트 기반으로 응답 추출 시도 성공")
                 else:
                     logger.warning("Fallback: 입력 프롬프트와 응답 시작 불일치, 전체 프롬프트 길이 사용")
                     response = full_response[len(prompt):].strip() # 최후의 수단
            else:
                 logger.warning("Fallback: 입력 프롬프트에서도 '[/INST]' 찾지 못함, 전체 프롬프트 길이 사용")
                 response = full_response[len(prompt):].strip() # 최후의 수단

        # --- 수정된 로직 끝 ---
        
        logger.info(f"최종 응답 길이: {len(response)}")
        
        # 응답이 없는 경우 기본 메시지 반환
        if not response or len(response.strip()) == 0:
            return "죄송합니다, 답변을 생성할 수 없습니다. 다른 질문을 시도해주세요."
        
        # 응답에 태그가 포함되어 있는지 확인하고 제거
        if "<RESPONSE_START>" in response or "</RESPONSE_END>" in response or "E_START>" in response:
            logger.warning("응답에 HTML 태그가 포함되어 있어 제거합니다.")
            response = response.replace("<RESPONSE_START>", "").replace("</RESPONSE_END>", "").replace("E_START>", "")
        
        # 최종 응답 반환
        return response
    except Exception as e:
        logger.error(f"텍스트 생성 중 오류: {str(e)}")
        return f"죄송합니다. 응답 생성 중 오류가 발생했습니다: {str(e)}"

# 스트리밍 응답 생성 함수
async def generate_stream_response(messages, max_tokens=2048, temperature=0.7):
    # 모델이 로드되지 않은 경우
    if model is None or tokenizer is None:
        yield f"data: {get_fallback_response('')}\n\n" # type: ignore
        yield f"data: [DONE]\n\n"
        return
    
    # Llama2 채팅 형식으로 변환
    prompt = ""
    for message in messages:
        if message.role == "user":
            prompt += f"<s>[INST] {message.content} [/INST]"
        elif message.role == "assistant":
            prompt += f" {message.content} </s>"
        elif message.role == "system" and len(prompt) == 0:
            prompt += f"<s>[INST] <<SYS>>\n{message.content}\n<</SYS>>\n\n"
    
    if messages[-1].role == "user":
        prompt += " "
    
    try:
        # 패딩 토큰 확인
        if tokenizer.pad_token is None:
            logger.warning("스트리밍: 패딩 토큰이 없습니다. EOS 토큰을 패딩 토큰으로 설정합니다.")
            tokenizer.pad_token = tokenizer.eos_token
        
        # 토크나이저 설정 개선 - padding 관련 오류 방지
        inputs = tokenizer(
            prompt, 
            return_tensors="pt",
            truncation=True,
            max_length=4096 - max_tokens
        ).to(model.device)
        
        # 생성 설정
        gen_config = {
            "max_new_tokens": max_tokens, 
            "temperature": temperature,
            "do_sample": temperature > 0,
            "top_p": 0.95,
            "top_k": 50,
            "repetition_penalty": 1.1,
            "pad_token_id": tokenizer.eos_token_id
        }
        
        # 스트리밍 텍스트 생성
        streamer = None  # 실제 구현에서는 스트리머 추가 가능
        
        # 간단한 스트리밍 시뮬레이션
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                **gen_config
            )
        
        full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 프롬프트 기반 응답 분리 - 정확한 방법
        prompt_end_pos = 0
        for i in range(min(len(prompt), len(full_response))):
            if i >= len(full_response) or i >= len(prompt) or full_response[i] != prompt[i]:
                prompt_end_pos = i
                break
        
        if prompt_end_pos > 0:
            response_text = full_response[prompt_end_pos:].strip()
        else:
            response_text = full_response[len(prompt):].strip()
        
        # 단어 단위로 스트리밍 시뮬레이션
        words = response_text.split()
        
        for i in range(len(words)):
            chunk = " ".join(words[:i+1])
            yield f"data: {chunk}\n\n"
            await asyncio.sleep(0.05)  # 실제 스트리밍처럼 보이도록 지연
        
        yield f"data: [DONE]\n\n"
    except Exception as e:
        logger.error(f"스트리밍 응답 생성 중 오류: {str(e)}")
        yield f"data: 죄송합니다. 응답 생성 중 오류가 발생했습니다: {str(e)}\n\n"
        yield f"data: [DONE]\n\n"

# --- LangGraph 노드 함수 정의 ---

def retrieve_node(state):
    """문서 검색 노드"""
    try:
        question = state["question"]
        
        # 질문의 임베딩 생성
        if "embedding" not in state:
            try:
                question_embedding = embedding_model.embed_query(question)
                state["embedding"] = question_embedding
            except Exception as e:
                return {"error": f"Embedding generation failed: {e}"}
        
        # 검색 매개변수 설정
        search_params = {
            "query_vector": state["embedding"],
            "limit": 5,
            "score_threshold": 0.5,
            "with_payload": True
        }
        
        # Qdrant 검색 실행
        search_results = qdrant_client_instance.search(
            collection_name=COLLECTION_NAME,
            **search_params
        )
        
        # 검색 결과 처리
        if not search_results:
            state["documents"] = []
            return state
            
        # 검색 결과를 Document 객체로 변환
        documents = []
        for result in search_results:
            doc = Document(
                page_content=result.payload.get('page_content', ''),
                metadata={
                    'score': result.score,
                    'source': result.payload.get('source', ''),
                    'page': result.payload.get('page', ''),
                    'kdb_number': result.payload.get('kdb_number', ''),
                    'second_category': result.payload.get('second_category', '')
                }
            )
            documents.append(doc)
        
        state["documents"] = documents
        return state
        
    except Exception as e:
        state["documents"] = []
        state["error"] = f"Document retrieval failed: {e}"
        return state

def generate_rag_answer_node(state: GraphState):
    """검색된 문서와 질문으로 초기 RAG 답변 생성"""
    logger.info("--- Node: generate_rag_answer ---")
    
    try:
        question = state['question']
        documents = state.get('documents', [])
        
        if not documents:
            logger.warning("문서가 없어 직접 답변 생성")
            prompt_input = f"""<s>[INST] <<SYS>>
You are a certification expert specializing in RF regulations. Provide a comprehensive and detailed answer based on your knowledge.
<</SYS>>

Question: {question}

Provide a detailed and informative answer that includes:
1. Definition and explanation
2. Key technical specifications
3. Regulatory requirements
4. Practical implications
[/INST]"""
            response = local_llm.invoke(prompt_input)
            if "[/INST]" in response:
                response = response.split("[/INST]")[-1].strip()
            return {"initial_answer": response, "error": None}
            
        # 문서 내용 조합
        context_parts = []
        for doc in documents:
            score = doc.metadata.get('score', 'N/A')
            content = doc.page_content
            context_parts.append(f"[관련성 점수: {score}]\n{content}")
        
        context = "\n\n".join(context_parts)
        
        # RAG 프롬프트 생성
        prompt_input = f"""<s>[INST] <<SYS>>
You are a certification expert specializing in RF regulations. Your task is to provide a comprehensive and detailed answer based on the provided information.

Guidelines for answering:
1. Start with a clear definition and explanation
2. Include all relevant technical specifications
3. Explain regulatory requirements in detail
4. Provide practical implications and applications
5. Use specific examples when possible
6. Ensure technical accuracy while being clear and understandable
<</SYS>>

Based on the following information, please provide a comprehensive answer:

Information:
{context}

Question: {question}

Provide a detailed answer that covers all aspects of the question:
[/INST]"""
        
        # 텐서를 올바른 디바이스로 이동
        with torch.cuda.device(device):
            response = local_llm.invoke(prompt_input)
            
        if "[/INST]" in response:
            response = response.split("[/INST]")[-1].strip()
        
        logger.info(f"Generated initial RAG answer (length: {len(response)})")
        return {"initial_answer": response, "error": None}
        
    except Exception as e:
        logger.error(f"Error during initial RAG answer generation: {e}")
        return {"initial_answer": "", "error": str(e)}

def summarize_answer_node(state: GraphState): # type: ignore
    """초기 RAG 답변을 요약하고, 초기 답변과 요약 모두 반환하는 노드"""
    logger.info("--- Node: summarize_answer ---")
    initial_answer = state['initial_answer']
    
    # 오류 또는 빈 답변 처리
    if state.get('error') or not initial_answer:
        logger.warning("Skipping summarization due to previous error or empty initial answer.")
        return {
            "initial_answer": initial_answer,
            "final_answer": initial_answer or "요약을 생성할 수 없습니다.",
            "error": state.get('error')
        }

    # 입력이 너무 길면 잘라내기 (안정성 증가)
    initial_answer = initial_answer[:4000]
    
    # "답변:" 이후 부분만 추출 (있는 경우)
    if "답변:" in initial_answer:
        processed_answer = initial_answer.split("답변:")[-1].strip()
    else:
        processed_answer = initial_answer

    # 요약 프롬프트 정의
    prompt_input = f"""<s>[INST] <<SYS>>
You are a certification expert specializing in RF regulations. Your task is to provide a clear and comprehensive summary of the technical information.

Guidelines for summarization:
1. Start with a clear definition
2. Include key technical specifications
3. Explain regulatory requirements
4. Provide practical implications
5. Maintain technical accuracy while being concise
6. Use bullet points or numbered lists for clarity
7. Include specific examples when relevant

Format your response as:
DEFINITION:
[Clear definition of the concept]

TECHNICAL SPECIFICATIONS:
[Key technical details and requirements]

REGULATORY REQUIREMENTS:
[Specific regulatory standards and guidelines]

PRACTICAL IMPLICATIONS:
[How this applies in real-world scenarios]

<</SYS>>

Content to summarize:
{processed_answer}

Provide a comprehensive technical summary:
[/INST]"""

    try:
        # 요약 생성
        if summarization_llm:
            logger.info("Using dedicated summarization LLM.")
            raw_summary = summarization_llm.invoke(prompt_input)
        else:
            logger.warning("Summarization LLM not available, falling back to default LLM.")
            raw_summary = local_llm.invoke(prompt_input)

        # 응답에서 프롬프트 제거
        if "[/INST]" in raw_summary:
            summary = raw_summary.split("[/INST]")[-1].strip()
        else:
            summary = raw_summary.strip()

        # 결과 로깅
        logger.info(f"Summary Length: {len(summary)}")
        logger.info(f"Summary Preview: {summary[:200]}")

        return {
            "initial_answer": initial_answer,
            "final_answer": summary,
            "error": None
        }
    except Exception as e:
        logger.error(f"Error during summarization: {e}")
        return {
            "initial_answer": initial_answer,
            "final_answer": initial_answer,
            "error": f"Summarization failed: {str(e)}"
        }

# API 엔드포인트
@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    global langgraph_app
    
    # 스트리밍 요청은 아직 LangGraph RAG 미지원
    if request.stream:
        logger.warning("Streaming is not supported with the LangGraph RAG pipeline.")
        raise HTTPException(status_code=400, detail="Streaming is not supported with RAG yet.")

    # LangGraph 앱 사용 가능 여부 확인
    if langgraph_app is None:
        logger.warning("LangGraph RAG pipeline is not available. Falling back to basic generation.")
        # Fallback 로직: 기존 generate_response 사용
        try:
            response = generate_response(request.messages, request.max_tokens, request.temperature)
            safe_response = f"<RESPONSE_START>{response}</RESPONSE_END>"
            return {"response": safe_response, "message_id": str(uuid.uuid4()) + "-fallback"}
        except Exception as fallback_e:
            logger.error(f"Fallback response generation failed: {fallback_e}")
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=503, detail="RAG pipeline and fallback are not available.")

    # LangGraph 파이프라인 실행
    try:
        start_time = time.time()
        user_question = request.messages[-1].content
        logger.info(f"Processing question with LangGraph RAG + Summarization: '{user_question}'")

        # LangGraph 실행
        inputs = {"question": user_question}
        final_state = langgraph_app.invoke(inputs)

        # 결과 추출
        initial_answer = final_state.get('initial_answer', '')
        final_answer = final_state.get('final_answer', '')
        error_message = final_state.get('error')

        if error_message:
            logger.error(f"LangGraph execution finished with error: {error_message}")
            final_answer = final_answer or initial_answer or f"오류가 발생했습니다: {error_message}"

        if not final_answer:
            final_answer = "죄송합니다, 답변을 생성하지 못했습니다."

        # 두 답변을 모두 포함한 응답 형식
        combined_response = f"""<RESPONSE_START>
<RAG_INITIAL_ANSWER>
{initial_answer}
</RAG_INITIAL_ANSWER>

<SUMMARY_ANSWER>
{final_answer}
</SUMMARY_ANSWER>
</RESPONSE_END>"""

        # 처리 시간 로깅
        process_time = time.time() - start_time
        logger.info(f"LangGraph RAG+Summary response generated: {process_time:.2f}s")
        logger.info(f"Initial Answer Length: {len(initial_answer)}, Summary Length: {len(final_answer)}")

        return {
            "response": combined_response,
            "message_id": str(uuid.uuid4()) + "-rag-summary"
        }
    except Exception as e:
        logger.error(f"Error during LangGraph RAG pipeline execution: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error processing RAG request with LangGraph: {str(e)}")

# 서버 상태 확인 엔드포인트
@app.get("/api/health")
async def health_check():
    return {"status": "healthy", "model": "Llama-2-13b-chat-hf with QLoRA"}

# 테스트 엔드포인트
@app.get("/api/test")
async def test():
    return {"message": "API 서버가 정상적으로 작동 중입니다."}

# Qdrant 클라이언트 초기화 함수 추가
def init_qdrant_client():
    try:
        client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        logger.info(f"Qdrant 클라이언트 초기화 성공 (host: {QDRANT_HOST}, port: {QDRANT_PORT})")
        return client
    except Exception as e:
        logger.error(f"Qdrant 클라이언트 초기화 실패: {str(e)}")
        return None

# 메인 실행
if __name__ == "__main__":
    import uvicorn
    try:
        print("서버를 시작합니다...")
        uvicorn.run(
            "main:app",
            host="0.0.0.0",
            port=8000,
            reload=True,
            reload_dirs=["./"]
        )
    except Exception as e:
        print(f"서버 시작 중 오류 발생: {e}")
        print(traceback.format_exc()) 