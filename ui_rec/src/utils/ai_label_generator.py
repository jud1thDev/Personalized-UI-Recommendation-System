"""
AI 기반 라벨 생성 모듈
Hugging Face 모델을 사용해서 기능 그룹의 제목을 자동 생성
"""

import re
import logging
from typing import List, Dict, Optional
from collections import Counter

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from transformers import pipeline
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("Transformers not available. Using fallback label generation.")


class AILabelGenerator:
    """AI 모델을 사용한 라벨 생성기"""
    
    MIN_LABEL_LENGTH = 3
    MAX_LABEL_LENGTH = 7
    AI_GENERATION_PARAMS = {
        "max_new_tokens": 15,
        "num_return_sequences": 1,
        "temperature": 0.7,
        "do_sample": True,
        "truncation": True
    }
    
    # 한글 키워드 매핑 (한국어 특화 모델 사용 안해서 일단 패턴매칭으로 진행)  # ToDo: 추후 디벨롭
    KOREAN_KEYWORDS = {
        # 계정/보안 관련 
        "account": "계정관리", "login": "로그인", "register": "회원가입", "profile": "프로필",
        "password": "비밀번호", "security": "보안설정", "verify": "본인인증", "auth": "인증",
        "user": "사용자", "member": "회원", "personal": "개인정보",
        
        # 금융 관련 
        "transfer": "이체", "payment": "결제", "loan": "대출", "investment": "투자",
        "savings": "적금", "deposit": "예금", "exchange": "환전", "card": "카드",
        "insurance": "보험", "pension": "연금", "fund": "펀드", "bank": "은행",
        "finance": "금융", "money": "자금", "cash": "현금", "credit": "신용",
        "debit": "직불", "withdraw": "출금", "deposit": "입금", "balance": "잔액",
        
        # 생활/편의 관련 
        "lifestyle": "생활", "utility": "공과금", "tax": "세금", "mobile": "휴대폰",
        "transport": "교통", "parking": "주차", "toll": "통행료", "convenience": "편의",
        "service": "서비스", "daily": "일상", "life": "생활", "convenient": "편의",
        
        # 건강/의료 관련 
        "health": "건강", "medical": "의료", "pharmacy": "약국", "fitness": "운동",
        "diet": "식단", "sleep": "수면", "wellness": "웰빙", "care": "케어",
        "hospital": "병원", "doctor": "의사", "medicine": "약품",
        
        # 쇼핑/상거래 관련 
        "shopping": "쇼핑", "mall": "쇼핑몰", "coupon": "쿠폰", "discount": "할인",
        "delivery": "배송", "refund": "환불", "review": "리뷰", "purchase": "구매",
        "buy": "구매", "sell": "판매", "market": "마켓", "store": "스토어",
        
        # 여행/레저 관련 
        "travel": "여행", "hotel": "호텔", "flight": "항공", "rental": "렌트",
        "tour": "투어", "booking": "예약", "reservation": "예약", "vacation": "휴가",
        "leisure": "레저", "trip": "여행", "journey": "여정",
        
        # 통신/기술 관련 
        "communication": "통신", "message": "메시지", "call": "통화", "sms": "문자",
        "email": "이메일", "chat": "채팅", "social": "소셜", "network": "네트워크",
        "internet": "인터넷", "online": "온라인", "digital": "디지털", "tech": "기술",
        
        # 교육/학습 관련 
        "education": "교육", "learning": "학습", "study": "공부", "course": "과정",
        "training": "훈련", "class": "수업", "lecture": "강의", "knowledge": "지식",
        
        # 엔터테인먼트 관련 
        "entertainment": "엔터", "game": "게임", "music": "음악", "video": "비디오",
        "movie": "영화", "book": "책", "news": "뉴스", "media": "미디어",
        "fun": "재미", "play": "놀이", "hobby": "취미",
        
        # 업무/비즈니스 관련 
        "business": "업무", "work": "작업", "office": "사무실", "project": "프로젝트",
        "meeting": "회의", "schedule": "일정", "task": "업무", "report": "보고서",
        "management": "관리", "planning": "계획", "organization": "조직",
        
        # 범용/공통 키워드 
        "main": "주요", "important": "중요", "essential": "필수", "basic": "기본",
        "advanced": "고급", "special": "특별", "premium": "프리미엄", "standard": "표준",
        "quick": "빠른", "easy": "쉬운", "simple": "간단", "smart": "스마트",
        "new": "새로운", "latest": "최신", "popular": "인기", "trending": "트렌드",
        "recommended": "추천", "favorite": "즐겨찾기", "recent": "최근", "history": "이력",
        "setting": "설정", "option": "옵션", "preference": "선호", "customize": "맞춤",
        "help": "도움", "support": "지원", "guide": "가이드", "tutorial": "튜토리얼",
        "info": "정보", "detail": "상세", "summary": "요약", "overview": "개요",
        "search": "검색", "find": "찾기", "browse": "둘러보기", "explore": "탐색",
        "notification": "알림", "alert": "경고", "reminder": "알림", "update": "업데이트"
    }
    
    # 서비스 클러스터별 fallback 라벨
    FALLBACK_LABELS = {
        "lifestyle": "생활서비스", "finance": "금융서비스", "account": "계정관리",
        "health": "건강관리", "shopping": "쇼핑서비스", "travel": "여행서비스",
        "security": "보안설정", "default": "추천기능"
    }
    
    _instance = None  # 싱글톤 패턴
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, model_name: str = "gpt2"):
        if hasattr(self, 'initialized'):
            return
        self.initialized = True
        
        self.model_name = model_name
        self.generator = None
        
        if TRANSFORMERS_AVAILABLE:
            self._initialize_model()
        else:
            logger.warning("Using fallback label generation due to missing dependencies.")
    
    def _initialize_model(self):
        """AI 모델 초기화"""
        try:
            logger.info(f"Loading AI model: {self.model_name}")
            self.generator = self._create_pipeline(self.model_name)
            logger.info("AI model loaded successfully!")
            
        except Exception as e:
            logger.error(f"Failed to load AI model: {e}")
            self._try_fallback_model()
    
    def _create_pipeline(self, model_name: str):
        """파이프라인 생성"""
        device = 0 if torch.cuda.is_available() else -1
        return pipeline("text-generation", model=model_name, device=device)
    
    def _try_fallback_model(self):
        """fallback 모델 시도"""
        try:
            logger.info("Trying fallback model...")
            self.generator = self._create_pipeline("gpt2")
            logger.info("Fallback model loaded successfully!")
        except Exception as e:
            logger.error(f"Fallback model also failed: {e}")
            logger.info("Will use rule-based label generation instead.")
            self.generator = None
    
    def generate_label(self, functions: List[Dict], cluster: str) -> str:
        """AI 모델을 사용해서 라벨 생성"""
        
        if self._should_use_ai(functions):
            try:
                logger.info(f"Attempting AI label generation for {len(functions)} functions")
                return self._generate_with_ai(functions, cluster)
            except Exception as e:
                logger.warning(f"AI generation failed: {e}, using fallback")
        
        logger.info("Using rule-based label generation")
        return self._generate_fallback(functions, cluster)
    
    def _should_use_ai(self, functions: List[Dict]) -> bool:
        """AI 모델 사용 여부 결정"""
        if not self.generator or len(functions) < 2:
            return False
        
        meaningful_count = sum(1 for func in functions 
                             if not self._is_function_id(func.get('function_id', '')))
        return meaningful_count >= 2
    
    def _is_function_id(self, func_id: str) -> bool:
        """기능 ID 패턴인지 확인"""
        return bool(re.match(r'^f\d+$', func_id))
    
    def _generate_with_ai(self, functions: List[Dict], cluster: str) -> str:
        """AI 모델로 라벨 생성"""
        function_names = [func.get('function_id', '') for func in functions]
        prompt = self._create_prompt(function_names, cluster)
        
        try:
            response = self.generator(prompt, **self.AI_GENERATION_PARAMS)
            generated_text = response[0]['generated_text']
            return self._extract_label_from_response(generated_text, prompt)
            
        except Exception as e:
            logger.error(f"AI generation error: {e}")
            return self._generate_fallback(functions, cluster)
    
    def _create_prompt(self, function_names: List[str], cluster: str) -> str:
        """AI 모델용 프롬프트 생성"""
        meaningful_names = [name for name in function_names if not self._is_function_id(name)]
        
        if not meaningful_names:
            return self._create_cluster_only_prompt(cluster)
        return self._create_function_based_prompt(meaningful_names, cluster)
    
    def _create_cluster_only_prompt(self, cluster: str) -> str:
        """서비스 클러스터만 사용하는 프롬프트"""
        return f"""다음 서비스 영역에 대한 자연스러운 한글 제목을 생성해주세요.

서비스 영역: {cluster}

제목은 다음 조건을 만족해야 합니다:
1. {self.MIN_LABEL_LENGTH}-{self.MAX_LABEL_LENGTH}글자로 적당한 길이로
2. 사용자가 직관적으로 이해할 수 있게
3. '~관리', '~서비스' 같은 형식적 표현 피하기
4. 자연스러운 한국어로
5. 기능 ID나 영어는 절대 포함하지 말고 한글로만 작성

제목:"""
    
    def _create_function_based_prompt(self, meaningful_names: List[str], cluster: str) -> str:
        """기능 기반 프롬프트"""
        return f"""다음 기능들을 포함하는 그룹의 자연스러운 한글 제목을 생성해주세요.

기능들: {', '.join(meaningful_names)}
서비스 영역: {cluster}

제목은 다음 조건을 만족해야 합니다:
1. {self.MIN_LABEL_LENGTH}-{self.MAX_LABEL_LENGTH}글자로 적당한 길이로
2. 사용자가 직관적으로 이해할 수 있게
3. '~관리', '~서비스' 같은 형식적 표현 피하기
4. 자연스러운 한국어로
5. 기능 ID나 영어는 절대 포함하지 말고 한글로만 작성

제목:"""
    
    def _extract_label_from_response(self, response: str, prompt: str) -> str:
        """AI 응답에서 라벨 추출"""
        # 프롬프트 제거 및 기본 정리
        label = self._clean_response(response, prompt)
        
        # 유효성 검증 및 길이 조정
        label = self._validate_and_adjust_label(label)
        
        return label
    
    def _clean_response(self, response: str, prompt: str) -> str:
        """응답 텍스트 정리"""
        label = response.replace(prompt, "").strip()
        label = label.split('\n')[0].strip()
        label = label.replace('"', '').replace("'", "")
        
        # 영어 기능 ID 패터 제거
        label = re.sub(r'f\d+', '', label)
        label = re.sub(r'[,\s]+', ' ', label).strip()
        
        return label
    
    def _validate_and_adjust_label(self, label: str) -> str:
        """라벨 유효성 검증 및 조정"""
        # 의미없는 텍스트 필터링
        if (re.match(r'^[a-zA-Z\s]+$', label) or 
            re.match(r'^[\d\s]+$', label) or 
            not re.search(r'[가-힣]', label)):
            return "추천 기능"
        
        # 길이 조정
        if len(label) < self.MIN_LABEL_LENGTH:
            return "추천 기능"
        elif len(label) > self.MAX_LABEL_LENGTH:
            return label[:self.MAX_LABEL_LENGTH]
        
        return label
    
    def _generate_fallback(self, functions: List[Dict], cluster: str) -> str:
        """AI 생성 실패 시 fallback 라벨 생성"""
        if not functions:
            return self.FALLBACK_LABELS.get(cluster, self.FALLBACK_LABELS["default"])
        
        # 기능명에서 한글 키워드 찾기
        keyword = self._find_best_keyword(functions)
        if keyword:
            return keyword
        
        # 서비스 클러스터 기반 fallback
        return self.FALLBACK_LABELS.get(cluster, self.FALLBACK_LABELS["default"])
    
    def _find_best_keyword(self, functions: List[Dict]) -> Optional[str]:
        """가장 적합한 키워드 찾기"""
        function_names = [func.get('function_id', '') for func in functions]
        found_keywords = []
        
        for name in function_names:
            name_lower = name.lower()
            for eng, kor in self.KOREAN_KEYWORDS.items():
                if eng in name_lower:
                    found_keywords.append(kor)
                    break
        
        if found_keywords:
            keyword_counts = Counter(found_keywords)
            return keyword_counts.most_common(1)[0][0]
        
        return None


def create_ai_label_generator() -> AILabelGenerator:
    """AI 라벨 생성기 인스턴스 생성 (싱글톤)"""
    return AILabelGenerator()
