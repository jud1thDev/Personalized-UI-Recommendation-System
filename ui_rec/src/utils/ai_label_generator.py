"""
AI 기반 자연스러운 라벨 생성 모듈
Hugging Face 모델을 사용해서 기능 그룹의 제목을 자동 생성
"""

import os
from typing import List, Dict, Optional
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("Transformers not available. Using fallback label generation.")


class AILabelGenerator:
    """AI 모델을 사용한 라벨 생성기"""
    
    _instance = None  # 싱글톤 패턴
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, model_name: str = "gpt2"):
        # 싱글톤 패턴으로 중복 초기화 방지
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
            
            # 파이프라인 생성 시 최소한의 파라미터만 사용
            self.generator = pipeline(
                "text-generation",
                model=self.model_name,
                device=0 if torch.cuda.is_available() else -1
            )
            
            logger.info("AI model loaded successfully!")
            
        except Exception as e:
            logger.error(f"Failed to load AI model: {e}")
            try:
                logger.info("Trying fallback model...")
                self.generator = pipeline(
                    "text-generation",
                    model="gpt2",  
                    device=0 if torch.cuda.is_available() else -1
                )
                logger.info("Fallback model loaded successfully!")
            except Exception as e2:
                logger.error(f"Fallback model also failed: {e2}")
                logger.info("Will use rule-based label generation instead.")
                self.generator = None
    
    def generate_label(self, functions: List[Dict], cluster: str) -> str:
        """AI 모델을 사용해서 라벨 생성"""
        
        if self.generator and self._should_use_ai(functions):
            try:
                logger.info(f"Attempting AI label generation for {len(functions)} functions")
                return self._generate_with_ai(functions, cluster)
            except Exception as e:
                logger.warning(f"AI generation failed: {e}, using fallback")
                logger.info("Falling back to rule-based label generation")
        
        # AI 생성 실패 시 fallback 사용
        logger.info("Using rule-based label generation")
        return self._generate_fallback(functions, cluster)
    
    def _should_use_ai(self, functions: List[Dict]) -> bool:
        """AI 모델 사용 여부 결정"""
        # 기능이 너무 적으면 AI 사용하지 않음
        return len(functions) >= 2
    
    def _generate_with_ai(self, functions: List[Dict], cluster: str) -> str:
        """AI 모델로 라벨 생성"""
        
        # 기능 정보 추출
        function_names = [func.get('function_id', '') for func in functions]
        service_cluster = cluster
        
        # 프롬프트 구성
        prompt = self._create_prompt(function_names, service_cluster)
        
        # AI 모델로 생성 - 최소한의 파라미터만 사용
        try:
            response = self.generator(
                prompt,
                max_new_tokens=15, 
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                truncation=True,
                pad_token_id=self.generator.tokenizer.eos_token_id
            )
            
            # 응답에서 라벨 추출
            generated_text = response[0]['generated_text']
            label = self._extract_label_from_response(generated_text, prompt)
            
            return label
            
        except Exception as e:
            logger.error(f"AI generation error: {e}")
            return self._generate_fallback(functions, cluster)
    
    def _create_prompt(self, function_names: List[str], cluster: str) -> str:
        """AI 모델용 프롬프트 생성"""
        
        prompt = f"""다음 기능들을 포함하는 그룹의 자연스러운 한글 제목을 생성해주세요.

기능들: {', '.join(function_names)}
서비스 영역: {cluster}

제목은 다음 조건을 만족해야 합니다:
1. 3-7글자로 적당한 길이로
2. 사용자가 직관적으로 이해할 수 있게
3. '~관리', '~서비스' 같은 형식적 표현 피하기
4. 자연스러운 한국어로

제목:"""
        
        return prompt
    
    def _extract_label_from_response(self, response: str, prompt: str) -> str:
        """AI 응답에서 라벨 추출"""
        
        # 프롬프트 부분 제거
        label = response.replace(prompt, "").strip()
        
        # 첫 번째 줄만 사용
        label = label.split('\n')[0].strip()
        
        # 특수문자 제거
        label = label.replace('"', '').replace("'", "")
        
        # 길이 제한 (3-7글자)
        if len(label) < 3:
            label = "추천 기능"
        elif len(label) > 7:
            label = label[:7]
        
        return label if label else "추천 기능"
    
    def _generate_fallback(self, functions: List[Dict], cluster: str) -> str:
        """AI 생성 실패 시 fallback 라벨 생성"""
        
        # 기능명에서 한글 키워드 추출하여 라벨 생성
        if functions:
            # 기능명에서 의미있는 한글 키워드 찾기
            function_names = [func.get('function_id', '') for func in functions]
            
            # 한글 키워드 매핑 (KB스타뱅킹 앱 기준)
            korean_keywords = {
                # 계정 관련
                "account": "계정 관리", "login": "로그인", "register": "회원가입", "profile": "프로필",
                "password": "비밀번호", "security": "보안 설정", "verify": "본인인증",
                
                # 금융 관련  
                "transfer": "계좌이체", "payment": "결제하기", "loan": "대출신청", "investment": "투자상품",
                "savings": "적금상품", "deposit": "예금상품", "exchange": "환전하기", "card": "카드서비스",
                "insurance": "보험상품", "pension": "연금상품", "fund": "펀드상품",
                
                # 생활 관련
                "lifestyle": "생활서비스", "utility": "공과금납부", "tax": "세금납부", "mobile": "휴대폰결제",
                "transport": "교통카드", "parking": "주차서비스", "toll": "통행료납부",
                
                # 건강 관련
                "health": "건강관리", "medical": "의료서비스", "pharmacy": "약국서비스", "fitness": "운동관리",
                "diet": "식단관리", "sleep": "수면관리",
                
                # 쇼핑 관련
                "shopping": "쇼핑몰", "mall": "쇼핑센터", "coupon": "쿠폰서비스", "discount": "할인혜택",
                "delivery": "배송조회", "refund": "환불신청", "review": "리뷰작성",
                
                # 여행 관련
                "travel": "여행서비스", "hotel": "호텔예약", "flight": "항공예약", "rental": "렌트서비스",
                "tour": "투어상품", "booking": "예약서비스", "reservation": "예약관리",
                
                # 기타
                "notification": "알림설정", "setting": "설정메뉴", "help": "도움말", "info": "정보안내",
                "search": "검색서비스", "favorite": "즐겨찾기", "history": "이용내역"
            }
            
            # 기능명에서 한글 키워드 찾기
            found_keywords = []
            for name in function_names:
                name_lower = name.lower()
                for eng, kor in korean_keywords.items():
                    if eng in name_lower:
                        found_keywords.append(kor)
                        break
            
            if found_keywords:
                # 가장 많이 나타나는 키워드 선택
                from collections import Counter
                keyword_counts = Counter(found_keywords)
                most_common = keyword_counts.most_common(1)[0][0]
                return most_common
        
        # 서비스 클러스터 기반 fallback 라벨 (KB스타뱅킹 앱 기준)
        fallback_labels = {
            "lifestyle": "생활서비스",
            "finance": "금융서비스", 
            "account": "계정관리",
            "health": "건강관리",
            "shopping": "쇼핑서비스",
            "travel": "여행서비스",
            "security": "보안설정",
            "default": "추천기능"
        }
        
        return fallback_labels.get(cluster, fallback_labels["default"])


def create_ai_label_generator() -> AILabelGenerator:
    """AI 라벨 생성기 인스턴스 생성 (싱글톤)"""
    return AILabelGenerator()
