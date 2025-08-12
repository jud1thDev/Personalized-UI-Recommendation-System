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
    """AI 모델을 사용한 자연스러운 라벨 생성기"""
    
    def __init__(self, model_name: str = "beomi/KcELECTRA-base"):
        self.model_name = model_name
        self.generator = None
        self.fallback_generator = None
        
        if TRANSFORMERS_AVAILABLE:
            self._initialize_model()
        else:
            logger.warning("Using fallback label generation due to missing dependencies.")
    
    def _initialize_model(self):
        """AI 모델 초기화"""
        try:
            logger.info(f"Loading AI model: {self.model_name}")
            
            # 한국어에 특화된 모델 사용
            self.generator = pipeline(
                "text-generation",
                model=self.model_name,
                device=0 if torch.cuda.is_available() else -1
            )
            
            logger.info("AI model loaded successfully!")
            
        except Exception as e:
            logger.error(f"Failed to load AI model: {e}")
            self.generator = None
    
    def generate_label(self, functions: List[Dict], cluster: str) -> str:
        """AI 모델을 사용해서 자연스러운 라벨 생성"""
        
        if self.generator and self._should_use_ai(functions):
            try:
                return self._generate_with_ai(functions, cluster)
            except Exception as e:
                logger.warning(f"AI generation failed: {e}, using fallback")
        
        # AI 생성 실패 시 fallback 사용
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
        
        # AI 모델로 생성
        response = self.generator(
            prompt,
            max_length=50,
            num_return_sequences=1,
            temperature=0.7,
            do_sample=True
        )
        
        # 응답에서 라벨 추출
        generated_text = response[0]['generated_text']
        label = self._extract_label_from_response(generated_text, prompt)
        
        return label
    
    def _create_prompt(self, function_names: List[str], cluster: str) -> str:
        """AI 모델용 프롬프트 생성"""
        
        prompt = f"""다음 기능들을 포함하는 그룹의 자연스러운 한글 제목을 생성해주세요.

기능들: {', '.join(function_names)}
서비스 영역: {cluster}

제목은 다음 조건을 만족해야 합니다:
1. 2-6글자로 간결하게
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
        
        # 길이 제한
        if len(label) > 10:
            label = label[:10]
        
        return label if label else "추천 기능"
    
    def _generate_fallback(self, functions: List[Dict], cluster: str) -> str:
        """AI 생성 실패 시 fallback 라벨 생성"""
        
        # 간단한 rule-based 라벨 생성
        fallback_labels = {
            "account": ["계정", "내 정보", "프로필"],
            "finance": ["금융", "자산", "내 돈"],
            "lifestyle": ["생활", "편의", "일상"],
            "health": ["건강", "웰빙", "건강관리"],
            "shopping": ["쇼핑", "구매", "상품"],
            "travel": ["여행", "교통", "이동"],
            "security": ["보안", "안전", "보호"]
        }
        
        labels = fallback_labels.get(cluster, [cluster])
        return labels[0] if labels else cluster


def create_ai_label_generator() -> AILabelGenerator:
    """AI 라벨 생성기 인스턴스 생성"""
    return AILabelGenerator()


# 사용 예시
if __name__ == "__main__":
    generator = create_ai_label_generator()
    
    # 테스트
    test_functions = [
        {"function_id": "f001", "service_cluster": "account"},
        {"function_id": "f002", "service_cluster": "account"}
    ]
    
    label = generator.generate_label(test_functions, "account")
    print(f"Generated label: {label}")
