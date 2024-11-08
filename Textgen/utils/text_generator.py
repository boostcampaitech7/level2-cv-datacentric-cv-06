from trdg.generators import GeneratorFromStrings
from trdg.generators import GeneratorFromWikipedia
from .language_filter import LanguageTextFilter
from .config import FontConfig
from tqdm.auto import tqdm
import random
from PIL import Image

class TextGenerator:
    def __init__(self, language: str, count: int = 128, font_size: int = 64):
        self.language = language
        self.count = count
        self.font_size = font_size
        self.font_path = FontConfig.get_font_path(language)
        self.text_filter = LanguageTextFilter(language)
        # 띄어쓰기 없는 언어 정의
        self.no_space_languages = {'ja', 'zh', 'th'}
        # 언어별 문자 단위 길이 설정
        self.char_length_limits = {
            'ja': 15,  # 일본어는 한 줄에 약 15자
            'zh': 15,  # 중국어는 한 줄에 약 15자
            'th': 20,  # 태국어는 한 줄에 약 20자
            'vi': 40   # 베트남어는 띄어쓰기가 있으므로 더 긴 길이 허용
        }

    def split_text_by_length(self, text: str) -> list:
        """띄어쓰기 없는 언어의 텍스트를 적절한 길이로 분할"""
        if self.language not in self.no_space_languages:
            return text.split()
            
        char_limit = self.char_length_limits.get(self.language, 15)
        result = []
        current_text = ""
        
        for char in text:
            current_text += char
            if len(current_text) >= char_limit:
                if current_text.strip():
                    result.append(current_text.strip())
                current_text = ""
        
        if current_text.strip():
            result.append(current_text.strip())
            
        return result

    def generate_word_images(self, texts, margin: int = 5):
        """텍스트를 적절한 길이로 나누어 이미지 생성"""
        # 텍스트 처리
        processed_texts = []
        for text in texts:
            if self.language in self.no_space_languages:
                # 띄어쓰기 없는 언어는 문자 수 기준으로 분할
                chunks = self.split_text_by_length(text)
                processed_texts.extend(chunks)
            else:
                # 띄어쓰기 있는 언어는 2~5개 단어로 구성된 문장으로 분할
                words = text.split()
                while len(words) >= 2:
                    chunk_size = random.randint(2, min(5, len(words)))
                    processed_texts.append(" ".join(words[:chunk_size]))
                    words = words[chunk_size:]
                if words:
                    processed_texts.append(" ".join(words))

        # 이미지 생성
        sentence_generator = GeneratorFromStrings(
            processed_texts,
            language=self.language,
            fonts=[self.font_path],
            size=32,
            margins=(margin, margin, margin, margin),
        )
        
        sentence_images = []
        sentence_progress = tqdm(
            range(len(processed_texts)),
            desc="문장 이미지 생성",
            position=2,
            leave=False
        )
        
        for idx in sentence_progress:
            try:
                image, text = next(sentence_generator)
                # 최소 크기 확인 및 조정
                min_height = 32
                if image.size[1] < min_height:
                    ratio = min_height / image.size[1]
                    new_size = (int(image.size[0] * ratio), min_height)
                    image = image.resize(new_size, Image.LANCZOS)
                
                # 최대 너비 제한
                max_width = 600  # 최대 너비 설정
                if image.size[0] > max_width:
                    ratio = max_width / image.size[0]
                    new_size = (max_width, int(image.size[1] * ratio))
                    image = image.resize(new_size, Image.LANCZOS)
                
                sentence_images.append({
                    "patch": image,
                    "text": text,
                    "size": image.size,
                    "margin": margin,
                    "bbox": image.getbbox()
                })
                
                sentence_progress.set_postfix({
                    'text': text[:20] + "..." if len(text) > 20 else text,
                    'size': image.size
                })
                
            except StopIteration:
                break
        
        sentence_progress.close()
        return sentence_images

    def generate_from_wikipedia(self):
        """Wikipedia에서 텍스트 생성"""
        wiki_generator = GeneratorFromWikipedia(
            language=self.language,
            count=self.count,
            fonts=[self.font_path],
            size=self.font_size
        )
        
        filtered_texts = []
        text_progress = tqdm(
            range(self.count),
            desc=f"텍스트 생성 ({self.language})",
            position=2,
            leave=False
        )
        
        for _ in text_progress:
            try:
                _, text = next(wiki_generator)
                filtered_text = self.text_filter.filter_text(text)
                if filtered_text.strip():
                    filtered_texts.append(filtered_text)
                    text_progress.set_postfix({
                        'text_length': len(filtered_text)
                    })
            except StopIteration:
                break
        
        text_progress.close()
        return filtered_texts