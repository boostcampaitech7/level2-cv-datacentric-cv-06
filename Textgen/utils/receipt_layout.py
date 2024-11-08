import random
from PIL import Image, ImageDraw

class ReceiptLayoutGenerator:
    def __init__(self, width: int = 800, height: int = 1200):
        self.width = width
        self.height = height
        self.margin = 40  # 좌우 여백
        self.line_spacing = 15  # 줄 간격
        self.sections = {
            'header': {'ratio': 0.15, 'align': 'center'},
            'items': {'ratio': 0.7, 'align': 'left'},
            'footer': {'ratio': 0.15, 'align': 'center'}
        }

    def create_layout(self, word_images):
        """영수증 스타일 레이아웃 생성"""
        receipt_image = Image.new('RGB', (self.width, self.height), 'white')
        draw = ImageDraw.Draw(receipt_image)
        
        # 영수증 테두리 및 구분선 그리기
        self._draw_receipt_lines(draw)
        
        current_y = self.margin
        placed_words = []
        remaining_words = word_images.copy()
        
        while remaining_words and current_y < self.height - self.margin:
            # 한 줄에 들어갈 단어 수 랜덤 선택 (1~4개)
            words_per_line = random.randint(1, 4)
            line_words = self._get_line_words(remaining_words, words_per_line)
            
            if not line_words:
                break
                
            # 줄 높이 계산 (가장 큰 단어 높이 + 여백)
            line_height = max(word['patch'].size[1] for word in line_words)
            
            # 다음 줄이 페이지를 벗어나는지 확인
            if current_y + line_height > self.height - self.margin:
                break
            
            # 줄 단위로 단어 배치
            line_placed_words = self._place_line_words(
                receipt_image, line_words, current_y, line_height)
            placed_words.extend(line_placed_words)
            
            current_y += line_height + self.line_spacing
            
            # 남은 단어 목록에서 사용한 단어 제거
            for word in line_words:
                if word in remaining_words:
                    remaining_words.remove(word)
        
        return receipt_image, placed_words

    def _draw_receipt_lines(self, draw):
        """영수증 테두리와 구분선 그리기"""
        # 점선 패턴 정의
        dash_pattern = [8, 4]  # 8픽셀 선, 4픽셀 공백
        
        # 테두리 그리기
        for i in range(0, self.height, sum(dash_pattern)):
            draw.line([(self.margin//2, i), (self.margin//2, min(i + dash_pattern[0], self.height))],
                     fill='gray', width=1)
            draw.line([(self.width - self.margin//2, i), 
                      (self.width - self.margin//2, min(i + dash_pattern[0], self.height))],
                     fill='gray', width=1)
        
        # 섹션 구분선 그리기 (점선)
        header_y = int(self.height * 0.15)
        footer_y = int(self.height * 0.85)
        
        for x in range(self.margin, self.width - self.margin, sum(dash_pattern)):
            draw.line([(x, header_y), (min(x + dash_pattern[0], self.width - self.margin), header_y)],
                     fill='gray', width=1)
            draw.line([(x, footer_y), (min(x + dash_pattern[0], self.width - self.margin), footer_y)],
                     fill='gray', width=1)

    def _get_line_words(self, sentences, max_sentences):
        """한 줄에 들어갈 문장들 선택"""
        if not sentences:
            return []
            
        line_sentences = []
        total_width = 0
        available_width = self.width - (self.margin * 2)
        min_font_size = 24  # 최소 폰트 크기
        
        for sentence in sentences[:max_sentences]:
            word_width = sentence['patch'].size[0]
            word_height = sentence['patch'].size[1]
            
            # 문장이 너무 큰 경우 크기 조절
            if word_width > available_width * 0.9:
                ratio = (available_width * 0.9) / word_width
                # 최소 폰트 크기 제한
                if word_height * ratio >= min_font_size:
                    new_size = (int(word_width * ratio), int(word_height * ratio))
                    sentence['patch'] = sentence['patch'].resize(new_size, Image.LANCZOS)
                    word_width = new_size[0]
                else:
                    continue
            
            if total_width + word_width + (len(line_sentences) * 20) <= available_width:
                line_sentences.append(sentence)
                total_width += word_width
            else:
                break
                
        return line_sentences

    def _place_line_words(self, image, sentences, y, line_height):
        """한 줄의 문장들 배치"""
        total_width = sum(sentence['patch'].size[0] for sentence in sentences)
        spacing = 20  # 문장 간 간격
        total_spacing = spacing * (len(sentences) - 1)
        
        # 시작 x 좌표 계산 (중앙 정렬)
        start_x = (self.width - (total_width + total_spacing)) // 2
        current_x = start_x
        
        placed_sentences = []
        for sentence in sentences:
            w, h = sentence['patch'].size
            # 세로 중앙 정렬을 위한 y 오프셋 계산
            y_offset = (line_height - h) // 2
            
            try:
                image.paste(sentence['patch'], (current_x, y + y_offset))
                
                placed_sentences.append({
                    'text': sentence['text'],
                    'bbox': [current_x, y + y_offset,
                            current_x + w, y + y_offset,
                            current_x + w, y + y_offset + h,
                            current_x, y + y_offset + h]
                })
                
                current_x += w + spacing
                
            except Exception as e:
                print(f"Error placing sentence: {sentence['text'][:20]}..., Error: {str(e)}")
                continue
                
        return placed_sentences