from dataclasses import dataclass
import os

@dataclass
class FontConfig:
    FONT_DIR = "./fonts"
    
    LANGUAGE_FONTS = {
        'vi': ['BeVietnamPro-LightItalic.ttf'],
        'th': ['NotoSansThai-VariableFont_wdth,wght.ttf'],
        'ja': ['NotoSansJP-VariableFont_wght.ttf'],
        'zh': ['NotoSansSC-VariableFont_wght.ttf'],
        'ko': ['NanumDdarEGeEomMaGa.ttf']
    }
    
    @classmethod
    def get_font_path(cls, language):
        if language not in cls.LANGUAGE_FONTS:
            raise ValueError(f"Unsupported language: {language}")
            
        font_file = cls.LANGUAGE_FONTS[language][0]
        font_path = os.path.join(cls.FONT_DIR, font_file)
        
        if not os.path.exists(font_path):
            raise FileNotFoundError(f"Font file not found: {font_path}")
            
        return font_path