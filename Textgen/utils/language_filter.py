class LanguageTextFilter:
    def __init__(self, language):
        self.language = language
        self.unicode_ranges = {
            'vi': [(0x0001, 0x007F), (0x0100, 0x02AF), (0x1E00, 0x1EFF)],
            'th': [(0x0E00, 0x0E7F)],
            'ja': [(0x3040, 0x309F), (0x30A0, 0x30FF), (0x4E00, 0x9FFF)],
            'zh': [(0x4E00, 0x9FFF)]
        }

    def is_valid_char(self, char):
        char_code = ord(char)
        ranges = self.unicode_ranges.get(self.language, [])
        return any(start <= char_code <= end for start, end in ranges)

    def filter_text(self, text):
        return ''.join(char for char in text if self.is_valid_char(char))