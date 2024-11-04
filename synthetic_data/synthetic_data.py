import treescope
treescope.basic_interactive_setup()
from trdg.generators import GeneratorFromStrings
import matplotlib.pyplot as plt


# TextRecognitionDataGenerator 패키지는 여러 방법으로 텍스트 이미지를 생성할 수 있습니다.
# 아래에서는 미리 저장해놓은 텍스트 파일을 이용하여 이미지를 생성하는데요,
# 이 외에도 wikipedia 검색, 랜덤 텍스트 생성 등 다양한 방법으로 텍스트 이미지를 생성할 수 있습니다.

def get_words(count=128):
    with open("korean.txt", "r") as f:
        paragrphs = f.readlines()

    paragrphs = " ".join([x.strip() for x in paragrphs])
    sentences = [x.strip() for x in paragrphs.split(" ")]
    sentences = [x for x in sentences if x]
    generator = GeneratorFromStrings(
        sentences,
        language="ko",
        size=64,
        count=count,
        fonts=["fonts/NanumDdarEGeEomMaGa.ttf"],  # <- 언어에 따라 폰트를 확보해 지정해주어야 합니다.
        margins=(5, 5, 5, 5),
    )
    words = []
    for _ , (patch, text) in enumerate(generator):
        words.append({"patch": patch, "text": text, "size": patch.size, "margin": 5})
    return words

words = get_words(24)


# 본 블록을 실행하면 Jupyter Notebook에서 Treescope가 변수 출력을 담당하게 됩니다.
import treescope
treescope.basic_interactive_setup()

# make_document 함수는 입력받은 단어 이미지를 이어붙여 문서처럼 보이는 이미지를 만들고, 이를 words 객체와 같이 출력합니다.
# 이를 지금부터 Document 타입의 객체라 불러보겠습니다.

from synth_utils import make_document, Document
document = make_document(words)
document