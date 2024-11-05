import cv2
import numpy as np
from PIL import Image
from text_generator import TextGenerator
from receipt_layout import ReceiptLayoutGenerator
import random
from synth_utils import *

text_generator = TextGenerator('vi', count=1)
layout_generator = ReceiptLayoutGenerator(2048, 2048)
texts = text_generator.generate_from_wikipedia()
word_images = text_generator.generate_word_images(texts)
receipt_image, placed_words = layout_generator.create_layout(word_images)

#print(placed_words)
doc = make_document(placed_words)
print(doc)
