import sys
import io
import platform
from pathlib import Path

import pytesseract
import fitz  # PyMuPDF
import cv2
import numpy as np
from PIL import Image


# ===============================
# TESSERACT CONFIG
# ===============================
class TesseractConfig:
    @staticmethod
    def setup():
        if platform.system() == "Windows":
            pytesseract.pytesseract.tesseract_cmd = (
                r"C:\Program Files\Tesseract-OCR\tesseract.exe"
            )
            
# ===============================
# IMAGE PREPROCESSOR
# ===============================
class ImagePreprocessor:
    @staticmethod
    def preprocess(image: Image.Image) -> np.ndarray:
        img = np.array(image)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        h, w = gray.shape
        if w < 1000:
            scale = 1000 / w
            gray = cv2.resize(
                gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC
            )

        gray = cv2.bilateralFilter(gray, 9, 75, 75)

        clahe = cv2.createCLAHE(2.0, (8, 8))
        gray = clahe.apply(gray)

        return gray


# ===============================
# OCR ENGINE
# ===============================
class OCREngine:
    def _init_(self, lang="eng"):
        self.lang = lang
        self.configs = [
            "--oem 3 --psm 6",
            "--oem 3 --psm 3",
            "--oem 3 --psm 4",
            "--oem 3 --psm 11",
        ]

    def extract_best_text(self, image: np.ndarray) -> str:
        results = []
        for cfg in self.configs:
            text = pytesseract.image_to_string(
                image, lang=self.lang, config=cfg
            )
            if text.strip():
                results.append(text)

        return max(results, key=len) if results else ""


# ===============================
# IMAGE TEXT EXTRACTOR
# ===============================
class ImageTextExtractor:
    def _init_(self, ocr_engine: OCREngine):
        self.ocr = ocr_engine

    def extract(self, path: str) -> str:
        image = Image.open(path).convert("RGB")
        processed = ImagePreprocessor.preprocess(image)
        return self.ocr.extract_best_text(processed)


# ===============================
# PDF TEXT EXTRACTOR
# ===============================
class PDFTextExtractor:
    def _init_(self, ocr_engine: OCREngine):
        self.ocr = ocr_engine

    @staticmethod
    def _extract_images_from_page(page):
        images = []
        for img in page.get_images(full=True):
            xref = img[0]
            base = page.parent.extract_image(xref)
            image = Image.open(io.BytesIO(base["image"])).convert("RGB")
            images.append(image)
        return images

    def extract(self, path: str) -> str:
        doc = fitz.open(path)
        output = ""

        for i, page in enumerate(doc, start=1):
            page_text = ""

            text = page.get_text()
            if text.strip():
                page_text += text + "\n"

            for img in self._extract_images_from_page(page):
                processed = ImagePreprocessor.preprocess(img)
                ocr_text = self.ocr.extract_best_text(processed)
                if ocr_text.strip():
                    page_text += ocr_text + "\n"

            if page_text.strip():
                output += f"\n--- Page {i} ---\n{page_text}"

        doc.close()
        return output.strip()


# ===============================
# MAIN FILE EXTRACTOR (ROUTER)
# ===============================
class FileTextExtractor:
    IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}

    def _init_(self):
        TesseractConfig.setup()
        self.ocr_engine = OCREngine()
        self.image_extractor = ImageTextExtractor(self.ocr_engine)
        self.pdf_extractor = PDFTextExtractor(self.ocr_engine)

    def extract_text(self, file_path: str) -> str:
        ext = Path(file_path).suffix.lower()

        if ext in self.IMAGE_EXTENSIONS:
            return self.image_extractor.extract(file_path)

        if ext == ".pdf":
            return self.pdf_extractor.extract(file_path)

        raise ValueError("Unsupported file type")