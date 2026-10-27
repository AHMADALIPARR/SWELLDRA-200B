from dataclasses import dataclass
from typing import List, Dict
import torch

@dataclass
class DocumentBlock:
    text: str
    bbox: List[float]
    block_type: str
    confidence: float
    reading_order: int

@dataclass
class DocumentTable:
    headers: List[str]
    rows: List[List[str]]
    bbox: List[float]
    markdown: str

@dataclass
class StructuredDocument:
    blocks: List[DocumentBlock]
    tables: List[DocumentTable]
    full_text: str
    markdown: str
    metadata: Dict

class OCRDocumentEngine:
    def __init__(self, vision_encoder="ViT-L-14-336"):
        self.vision_encoder = vision_encoder
    def ingest(self, image_path_or_pdf: str):
        raise NotImplementedError("Load ViT-L-14 checkpoint")
    @staticmethod
    def test_interface():
        doc = StructuredDocument(
            blocks=[DocumentBlock("Hello world", [0,0,100,20], "text", 0.99, 0)],
            tables=[], full_text="Hello world", markdown="Hello world", metadata={"pages":1}
        )
        assert len(doc.blocks) == 1
        print("OCR interface test passed")
