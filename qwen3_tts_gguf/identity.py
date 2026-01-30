"""
identity.py - 音色锚点管理 (五要素版)
用于保存和加载克隆或固定音色所需的要素：
1. text (str)
2. text_ids (List[int])
3. spk_emb (np.ndarray, 2048)
4. codes (np.ndarray, T, 16)
5. summed_embeds (np.ndarray, T, 2048)
"""
import json
import os
import numpy as np
from typing import List, Optional
from . import logger

class VoiceIdentity:
    def __init__(self):
        self.text: Optional[str] = None
        self.text_ids: Optional[List[int]] = None
        self.spk_emb: Optional[np.ndarray] = None
        self.codes: Optional[np.ndarray] = None
        self.summed_embeds: Optional[np.ndarray] = None

    @property
    def is_set(self) -> bool:
        """检查锚点是否已设置"""
        return all(x is not None for x in [self.text_ids, self.spk_emb, self.codes, self.summed_embeds])

    def reset(self):
        """重置所有身份数据"""
        self.text = None
        self.text_ids = None
        self.spk_emb = None
        self.codes = None
        self.summed_embeds = None
        logger.info("🧹 VoiceIdentity data reset.")

    def set_identity(self, text: str, text_ids: List[int], spk_emb: np.ndarray, codes: np.ndarray, summed_embeds: np.ndarray):
        """手动设置锚点属性"""
        self.text = text
        self.text_ids = text_ids
        self.spk_emb = spk_emb
        self.codes = codes
        self.summed_embeds = summed_embeds
        logger.info(f"✅ Identity set. Text: '{text}', Audio: {codes.shape[0]} frames.")

    def save_identity(self, path: str):
        """保存到 JSON 文件"""
        if not self.is_set:
            logger.warning("⚠️ Identity not set, nothing to save.")
            return

        data = {
            "text": self.text,
            "text_ids": self.text_ids,
            "spk_emb": self.spk_emb.tolist(),
            "codes": self.codes.tolist(),
            "summed_embeds": self.summed_embeds.tolist()
        }
        
        # 确保目录存在
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"💾 Identity saved to {path}")

    def read_identity(self, path: str):
        """从 JSON 文件读取"""
        if not os.path.exists(path):
            logger.error(f"❌ Identity file not found: {path}")
            return

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        self.text = data.get("text", "")
        self.text_ids = data["text_ids"]
        self.spk_emb = np.array(data["spk_emb"], dtype=np.float32)
        self.codes = np.array(data["codes"], dtype=np.int64)
        self.summed_embeds = np.array(data["summed_embeds"], dtype=np.float32)
        
        logger.info(f"📂 Identity loaded from {path}. Text: '{self.text}', Audio: {self.codes.shape[0]} frames.")
