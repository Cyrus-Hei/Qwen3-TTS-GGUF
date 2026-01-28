import torch
import torch.nn as nn
from . import logger

# 引入本地模型定义
try:
    from .tokenizer_12hz.modeling_tokenizer import Qwen3TTSTokenizerV2Model
except ImportError:
    Qwen3TTSTokenizerV2Model = None
    logger.warning("Could not import Qwen3TTSTokenizerV2Model from .modeling_tokenizer")

class CodecEncoderExportWrapper(nn.Module):
    """
    Qwen3-TTS 音频编码器导出包装类。
    
    【核心修复 V3】
    针对 JIT Trace 报 `unordered_map` 的问题，这通常是因为 Transformers 的子模块
    （MimiEncoder 或 Quantizer）内部返回了 ModelOutput 对象（类似字典）。
    
    本类采取以下策略：
    1. 强制覆盖 config.return_dict = False。
    2. 对于 Quantizer 这种可能不遵守 config 的组件，尝试通过属性访问手动剥离对象。
    """
    def __init__(self, model: Qwen3TTSTokenizerV2Model):
        super().__init__()
        self.config = model.config
        
        # 1. 强力覆写配置
        self.config.return_dict = False
        self.config.use_cache = False
        
        # 2. 提取核心模块 (MimiModel 内部组件)
        self.mimi_model = model.encoder
        self.cnn_encoder = self.mimi_model.encoder 
        self.transformer = self.mimi_model.encoder_transformer
        self.downsample = self.mimi_model.downsample
        self.quantizer = self.mimi_model.quantizer
        
        # 3. 锁定为评估模式，并递归禁用 return_dict/use_cache
        self.eval()
        for m in self.modules():
            if hasattr(m, 'config'):
                m.config.return_dict = False
                m.config.use_cache = False

    def forward(self, input_values):
        """
        Args:
            input_values (torch.FloatTensor): Shape [Batch, Time]
        Returns:
            audio_codes (torch.LongTensor): Shape [Batch, Time, Q]
        """
        # 1. Input: [B, T] -> [B, 1, T]
        x = input_values.unsqueeze(1)
        
        # 2. Step 1: CNN Encoder (SEANet)
        # Returns [B, Hidden, T_inner]
        cnn_out = self.cnn_encoder(x)
        if isinstance(cnn_out, (tuple, list)):
            hidden_states = cnn_out[0]
        else:
            hidden_states = cnn_out

        # 3. Step 2: Transformer Encoder
        # Transformer expects [B, T_inner, Hidden]
        hidden_states = hidden_states.transpose(1, 2)
        
        # return_dict=False is critical here
        trans_out = self.transformer(hidden_states, return_dict=False)
        
        if isinstance(trans_out, (tuple, list)):
            hidden_states = trans_out[0]
        else:
            hidden_states = trans_out
            
        # 4. Step 3: Downsample (if configured)
        # Downsample expects [B, Hidden, T_inner]
        hidden_states = hidden_states.transpose(1, 2)
        if self.downsample is not None:
            hidden_states = self.downsample(hidden_states)

        # 5. Step 4: Quantizer (RVQ)
        # MimiResidualVectorQuantizer.encode returns [Q, B, T]
        codes = self.quantizer.encode(hidden_states)
        
        # Defense against potential tuple/ModelOutput (though RVQ usually returns Tensor)
        if isinstance(codes, (tuple, list)):
            codes = codes[0]
        elif hasattr(codes, 'audio_codes'):
            codes = codes.audio_codes

        # 6. Qwen3 Slice Logic: Just take the valid number of quantizers
        valid_q = self.config.encoder_valid_num_quantizers
        codes = codes[:valid_q, :, :] # Slice first dim [Q]
        
        # 7. Transpose to final format: [Q, B, T] -> [B, T, Q]
        codes = codes.transpose(0, 1).transpose(1, 2)
        
        return codes

class CodecExportWrapper(nn.Module):
    """
    Qwen3-TTS 音频解码器导出包装类 (保持不变)。
    """
    def __init__(self, model: Qwen3TTSTokenizerV2Model):
        super().__init__()
        self.decoder = model.decoder
        self.config = model.config
        self.decoder.eval()

    def forward(self, audio_codes):
        # Input: [B, T, Q] -> [B, Q, T]
        codes = audio_codes.transpose(1, 2)
        wav = self.decoder(codes)
        # Output: [B, 1, T] -> [B, T]
        return wav.squeeze(1)

class SpeakerEncoderExportWrapper(nn.Module):
    """
    Qwen3-TTS 说话人编码器导出包装类。
    """
    def __init__(self, speaker_encoder):
        super().__init__()
        self.speaker_encoder = speaker_encoder
        self.speaker_encoder.eval()

    def forward(self, mels):
        """
        Args:
            mels (torch.FloatTensor): [Batch, Seq, MelDim(128)]
        Returns:
            spk_emb (torch.FloatTensor): [Batch, 512]
        """
        return self.speaker_encoder(mels)

class MTPStepExportWrapper(nn.Module):
    """
    Qwen3-TTS MTP (Code Predictor) 单步导出包装类。
    因为 MTP 有 15 个不同的 Head 和 Embedding，我们将它们全部包含进来。
    """
    def __init__(self, code_predictor):
        super().__init__()
        self.code_predictor_model = code_predictor.model
        self.lm_heads = code_predictor.lm_head # ModuleList of 15 Linears
        self.small_to_mtp_projection = code_predictor.small_to_mtp_projection
        self.eval()

    def forward(self, inputs_embeds, step_idx, past_key_values_list=None):
        """
        Args:
            inputs_embeds: [Batch, Seq, Hidden]
            step_idx: int (0 to 14)
            past_key_values_list: Optional list/tuple of past KV
        """
        # 1. Project if needed
        hidden_states = self.small_to_mtp_projection(inputs_embeds)
        
        # 2. Model Forward
        # 注意：这里我们简化处理，假设一次跑一步或者一段。
        # 为了 ONNX 导出方便，我们可能需要更具体的 KV Cache 处理。
        # 但 MTP 总共只有 15 步，且序列很短（总共 16 2?），不带 KV Cache 跑 full self-attn 也可以。
        # 考虑到复杂度，这里先实现一个全序列版本的，推理时我们可以拼接序列。
        
        outputs = self.code_predictor_model(
            inputs_embeds=hidden_states,
            past_key_values=None, # 简化：不使用 KV Cache 避免 ONNX 复杂化
            use_cache=False
        )
        
        last_hidden = outputs.last_hidden_state[:, -1:, :]
        
        # 3. Apply the specific head for this step
        # 由于 ONNX 不支持动态索引 ModuleList，我们需要用一种 trick 或者导出多个模型。
        # 或者使用一个逻辑合并所有的 Head。
        # 这里为了导出一个模型，我们写一个 loop (ONNX 会展开它或者我们可以用 index_select)
        
        # 假设 step_idx 是 tensor
        all_logits = []
        for head in self.lm_heads:
            all_logits.append(head(last_hidden))
        
        # [15, B, 1, Vocab] -> [B, 1, Vocab]
        stacked_logits = torch.stack(all_logits, dim=0)
        # 使用 step_idx 选取对应的 logits
        logits = stacked_logits[step_idx] 
        
        return logits
