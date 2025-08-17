# llama_backbone/encoder.py
import os
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, TaskType, get_peft_model

# Optional: keep cache local to the repo
os.environ.setdefault("TRANSFORMERS_CACHE",
    "/home/manhduc/manhduc/pointnet.pytorch-modified/")

# ---------- Point -> Token adapter ----------
class PointTokenizer(nn.Module):
    """
    Maps numeric point features [B, N, F] -> [B, N, d_model] for LLaMA via a small MLP.
    Optional Fourier features help inject locality for 3D coords.
    """
    def __init__(self, in_dim: int, d_model: int, fourier_k: int = 32):
        super().__init__()
        self.fourier_k = fourier_k
        if fourier_k > 0:
            B = torch.randn(in_dim, fourier_k) * 10.0
            self.register_buffer("B", B)  # [F, K]
            ext = 2 * fourier_k
        else:
            ext = 0
        self.net = nn.Sequential(
            nn.Linear(in_dim + ext, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, d_model),
            nn.LayerNorm(d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N, F]
        if self.fourier_k > 0:
            proj = x @ self.B  # [B, N, K]
            x = torch.cat([x, torch.sin(proj), torch.cos(proj)], dim=-1)
        return self.net(x)  # [B, N, d_model]

# ---------- LLaMA encoder that accepts numeric points ----------
class LLaMAEncoder(nn.Module):
    """
    Use LLaMA as a bidirectional-ish feature encoder for point sets, feeding inputs_embeds.
    Inputs:  points [B, N, F], attn_mask [B, N] (1=real, 0=pad)
    Output:  hidden [B, N, d_model]
    """
    def __init__(
        self,
        model_or_path: str,
        in_dim: int = 3,
        fourier_k: int = 32,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        load_in_4bit: bool = True,
        compute_dtype: torch.dtype = torch.float16,   # Ampere (RTX 3080) -> fp16
        trust_remote_code: bool = False,
    ):
        super().__init__()

        # BitsAndBytes 4-bit
        qconfig = None
        if load_in_4bit:
            qconfig = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=True,
            )

        HF_TOKEN = os.environ.get("HUGGINGFACE_HUB_TOKEN") or os.environ.get("HF_TOKEN")

        # Load the LM wrapper (NOT the bare LlamaModel)
        self.llm = AutoModelForCausalLM.from_pretrained(
            model_or_path,
            quantization_config=qconfig,
            device_map="auto",
            trust_remote_code=trust_remote_code,
            token=HF_TOKEN,
        )
        self.llm.config.output_hidden_states = True
        self.llm.config.use_cache = False
        try:
            self.llm.gradient_checkpointing_enable()
        except Exception:
            pass

        # LoRA on the wrapper
        lora_cfg: LoraConfig = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
        )
        self.llm = get_peft_model(self.llm, lora_cfg)

        # Transformer blocks (LlamaModel) for direct feature extraction
        self.backbone = self.llm.model
        d_model = self.backbone.config.hidden_size

        # Numeric point -> token adapter
        self.point_tok = PointTokenizer(in_dim=in_dim, d_model=d_model, fourier_k=fourier_k)

    @property
    def d_model(self) -> int:
        return self.backbone.config.hidden_size

    def forward(self, points: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        # points: [B,N,F], attn_mask: [B,N] (1 real, 0 pad)
        embeds = self.point_tok(points)  # fp32 by default
        bb_dtype = next(self.backbone.parameters()).dtype
        embeds = embeds.to(dtype=bb_dtype)

        # Call transformer body (not lm_head)
        out = self.backbone(
            inputs_embeds=embeds,
            attention_mask=attn_mask,
            output_hidden_states=True,
            use_cache=False,
        )
        # Robustly fetch last hidden layer
        if getattr(out, "hidden_states", None) is not None:
            return out.hidden_states[-1]     # [B, N, d_model]
        if hasattr(out, "last_hidden_state"):
            return out.last_hidden_state
        raise RuntimeError("Backbone output has neither hidden_states nor last_hidden_state")