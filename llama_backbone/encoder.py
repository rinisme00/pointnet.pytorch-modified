import os
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model

# (Optional) keep all caches under your project folder
os.environ["TRANSFORMERS_CACHE"] = "/home/manhduc/manhduc/pointnet.pytorch-modified/"

class LLaMAEncoder(torch.nn.Module):
    def __init__(
        self,
        model_id: str = "TheBloke/Llama-2-7B-GPTQ",
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
    ):
        super().__init__()
        # 1) tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id, use_auth_token=True
        )

        # 2) load quantized in 4-bit
        qconfig = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=qconfig,
            device_map="auto",
            trust_remote_code=True,
            use_auth_token=True,
        )

        # 3) enable hidden states output
        self.model.config.output_hidden_states = True

        # 4) attach LoRA for fine-tuning
        lora_cfg = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=["q_proj", "v_proj"],  # adapt as needed
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        self.model = get_peft_model(self.model, lora_cfg)

    def forward(self, texts: list[str]) -> torch.Tensor:
        # tokenize
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(self.model.device)

        # forward—with grads enabled for LoRA layers
        outputs = self.model.model(
            **inputs,
            output_hidden_states=True
        )
        # last hidden layer: (batch, seq_len, hidden_dim)
        last_hidden = outputs.hidden_states[-1]
        # mean-pool → (batch, hidden_dim)
        return last_hidden.mean(dim=1)