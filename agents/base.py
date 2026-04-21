"""Base agent and LLM inference utilities."""
import os
import json
import yaml
from typing import List, Dict, Optional
from dataclasses import dataclass, field
from loguru import logger


@dataclass
class Message:
    role: str  # "tutor", "patient", "student", "assessment", "system"
    content: str
    metadata: Optional[Dict] = None


@dataclass
class DialogueState:
    """Tracks the full state of a tutoring dialogue."""

    case_id: str
    condition: str  # "multi_agent", "single_agent", "direct_qa"
    student_level: str
    seed: int
    messages: List[Message] = field(default_factory=list)
    assessments: List[Dict] = field(default_factory=list)
    turn_count: int = 0
    diagnosis_reached: bool = False
    final_diagnosis: str = ""
    max_turns: int = 20

    def add_message(self, role: str, content: str, metadata: Dict = None):
        self.messages.append(Message(role=role, content=content, metadata=metadata))
        self.turn_count += 1

    def get_dialogue_history(self, for_role: str = None) -> str:
        """Format dialogue history as text for LLM context."""
        lines = []
        for msg in self.messages:
            label = msg.role.upper()
            lines.append(f"[{label}]: {msg.content}")
        return "\n\n".join(lines)

    def is_complete(self) -> bool:
        return self.diagnosis_reached or self.turn_count >= self.max_turns

    def to_dict(self) -> Dict:
        return {
            "case_id": self.case_id,
            "condition": self.condition,
            "student_level": self.student_level,
            "seed": self.seed,
            "turn_count": self.turn_count,
            "diagnosis_reached": self.diagnosis_reached,
            "final_diagnosis": self.final_diagnosis,
            "messages": [
                {"role": m.role, "content": m.content, "metadata": m.metadata}
                for m in self.messages
            ],
            "assessments": self.assessments,
        }


class LLMBackend:
    """Unified LLM backend: vLLM (batch) or transformers (fallback)."""

    def __init__(self, model_name: str, config_path: str = None):
        self.model_name = model_name
        self.engine = None
        self.tokenizer = None
        self.backend_type = None

        if config_path:
            with open(config_path) as f:
                cfg = yaml.safe_load(f)
            self.temperature = cfg["llm"].get("temperature", 0.7)
            self.max_tokens = cfg["llm"].get("max_tokens", 512)
        else:
            self.temperature = 0.7
            self.max_tokens = 512

    def initialize(self):
        """Try vLLM first, fall back to transformers."""
        try:
            from vllm import LLM, SamplingParams

            logger.info(f"Initializing vLLM for {self.model_name}")
            self.engine = LLM(
                model=self.model_name,
                tensor_parallel_size=4,
                gpu_memory_utilization=0.90,
                max_model_len=4096,
                trust_remote_code=True,
            )
            self.sampling_params = SamplingParams(
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                top_p=0.95,
            )
            self.backend_type = "vllm"
            logger.info("vLLM initialized successfully")
        except ImportError:
            logger.warning("vLLM not available, using transformers")
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, trust_remote_code=True
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            self.engine = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
            )
            self.engine.eval()
            self.backend_type = "transformers"

    def generate(self, prompt: str) -> str:
        """Generate a single response."""
        if self.backend_type == "vllm":
            outputs = self.engine.generate([prompt], self.sampling_params)
            return outputs[0].outputs[0].text.strip()
        else:
            return self._transformers_generate(prompt)

    def generate_batch(self, prompts: List[str]) -> List[str]:
        """Generate responses for a batch of prompts."""
        if self.backend_type == "vllm":
            outputs = self.engine.generate(prompts, self.sampling_params)
            return [o.outputs[0].text.strip() for o in outputs]
        else:
            return [self._transformers_generate(p) for p in prompts]

    def _transformers_generate(self, prompt: str) -> str:
        import torch

        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=3584
        ).to(self.engine.device if hasattr(self.engine, "device") else "cuda")

        with torch.no_grad():
            out = self.engine.generate(
                **inputs,
                max_new_tokens=self.max_tokens,
                temperature=self.temperature,
                do_sample=True,
                top_p=0.95,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        input_len = inputs["input_ids"].shape[1]
        return self.tokenizer.decode(out[0][input_len:], skip_special_tokens=True).strip()
