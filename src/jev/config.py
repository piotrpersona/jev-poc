from __future__ import annotations

import logging
import os
from dataclasses import dataclass

import structlog
import torch
from dotenv import load_dotenv


def resolve_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@dataclass(frozen=True)
class JevConfig:
    backbone: str = "answerdotai/ModernBERT-base"
    device: str = "auto"
    max_state_tokens: int = 1024
    max_option_tokens: int = 32
    batch_size: int = 8
    attention_heads: int = 8

    @classmethod
    def from_env(cls) -> JevConfig:
        load_dotenv()
        return cls(
            backbone=os.getenv("JEV_BACKBONE", cls.backbone),
            device=os.getenv("JEV_DEVICE", cls.device),
            max_state_tokens=int(os.getenv("JEV_MAX_STATE_TOKENS", cls.max_state_tokens)),
            max_option_tokens=int(os.getenv("JEV_MAX_OPTION_TOKENS", cls.max_option_tokens)),
            batch_size=int(os.getenv("JEV_BATCH_SIZE", cls.batch_size)),
        )

    @property
    def torch_device(self) -> torch.device:
        return resolve_device(self.device)


def configure_logging(level: int = logging.INFO) -> None:
    structlog.configure(
        processors=[
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        cache_logger_on_first_use=True,
    )
