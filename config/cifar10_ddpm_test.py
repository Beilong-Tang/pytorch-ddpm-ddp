from .template import DefaultConfig

from dataclasses import dataclass

@dataclass
class Config(DefaultConfig):
    def __post_init__(self):
        self.logdir="./logs/test"
