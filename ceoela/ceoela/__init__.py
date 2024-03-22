from typing import List
from .ceoela_pipeline import ceoela_pipeline
from .utils import excel2doe



__all__: List[str] = [
    "ceoela_pipeline",
    "excel2doe",
]


# comment this to hide the print out :(
from art import tprint
tprint("CEOELA", font="big") # font = {"random", "rnd-small", "rnd-medium", "rnd-large", "rnd-xlarge"}