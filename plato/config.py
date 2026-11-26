"""Immutable public checkpoints. All runs use frozen weights and greedy decoding."""

MODELS = {
    "qwen": ("Qwen/Qwen2.5-1.5B", "8faed761d45a263340a0528343f099c05c9a4323"),
    "smol": ("HuggingFaceTB/SmolLM2-1.7B", "effd688a12921b4cc83e3312b6feb579f70f9c71"),
    "pythia": ("EleutherAI/pythia-1.4b-deduped", "554d9c1bae3877f740aece41feb90f912cf9fedc"),
}
