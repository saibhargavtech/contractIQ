fast_mode_config = {
    "CHUNK_MAX_CHARS": 1000,  # Smaller chunks
    "EXTRACT_TEMP": 0.1,      # Faster LLM responses
    "STRICT_PROMPT": False,   # More lenient extraction
    "REQUIRE_EVIDENCE": False, # Skip evidence requirement
    "MAX_COOCC_PAIRS": 3,     # Fewer co-occurrence pairs
    "ENABLE_FALLBACK_COOCC": False,  # Skip fallback processing
    "LOUVAIN_RESOLUTION": 0.8, # Less complex clustering
}
