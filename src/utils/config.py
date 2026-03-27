"""
config.py — Central configuration for the Handwriting Personality AI project.
"""

import os

# ─── Base Paths ───────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

INPUT_DIR        = os.path.join(BASE_DIR, "input", "handwriting_images")
OUTPUT_RESULTS   = os.path.join(BASE_DIR, "output", "results")
OUTPUT_REPORTS   = os.path.join(BASE_DIR, "output", "reports")
MODELS_PRETRAINED = os.path.join(BASE_DIR, "models", "pretrained")
MODELS_ML        = os.path.join(BASE_DIR, "models", "ml_models")

# ─── Preprocessing ────────────────────────────────────────────────────────────
PREPROCESS = {
    "resize_width":  1200,          # px — normalise image width
    "resize_height": None,          # None = keep aspect ratio
    "denoise_h":     10,            # OpenCV fastNlMeans filter strength
    "threshold_method": "otsu",     # "otsu" | "adaptive"
    "adaptive_block_size": 15,
    "adaptive_C":    8,
    "morph_kernel_size": (2, 2),
}

# ─── Feature Extraction ───────────────────────────────────────────────────────
FEATURES = {
    # Slant
    "slant_angle_bins":   180,
    "slant_min_line_len": 20,

    # Spacing
    "word_gap_threshold": 15,       # px between word-gaps
    "line_gap_threshold": 10,

    # Pressure
    "pressure_dark_thresh": 80,     # pixel intensity ≤ this = dark stroke

    # Baseline
    "baseline_ransac_residual": 5,

    # Letter size
    "min_component_area": 50,       # ignore noise blobs below this area

    # Margins
    "margin_threshold": 200,        # column pixel density threshold
}

# ─── Deep Learning ────────────────────────────────────────────────────────────
DEEP = {
    "device": "cpu",                # "cpu" | "cuda"
    "resnet_model":  "resnet50",
    "vit_model":     "google/vit-base-patch16-224",
    "image_size":    224,
    "batch_size":    1,
    "resnet_feature_dim": 2048,
    "vit_feature_dim":    768,
}

# ─── Personality Model ────────────────────────────────────────────────────────
PERSONALITY = {
    "traits": [
        "Openness",
        "Conscientiousness",
        "Extraversion",
        "Agreeableness",
        "Neuroticism",
    ],
    "model_path": os.path.join(MODELS_ML, "personality_model.pkl"),
    "use_rule_engine": True,         # fall back to rule engine when no pkl
}

# ─── Output ───────────────────────────────────────────────────────────────────
OUTPUT = {
    "save_features_json":  True,
    "save_report_txt":     True,
    "save_analysis_image": True,
    "report_filename":     "personality_report",
    "features_filename":   "features",
}
