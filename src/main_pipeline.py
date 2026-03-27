"""
main_pipeline.py — Orchestrates the full handwriting → personality pipeline.

Steps
-----
1.  Load & clean image
2.  Threshold → binary
3.  Segment lines, words, characters
4.  Extract handcrafted features
5.  Extract deep features  (ResNet + ViT)
6.  Fuse features
7.  Predict personality
8.  Save results + report
"""

import os
import json
import time
from datetime import datetime
from typing import Optional

import numpy as np

from src.utils.config import OUTPUT, OUTPUT_RESULTS, OUTPUT_REPORTS, PERSONALITY
from src.utils.helper import (
    load_image, save_image, save_json, ensure_dirs,
    timestamped_name, logger,
)

# Preprocessing
from src.preprocessing.image_cleaning  import clean_image
from src.preprocessing.thresholding    import threshold_image
from src.preprocessing.segmentation    import segment_all

# Feature extraction
from src.feature_extraction.slant_detection   import detect_slant
from src.feature_extraction.spacing_analysis  import analyze_spacing
from src.feature_extraction.pressure_analysis import analyze_pressure, stroke_width_variation
from src.feature_extraction.baseline_detection import detect_baseline
from src.feature_extraction.letter_size        import analyze_letter_size
from src.feature_extraction.margins            import detect_margins

# Deep features
from src.deep_features.resnet_features import extract_resnet_features
from src.deep_features.vit_features    import extract_vit_features
from src.deep_features.feature_fusion  import fuse_features

# Personality
from src.personality_model.personality_predictor import PersonalityPredictor

# Visualisation
from src.utils.visualization import plot_full_analysis


# ─── Lazy singleton predictor ─────────────────────────────────────────────────
_PREDICTOR: Optional[PersonalityPredictor] = None

def _get_predictor() -> PersonalityPredictor:
    global _PREDICTOR
    if _PREDICTOR is None:
        _PREDICTOR = PersonalityPredictor()
    return _PREDICTOR


# ─── Main pipeline function ───────────────────────────────────────────────────

def run_pipeline(
    image_path: str,
    use_deep_features: bool = True,
    save_outputs: bool = True,
    is_signature: bool = False,
) -> dict:
    """
    Full pipeline from image path to personality prediction.

    Parameters
    ----------
    image_path        : path to the handwriting image
    use_deep_features : whether to run ResNet + ViT (slower but richer)
    save_outputs      : save results, reports, and visualisation to disk
    is_signature      : enable signature specific extraction logic

    Returns
    -------
    dict with keys: features, personality, output_paths, elapsed_sec
    """
    t0 = time.time()
    ensure_dirs(OUTPUT_RESULTS, OUTPUT_REPORTS)

    logger.info("═" * 60)
    logger.info("Starting pipeline for: %s", image_path)

    # ── Step 1: Load & clean ──────────────────────────────────────────────
    if image_path.lower().endswith(".pdf"):
        from src.preprocessing.pdf_parser import extract_images_from_pdf
        pdf_images = extract_images_from_pdf(image_path)
        if not pdf_images:
            raise ValueError("No images could be extracted/rasterised from the PDF.")
        bgr = pdf_images[0]
    else:
        bgr   = load_image(image_path)
        
    gray  = clean_image(bgr)

    # ── Step 2: Threshold ─────────────────────────────────────────────────
    binary = threshold_image(gray)

    # ── Step 3: Segment ───────────────────────────────────────────────────
    seg = segment_all(binary)

    # ── Step 4: Handcrafted features ──────────────────────────────────────
    features: dict = {}
    
    if is_signature:
        from src.feature_extraction.signature import analyze_signature
        features.update(analyze_signature(binary, seg["char_bboxes"]))
        
    features.update(detect_slant(binary))
    features.update(analyze_spacing(binary, seg["lines"], seg["char_bboxes"]))
    features.update(analyze_pressure(gray, binary))
    features.update(stroke_width_variation(binary))
    features.update(detect_baseline(binary, seg["char_bboxes"]))
    features.update(analyze_letter_size(seg["char_bboxes"], gray.shape[0]))
    features.update(detect_margins(binary))

    # Segmentation meta-features
    features["num_lines"]      = seg["num_lines"]
    features["num_chars"]      = seg["num_chars"]
    features["avg_words_line"] = seg["avg_words_line"]
    
    # Emotion / State of mind estimation
    from src.feature_extraction.emotion import estimate_emotion
    emotions = estimate_emotion(features)
    for k, v in emotions.items():
        features[k] = v

    logger.info("Handcrafted features extracted: %d features", len(features))

    # ── Step 5: Deep features ─────────────────────────────────────────────
    resnet_vec = np.zeros(2048, dtype=np.float32)
    vit_vec    = np.zeros(768,  dtype=np.float32)

    if use_deep_features:
        try:
            resnet_vec = extract_resnet_features(gray)
        except Exception as exc:
            logger.warning("ResNet extraction failed: %s", exc)
        try:
            vit_vec = extract_vit_features(gray)
        except Exception as exc:
            logger.warning("ViT extraction failed: %s", exc)

    # ── Step 6: Fuse ──────────────────────────────────────────────────────
    fused_vec = fuse_features(features, resnet_vec, vit_vec,
                              use_deep=use_deep_features)

    # ── Step 7: Predict ───────────────────────────────────────────────────
    predictor   = _get_predictor()
    prediction  = predictor.predict(features, fused_vec)
    scores      = prediction["scores"]
    labels      = prediction["labels"]
    summary_txt = predictor.personality_summary(scores, labels)

    logger.info("Personality prediction complete (%s).", prediction["method"])
    logger.info("\n%s", summary_txt)
    
    # Log to SQLite DB
    from src.utils.database import log_analysis
    log_file_name = os.path.basename(image_path)
    try:
        log_analysis(log_file_name, scores, prediction["method"])
        logger.info("Saved analysis to database.")
    except Exception as exc:
        logger.warning("Failed to save to database: %s", exc)

    # ── Step 8: Save outputs ──────────────────────────────────────────────
    output_paths = {}
    stem = os.path.splitext(os.path.basename(image_path))[0]
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")

    if save_outputs:
        # Features JSON
        if OUTPUT["save_features_json"]:
            json_path = os.path.join(OUTPUT_RESULTS, f"{stem}_features_{ts}.json")
            save_json(features, json_path)
            output_paths["features_json"] = json_path

        # Personality report TXT
        if OUTPUT["save_report_txt"]:
            report_path = os.path.join(OUTPUT_REPORTS, f"{stem}_report_{ts}.txt")
            _write_report(report_path, image_path, features, prediction, summary_txt)
            output_paths["report_txt"] = report_path

        # Analysis visualisation PNG
        if OUTPUT["save_analysis_image"]:
            fig_path = os.path.join(OUTPUT_RESULTS, f"{stem}_analysis_{ts}.png")
            try:
                plot_full_analysis(bgr, features, scores, labels, save_path=fig_path)
                output_paths["analysis_png"] = fig_path
            except Exception as exc:
                logger.warning("Visualisation failed: %s", exc)

        # PDF Report
        try:
            from src.utils.pdf_generator import generate_pdf_report
            pdf_path = os.path.join(OUTPUT_REPORTS, f"{stem}_report_{ts}.pdf")
            generate_pdf_report(
                path=pdf_path,
                image_path=image_path,
                features=features,
                prediction=prediction,
                summary=summary_txt,
                analysis_image_path=output_paths.get("analysis_png")
            )
            output_paths["report_pdf"] = pdf_path
            logger.info("PDF Report saved → %s", pdf_path)
        except Exception as exc:
            logger.warning("PDF generation failed: %s", exc)
            
        # XAI Heatmap
        if use_deep_features:
            try:
                from src.deep_features.xai import generate_activation_heatmap
                xai_img = generate_activation_heatmap(bgr)
                xai_path = os.path.join(OUTPUT_RESULTS, f"{stem}_xai_{ts}.jpg")
                import cv2
                cv2.imwrite(xai_path, xai_img)
                output_paths["xai_heatmap"] = xai_path
                logger.info("XAI Heatmap saved → %s", xai_path)
            except Exception as exc:
                logger.warning("XAI Heatmap generation failed: %s", exc)

    elapsed = round(time.time() - t0, 2)
    logger.info("Pipeline complete in %.2f seconds.", elapsed)
    logger.info("═" * 60)

    return {
        "features":     features,
        "personality":  prediction,
        "output_paths": output_paths,
        "elapsed_sec":  elapsed,
    }


# ─── Report writer ────────────────────────────────────────────────────────────

def _write_report(path: str, image_path: str, features: dict,
                  prediction: dict, summary: str) -> None:
    scores = prediction["scores"]
    labels = prediction["labels"]
    rules  = prediction["rules"]

    lines = [
        "=" * 62,
        "    HANDWRITING PERSONALITY ANALYSIS REPORT",
        "=" * 62,
        f"  Image      : {os.path.basename(image_path)}",
        f"  Generated  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"  Method     : {prediction['method']}",
        "",
        "─" * 62,
        "  PERSONALITY PROFILE",
        "─" * 62,
        summary,
        "",
        "─" * 62,
        "  DETAILED SCORES",
        "─" * 62,
    ]
    for trait, score in scores.items():
        lines.append(f"  {trait:<18} {score:.4f}   {labels[trait]}")

    lines += [
        "",
        "─" * 62,
        "  EXTRACTED FEATURES",
        "─" * 62,
    ]
    for k, v in features.items():
        if isinstance(v, float):
            lines.append(f"  {k:<30} {v:.4f}")
        elif isinstance(v, (int, str)):
            lines.append(f"  {k:<30} {v}")

    lines += [
        "",
        "─" * 62,
        "  RULE ENGINE — FIRED RULES",
        "─" * 62,
    ]
    for r in rules:
        lines.append(
            f"  [{r['trait']:<18}]  {r['feature']:<28} "
            f"val={r['value']:.3f}  w={r['weight']:+.2f}  → {r['effect']}"
        )
        lines.append(f"      Reason: {r['reasoning']}")

    lines += ["", "=" * 62, "  End of Report", "=" * 62]

    ensure_dirs(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    logger.info("Report saved → %s", path)
