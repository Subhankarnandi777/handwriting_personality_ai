def estimate_emotion(features: dict) -> dict:
    """
    Approximates short-term emotional states and mental focus 
    based on variations in the handwriting metrics.
    """
    emotion = {
        "stress_level": 0.5, 
        "energy_level": 0.5, 
        "focus_level": 0.5
    }
    
    # High pressure + wavy baseline = Stress
    pressure = features.get("pressure_score", 0.5)
    baseline_reg = features.get("baseline_regularity", 0.5)
    emotion["stress_level"] = max(0.0, min(1.0, pressure + (1.0 - baseline_reg) * 0.5))
    
    # Fast slant + large size = High energy
    slant = features.get("slant_score", 0.5)
    size = features.get("letter_size_score", 0.5)
    emotion["energy_level"] = max(0.0, min(1.0, (slant + size) / 2.0))
    
    # Consistent spacing & letters = High focus
    spacing = features.get("word_spacing_score", 0.5)
    size_consist = features.get("letter_consistency", 0.5)
    emotion["focus_level"] = max(0.0, min(1.0, (spacing + size_consist) / 2.0))
    
    return emotion
