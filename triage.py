"""Rule-based triage + first-aid guidance for rural health workers (decision support only)."""
RED_FLAGS = {"chest_pain", "breathlessness", "coma", "altered_sensorium", "weakness_of_one_body_side",
             "slurred_speech", "blood_in_sputum", "bloody_stool", "vomiting_blood", "stomach_bleeding",
             "loss_of_consciousness", "high_fever", "paralysis"}

def triage(symptoms, top_prob, xray_prob=None):
    s = set(symptoms)
    if s & RED_FLAGS or (xray_prob is not None and xray_prob >= 0.85):
        return ("EMERGENCY", "Refer immediately to the nearest district hospital / call 108.")
    if top_prob >= 0.6 or (xray_prob is not None and xray_prob >= 0.5) or len(s) >= 6:
        return ("URGENT", "Consult a doctor at the PHC within 24 hours (tele-consultation if available).")
    return ("ROUTINE", "Rest, fluids, monitor. Visit PHC if symptoms persist beyond 2-3 days.")
