from deepface import DeepFace

def analyze_face(face_crop):
    """Predicts the age and gender of a detected face using DeepFace."""
    try:
        result = DeepFace.analyze(face_crop, actions=["age", "gender"], enforce_detection=False)
        if result:
            age = result[0].get("age", "Unknown")
            gender = result[0].get("dominant_gender", "Unknown")
            return age, gender
    except Exception as e:
        print(f"Error analyzing face: {e}")

    return "Unknown", "Unknown"
