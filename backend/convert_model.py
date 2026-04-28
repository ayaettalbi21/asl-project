import tensorflow as tf
from tensorflow.keras.models import load_model

OLD_PATH = "models/asl_sign_model_fixed.h5"
NEW_PATH = "models/asl_sign_model.keras"

# --- Fix for "Unknown dtype policy: DTypePolicy"
# Some models saved with dtype policy need these custom objects.
custom_objects = {}

# Keras/TF versions differ; we try the safest mappings.
try:
    from tensorflow.keras.mixed_precision import Policy
    custom_objects["DTypePolicy"] = Policy
except Exception:
    pass

try:
    # TF 2.15+ sometimes exposes a DTypePolicy class
    from tensorflow.keras.mixed_precision import DTypePolicy
    custom_objects["DTypePolicy"] = DTypePolicy
except Exception:
    pass

# Load model
model = load_model(OLD_PATH, compile=False, custom_objects=custom_objects)

# Save in modern format
model.save(NEW_PATH)
print("✅ Converted to:", NEW_PATH)
