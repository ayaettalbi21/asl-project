import json
import shutil
import h5py

IN_PATH = "models/asl_sign_model.h5"
OUT_PATH = "models/asl_sign_model_fixed.h5"

# 1) Copy to a new file (never modify original)
shutil.copy(IN_PATH, OUT_PATH)

with h5py.File(OUT_PATH, "r+") as f:
    if "model_config" not in f.attrs:
        raise RuntimeError("This .h5 does not contain 'model_config' attribute.")

    model_config = f.attrs["model_config"]
    if isinstance(model_config, bytes):
        model_config = model_config.decode("utf-8")

    cfg = json.loads(model_config)

    # Keras H5 usually: cfg["config"]["layers"]
    layers = cfg.get("config", {}).get("layers", [])
    changed = 0

    for layer in layers:
        if layer.get("class_name") == "InputLayer":
            lcfg = layer.get("config", {})
            if "batch_shape" in lcfg:
                # Replace the key
                lcfg["batch_input_shape"] = lcfg.pop("batch_shape")
                changed += 1

    if changed == 0:
        print("⚠️ No InputLayer batch_shape found. Nothing changed.")
    else:
        print(f"✅ Patched {changed} InputLayer(s): batch_shape -> batch_input_shape")

    f.attrs["model_config"] = json.dumps(cfg).encode("utf-8")

print("✅ Saved patched model to:", OUT_PATH)
