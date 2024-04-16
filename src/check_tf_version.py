import tensorflow as tf

def check_tf_version(saved_model_dir):
    # Load the saved model
    loaded_model = tf.saved_model.load(saved_model_dir)

    # Check for TensorFlow version information in the saved model
    try:
        model_version_info = loaded_model.version
        print(f"Model was saved using TensorFlow version: {model_version_info}")
    except AttributeError:
        print("Model does not contain explicit TensorFlow version information.")

    # Check for eager execution (default in TF2)
    if tf.executing_eagerly():
        print("Currently executing in eager mode (TF2 default).")
    else:
        print("Currently executing in graph mode (TF1 default).")

    # Try to access a feature typical for TF2 (like a Keras model within the SavedModel)
    try:
        _ = loaded_model.signatures
        print("Model contains signatures attribute, likely a TF2 SavedModel.")
    except AttributeError:
        print("Model does not contain signatures attribute, may be a TF1 model.")

# Provide the path to your saved model directory
check_tf_version('models/detectors_d1000_debris_only_versions_13_export/saved_model/13')
