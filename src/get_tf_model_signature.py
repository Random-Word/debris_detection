import tensorflow as tf
import argparse

def inspect_model(saved_model_path):
    # Load the saved TensorFlow model
    loaded_model = tf.saved_model.load(saved_model_path)

    # Retrieve the `serving_default` signature
    serving_default = loaded_model.signatures['serving_default']

    # Inspect input and output tensors
    print('Input Tensors for serving:')
    for input_name, input_tensor in serving_default.structured_input_signature[1].items():
        print(f"Name: {input_name}, Shape: {input_tensor.shape}, Dtype: {input_tensor.dtype}")

    print('\nOutput Tensors for serving:')
    for output_name, output_tensor in serving_default.structured_outputs.items():
        print(f"Name: {output_name}, Shape: {output_tensor.shape}, Dtype: {output_tensor.dtype}")


def main():
    parser = argparse.ArgumentParser(description='Inspect TensorFlow SavedModel Input/Output')
    parser.add_argument('model_path', type=str, help='Path to the TensorFlow SavedModel directory')
    args = parser.parse_args()

    # Inspect the model
    inspect_model(args.model_path)

if __name__ == '__main__':
    main()

