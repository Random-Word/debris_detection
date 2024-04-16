"""
Load a tensorflow model and visualize it using tensorboard.
"""

import argparse
import tensorflow as tf
from tensorflow import summary

def load_and_log_model(saved_model_path: str, log_dir: str = './logs'):
    # Load the saved TensorFlow model
    loaded_model = tf.saved_model.load(saved_model_path)

    # Create a summary writer
    writer = summary.create_file_writer(log_dir)

    # Access the root function of the loaded model to get the concrete function
    concrete_func = loaded_model.signatures['serving_default']

    # Log the graph to TensorBoard
    with writer.as_default():
        tf.summary.graph(concrete_func.graph)

    # Close the writer after logging the graph
    writer.close()

    # Print instructions to start TensorBoard
    print(f'TensorBoard can be accessed by running: tensorboard --logdir={log_dir}')

def main():
    parser = argparse.ArgumentParser(description='TensorFlow SavedModel Graph Visualizer')
    parser.add_argument('model_path', type=str, help='Path to the TensorFlow SavedModel directory')
    parser.add_argument('--log_dir', type=str, default='./logs', help='Path to the directory where TensorBoard logs should be saved')
    args = parser.parse_args()

    # Call the function to load and log the model
    load_and_log_model(args.model_path, args.log_dir)

if __name__ == '__main__':
    main()
