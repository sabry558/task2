from sequence import sequence
import numpy as np

def test_sequence_model():
    # Initialize the model
    num_of_layers = 10  # Adjust based on your architecture
    epochs = 10
    seq_model = sequence(num_of_layers, epochs)

    # Build layers
    seq_model.build_layers()

    # Preprocess data
    x_train, x_test, y_train, y_test = seq_model.preprocess()

    # Train the model
    print("Starting training...")
    seq_model.train()
    print("Training completed!")

    # Evaluate on test data
    print("\nEvaluating on test data...")
    total_loss = 0
    correct_predictions = 0

    x_test = x_test.to_numpy()  # Convert to NumPy for faster calculations
    y_test = y_test.to_numpy()

    for i, sample in enumerate(x_test):
        seq_model.forward_propagation(sample)
        predicted_output = seq_model.layers[-1].a_out
        true_output = y_test[i]

        # Calculate loss (Mean Squared Error for regression tasks)
        total_loss += np.mean((predicted_output - true_output) ** 2)

        # For classification: Choose the neuron with the highest activation as the predicted class
        predicted_class = np.argmax(predicted_output)
        true_class = np.argmax(true_output)
        print(predicted_class,true_class)

        if predicted_class == true_class:
            correct_predictions += 1

    # Calculate metrics
    average_loss = total_loss / len(x_test)
    accuracy = correct_predictions / len(x_test) * 100

    print(f"Average Loss on Test Data: {average_loss}")
    print(f"Accuracy on Test Data: {accuracy:.2f}%")

    # Return metrics for further analysis
    return average_loss, accuracy


# Run the test
if __name__ == "__main__":
    loss, accuracy = test_sequence_model()
    print(f"\nFinal Results:\nLoss: {loss}\nAccuracy: {accuracy:.2f}%")
