import numpy as np
import joblib


# Function to load the correct model based on user input
def load_model(model_name):
    if model_name.lower() == 'knn':
        model = joblib.load('Models/knn_model.pkl')
    elif model_name.lower() == 'svm':
        model = joblib.load('Models/svm_model.pkl')
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return model


def get_user_input():
    while True:
        try:
            # Prompt user for 8 space-separated numbers
            input_str = input("Enter 8 numerical values separated by spaces: ")
            
            # Split input by spaces and convert to a list of floats
            input_values = list(map(float, input_str.split()))

            # Check if exactly 8 values were entered
            if len(input_values) != 8:
                raise ValueError("Please enter exactly 8 values.")

            # Convert to a 2D NumPy array
            return np.array([input_values])

        except ValueError as e:
            print(f"Invalid input: {e}. Try again.")



def main():

    print("Input step reached")

    model_name = input("Which model would you like to use? (KNN/SVM): ").strip()

    # Check if model_name is captured
    print(f"Captured input: {model_name}")


    if not model_name:
        print("Error: No model name entered.")
        return

    print(f"Model chosen: {model_name}")

    try:

        model = load_model(model_name)
    except ValueError as e:
        print(e)
        return


    user_input = get_user_input()


    prediction = model.predict(user_input)
    print(f"Predicted class: {prediction[0]}")

if __name__ == "__main__":
    main()
