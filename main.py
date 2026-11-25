import argparse

from model_pipeline import (
    prepare_data,
    train_model,
    evaluate_model,
    save_model,
    load_model
)

# Arguments
parser = argparse.ArgumentParser(description="Machine Learning Pipeline")

parser.add_argument("--all", action="store_true", help="Run the complete pipeline")
parser.add_argument("--prepare", action="store_true", help="Prepare the data")
parser.add_argument("--train", action="store_true", help="Train the model")
parser.add_argument("--evaluate", action="store_true", help="Evaluate the model")
parser.add_argument("--save", action="store_true", help="Save the model")
parser.add_argument("--load", action="store_true", help="Load the model")
parser.add_argument("--test", action="store_true", help="Test predictions")

args = parser.parse_args()

# Prepare data
if args.all or args.prepare or args.train or args.evaluate or args.test or args.save:
    x_train, x_test, y_train, y_test = prepare_data("Churn_Modelling.csv")
    print("Data preparation successful!")

# Train model
if args.all or args.train or args.evaluate or args.save or args.test:
    model = train_model(x_train, y_train)
    print("Model training successful!")

# Evaluate model
if args.all or args.evaluate or args.test or args.save:
    evaluate_model(model, x_test, y_test)
    print("Model evaluation successful!")
# Save model
if args.save or args.all or args.test:
    save_model(model)
    print("Model saved successfully!")
# Load model
if args.load or args.all or args.test:
    loaded_model = load_model()
    print("Model loaded successfully!")
# Test predictions
if args.test or args.all:
    predictions = loaded_model.predict(x_test)

    print("First 5 predictions:", predictions[:5])
    print("First 5 actual values:", y_test.iloc[:5].values)
