import pandas as pd

def display_sample_table(sources, targets, predictions, num_samples=10):
    df = pd.DataFrame({
        "Source": sources[:num_samples],
        "Target": targets[:num_samples],
        "Prediction": predictions[:num_samples]
    })
    print(df.to_markdown(index=False))

# Example usage:
# display_sample_table(source_texts, expected, predicted)