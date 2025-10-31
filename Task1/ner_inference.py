import argparse
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline


def extract_mountains(model_dir, text):
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForTokenClassification.from_pretrained(model_dir)
    # Create ner pipeline
    nlp = pipeline(
        "ner",
        model=model,
        tokenizer=tokenizer,
        aggregation_strategy="simple"  # create word from B/I tokens
    )

    # Get predictions
    results = nlp(text)
    # Take labels
    mountains = [r["word"].strip(',.!?') for r in results if "MOUNTAIN" in r["entity_group"]]
    return mountains


def main(args):
    mountains = extract_mountains(args.model_dir, args.text)
    if mountains:
        print(f'Mountains found: {", ".join(mountain.capitalize() for mountain in mountains)}')
    else: 
        print("Mountains not found!")
    # print(mountains[0] if mountains else "Mountains not found!")


if __name__ == "__main__":
    # Argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True, help="Path to the trained model directory")
    parser.add_argument("--text", type=str, required=True, help="Input text to process")
    args = parser.parse_args()
    main(args)
