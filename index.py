import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import time

MODEL_NAME = "distilbert-base-uncased"

id2label = {
    0: "admiration", 1: "amusement", 2: "anger", 3: "annoyance", 4: "approval",
    5: "caring", 6: "confusion", 7: "curiosity", 8: "desire", 9: "disappointment",
    10: "disapproval", 11: "disgust", 12: "embarrassment", 13: "excitement",
    14: "fear", 15: "gratitude", 16: "grief", 17: "joy", 18: "love", 19: "nervousness",
    20: "neutral", 21: "optimism", 22: "pride", 23: "realization", 24: "relief",
    25: "remorse", 26: "sadness", 27: "surprise"
}

def load_model():
    tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)
    model = DistilBertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=28)
    model.eval()
    
    model = torch.quantization.quantize_dynamic(
        model,
        {torch.nn.Linear},
        dtype=torch.qint8
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return tokenizer, model, device

def predict_emotion(text, tokenizer, model, device, top_k=3):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        max_length=64,
        truncation=True,
        padding='max_length'
    ).to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
        
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]
    top_probs, top_indices = torch.topk(probs, top_k)
    
    return [
        {"emotion": id2label[i.item()], "confidence": p.item()}
        for p, i in zip(top_probs, top_indices)
    ]

def main():
    print("Loading model (this will take ~2 seconds)...")
    start = time.time()
    tokenizer, model, device = load_model()
    print(f"Model loaded in {time.time() - start:.1f}s\n")
    
    print("Emotion Detection ready! Type a sentence and press Enter.")
    print("Type 'quit' or 'exit' to end.\n")
    
    while True:
        try:
            text = input("You: ").strip()
            if text.lower() in ['quit', 'exit']:
                break
                
            if not text:
                print("Please enter some text")
                continue
                
            start_time = time.time()
            results = predict_emotion(text, tokenizer, model, device)
            elapsed = time.time() - start_time
            
            print("\nDetected Emotions:")
            for i, res in enumerate(results, 1):
                print(f"{i}. {res['emotion']} ({res['confidence']*100:.1f}%)")
            print(f"(Processed in {elapsed:.2f}s)\n")
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except EOFError:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
