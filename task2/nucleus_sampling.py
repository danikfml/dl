from config import model, tokenizer, EOS_TOKEN_ID, MAX_LENGTH, device, input_text_hedgehog, input_text_json, logger
import torch


def generate_nucleus(input_text, temperature=1.0, top_p=0.9):
    try:
        assert 0 < top_p <= 1.0, "top_p должен быть в (0, 1]"

        inputs = tokenizer(input_text, return_tensors="pt").to(device)
        input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask

        for _ in range(MAX_LENGTH - input_ids.shape[1]):
            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits[0, -1] / temperature
            probs = torch.softmax(logits, dim=-1)

            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
            sorted_indices_to_remove[0] = 0

            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            probs[indices_to_remove] = 0
            probs /= probs.sum()

            next_token = torch.multinomial(probs, 1)
            input_ids = torch.cat([input_ids, next_token.view(1, 1).to(device)], dim=1)
            attention_mask = torch.cat([attention_mask, torch.ones((1, 1), device=device)], dim=1)

            if next_token.item() == EOS_TOKEN_ID:
                break

        return tokenizer.decode(input_ids[0], skip_special_tokens=False)

    except Exception as e:
        logger.error(f"Ошибка nucleus sampling: {e}")
        return ""


if __name__ == "__main__":
    params = [
        {"temperature": 1.0, "top_p": 0.9},
        {"temperature": 1.0, "top_p": 0.15},
        {"temperature": 0.5, "top_p": 0.9},
        {"temperature": 0.5, "top_p": 0.15},
    ]

    for param in params:
        print(f"\n=== Параметры {param} ===")
        print("Сказка:", generate_nucleus(input_text_hedgehog, **param))
        print("JSON:", generate_nucleus(input_text_json, **param))