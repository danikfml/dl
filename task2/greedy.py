from config import model, tokenizer, EOS_TOKEN_ID, MAX_LENGTH, device, input_text_hedgehog, input_text_json, logger
import torch


def generate_greedy(input_text):
    try:
        inputs = tokenizer(input_text, return_tensors="pt").to(device)
        input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask

        for _ in range(MAX_LENGTH - input_ids.shape[1]):
            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            next_token = torch.argmax(outputs.logits[0, -1])

            input_ids = torch.cat([input_ids, next_token.view(1, 1).to(device)], dim=1)
            attention_mask = torch.cat([attention_mask, torch.ones((1, 1), device=device)], dim=1)

            if next_token.item() == EOS_TOKEN_ID:
                break

        return tokenizer.decode(input_ids[0], skip_special_tokens=False)

    except Exception as e:
        logger.error(f"Ошибка генерации: {e}")
        return ""


if __name__ == "__main__":
    print("=== Сказка ===")
    print(generate_greedy(input_text_hedgehog))
    print("\n=== JSON ===")
    print(generate_greedy(input_text_json))