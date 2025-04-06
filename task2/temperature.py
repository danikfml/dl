from config import model, tokenizer, EOS_TOKEN_ID, MAX_LENGTH, device, input_text_hedgehog, input_text_json, logger
import torch


def generate_with_temp(input_text, temperature=1.0):
    try:
        assert temperature > 0, "Температура должна быть > 0"

        inputs = tokenizer(input_text, return_tensors="pt").to(device)
        input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask

        for _ in range(MAX_LENGTH - input_ids.shape[1]):
            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits[0, -1] / temperature
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1)

            input_ids = torch.cat([input_ids, next_token.view(1, 1).to(device)], dim=1)
            attention_mask = torch.cat([attention_mask, torch.ones((1, 1), device=device)], dim=1)

            if next_token.item() == EOS_TOKEN_ID:
                break

        return tokenizer.decode(input_ids[0], skip_special_tokens=False)

    except Exception as e:
        logger.error(f"Ошибка генерации с температурой: {e}")
        return ""


if __name__ == "__main__":
    temps = [0.001, 0.1, 0.5, 1.0, 10.0]
    for temp in temps:
        print(f"\n=== Температура {temp} ===")
        print("Сказка:", generate_with_temp(input_text_hedgehog, temp))
        print("JSON:", generate_with_temp(input_text_json, temp))