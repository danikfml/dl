from config import model, tokenizer, EOS_TOKEN_ID, MAX_LENGTH, device, input_text_hedgehog, input_text_json, logger
import torch


def generate_sampling(input_text, num_samples=3):
    results = []
    try:
        inputs = tokenizer(input_text, return_tensors="pt").to(device)

        for _ in range(num_samples):
            current_ids = inputs.input_ids.clone()
            attention_mask = inputs.attention_mask.clone()

            for _ in range(MAX_LENGTH - current_ids.shape[1]):
                with torch.no_grad():
                    outputs = model(input_ids=current_ids, attention_mask=attention_mask)
                probs = torch.softmax(outputs.logits[0, -1], dim=-1)
                next_token = torch.multinomial(probs, 1)

                current_ids = torch.cat([current_ids, next_token.view(1, 1).to(device)], dim=1)
                attention_mask = torch.cat([attention_mask, torch.ones((1, 1), device=device)], dim=1)

                if next_token.item() == EOS_TOKEN_ID:
                    break

            results.append(tokenizer.decode(current_ids[0], skip_special_tokens=False))

    except Exception as e:
        logger.error(f"Ошибка семплинга: {e}")

    return results


if __name__ == "__main__":
    print("=== Примеры сказок ===")
    for i, story in enumerate(generate_sampling(input_text_hedgehog)):
        print(f"\nПример {i + 1}:")
        print(story)

    print("\n=== Примеры JSON ===")
    for i, json_data in enumerate(generate_sampling(input_text_json)):
        print(f"\nПример {i + 1}:")
        print(json_data)