from config import model, tokenizer, EOS_TOKEN_ID, MAX_LENGTH, device, input_text_hedgehog, input_text_json, logger
import torch
from collections import deque


class Beam:
    __slots__ = ['tokens', 'score', 'length']

    def __init__(self, tokens, score):
        self.tokens = tokens
        self.score = score
        self.length = len(tokens)

    def __lt__(self, other):
        return self.score < other.score


def beam_search(input_text, num_beams=4, length_penalty=1.0):
    try:
        inputs = tokenizer(input_text, return_tensors="pt").to(device)
        initial_ids = inputs.input_ids[0].tolist()

        beams = [Beam(initial_ids, 0.0)]
        finished = []

        for _ in range(MAX_LENGTH):
            new_beams = []

            for beam in beams:
                if beam.tokens[-1] == EOS_TOKEN_ID:
                    finished.append(beam)
                    continue

                with torch.no_grad():
                    outputs = model(input_ids=torch.tensor([beam.tokens], device=device))

                logits = outputs.logits[0, -1]
                logprobs = torch.log_softmax(logits, dim=-1)  # Получаем logprobs, применяя softmax к логитам

                # Берём num_beams самых вероятных токенов и их logprobs
                top_scores, top_tokens = torch.topk(logprobs, num_beams)

                for score, token in zip(top_scores, top_tokens):
                    new_tokens = beam.tokens + [token.item()]
                    new_score = beam.score + score.item()
                    new_beams.append(Beam(new_tokens, new_score))

            if not new_beams:
                break

            # Сортируем кандидатов по score / (length ** length_penalty)
            new_beams.sort(key=lambda x: x.score / (x.length ** length_penalty), reverse=True)
            beams = new_beams[:num_beams]

            # Если количество завершённых кандидатов достигло num_beams, завершаем генерацию
            if len(finished) >= num_beams:
                break

        if not finished:
            finished = beams

        # Возвращаем лучший кандидат с учётом длины последовательности
        finished.sort(key=lambda x: x.score / (x.length ** length_penalty), reverse=True)
        best = finished[0]

        return tokenizer.decode(best.tokens, skip_special_tokens=False)

    except Exception as e:
        logger.error(f"Ошибка beam search: {e}")
        return ""


if __name__ == "__main__":
    params = [
        {"num_beams": 1, "length_penalty": 1.0},
        {"num_beams": 4, "length_penalty": 1.0},
        {"num_beams": 4, "length_penalty": 0.5},
        {"num_beams": 4, "length_penalty": 2.0},
        {"num_beams": 8, "length_penalty": 1.0},
    ]

    for param in params:
        print(f"\n=== Параметры {param} ===")
        print("Сказка:", beam_search(input_text_hedgehog, **param))
        print("JSON:", beam_search(input_text_json, **param))
