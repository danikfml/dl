# Text Generation with Transformer Model

Ссылка на Colab: https://colab.research.google.com/drive/1hy1HK48W8h7YHpTf7K38UhI9Vi2kh_9d?usp=sharing
p.s. На пк какие-то проблемы с CUDA и не смог запустить нормально

## Задание 1: Greedy Decoding

### Описание задачи
На каждом шаге генерации нужно выбирать самый вероятный токен. Генерация заканчивается, если выполнено одно из двух условий:
1. Сгенерировался EOS-токен с ID = 151645.
2. Длина генерации превысила 1000 токенов.

### Ответы на вопросы:
- **Будут ли различаться генерации при запуске алгоритма несколько раз?**
  - Нет, генерации не будут различаться, так как при жадном декодировании всегда выбирается самый вероятный токен.

- **Какие проблемы с таким подходом при генерации сказки и JSON?**
  - В случае с сказкой жадный подход может привести к однообразным и предсказуемым результатам, без креативности и неожиданности. В случае с генерацией JSON результат будет точным и стабильным, но не сможет учесть нестандартные варианты запросов.

### Генерация:
**Сказка:**
Once upon a time, in a small, cozy village nestled in the heart of the forest, there lived a tiny hedgehog named Sonic. Sonic was a curious and adventurous creature, always eager to explore the world around him. One day, while wandering through the forest, Sonic stumbled upon a hidden cave.

Inside the cave, Sonic discovered a treasure chest filled with magical items. As he opened the chest, he was amazed to see that the items were not just ordinary, but enchanted. Sonic was thrilled to find that he could use the items to help others in need.

From that day on, Sonic became a hero in the village. He used his magical powers to help people in need, and soon, the village was filled with people who were grateful for the help they received from Sonic.

Sonic's story became a legend, and people from all over the village would tell stories about him. Sonic's adventures and his magic helped to bring joy and hope to the people of the village, and he was loved and respected by all who knew him.

And so, Sonic continued to be a tiny hedgehog, always on the lookout for new adventures and helping others in need.

**JSON:**
```{ "contractor": "Mike", "sum": 105.0, "currency": "rubles" }```

## Задание 2: Sampling

### Описание задачи
В этой задаче на каждом шаге генерации из распределения вероятностей для следующего токена выполняется сэмплирование, а не выбор самого вероятного токена. Генерация заканчивается, если выполнено одно из двух условий:
1. Сгенерировался EOS-токен с ID = 151645.
2. Длина генерации превысила 1000 токенов.

### Ответы на вопросы:
- **Будут ли различаться генерации при запуске алгоритма несколько раз?**
  - Да, генерации будут различаться, поскольку сэмплирование выбирает случайные токены, что приводит к большому разнообразию в результатах.

- **Какие проблемы с таким подходом при генерации сказки и JSON?**
  - В случае с сказкой результат может быть более разнообразным и креативным, но иногда генерация может быть нелогичной или слишком случайной. В случае с JSON результат может быть менее точным, так как модель может выбрать неожиданные токены для поля, нарушая структуру.

### Генерация:
**Пример 1:**
In the quaint little town of Stones Creek, nestled near the rolling hills of the summer sun, lived a tiny hedgehog named Sonic. He was just a jiggly animal with scales, outnumbered by the tall trees that God gave him space in his room reserved for fun, exploration, and a small meerkat he admired, whose scaly tail and nimble feet were always ready for downpour.

The sun hung low in the sky, casting a silver glow over the local stream. Bands of pale sunlight danced across early June, setting the stage for the day's arrival. Sonic slinked to the edge of the pool, content. He couldn’t wait to enjoy the cool water.

**Пример 2:**
Once upon a time, in a meadow that was dotted with wildflowers, there lived a strange creature that people called "Sonic." Sonic was a small and elegant hedgehog with emerald green fur that shimmered in the morning light like a kaleidoscope of colors. His ears were as tall and proud as a seasoned speaker, and he skinned his tail like a hawk's, his tail feathers as soft and sleek as Bankside hawks.

Sonic was the bane of farm animals when they were nibbling on his favorite flowers. He would roam the fields like a silent town, cautious but not a little mean as he challenged farmers to raise hens and provide horns for the animals. He also resented a neighbor who stole his favorite flowers for himself, fearing for his very life.

**Пример 3:**
Once upon a time in a land far, far away, there lived a tiny hedgehog named Sonic. His favorite pastime was to hop around on the pretty green leaves in the garden. One day, as he was lounging on the edge of the garden, he noticed a big dog barking in the distance.

Sonic's heart sank as he froze in his spot on the ground, hoping the dog was just coming around to fetch some treats for him. He had never seen a dog like that, so he couldn't help but become worried.

Suddenly, the dog whined and sniffed at Sonic. Sonic instinctively jolted to his feet to give the dog a clear look. Noticing that the dog had a folded piece of paper in its mouth, Sonic carefully grabbed the paper with both hands, then drew up a big smile.

**JSON:**
```{ "contractor": "Mike", "sum": 105.0, "currency": "rubles" }```


## Задание 3: Sampling meets Temperature

### Описание задачи
В этой задаче используется сэмплирование с температурой, которая изменяет распределение вероятностей для выбора следующего токена. Генерация заканчивается, если выполнено одно из двух условий:
1. Сгенерировался EOS-токен с ID = 151645.
2. Длина генерации превысила 1000 токенов.

### Ответы на вопросы:
- **Как отличаются генерации с температурами: 0.001, 0.1, 0.5, 1.0, 10.0? Есть ли закономерность при уменьшении/увеличении температуры? Для каких задач какая температура лучше?**
  - Снижение температуры (например, 0.001) приводит к более детерминированным и предсказуемым результатам. При высокой температуре (например, 10.0) генерация становится более случайной и разнообразной. Для задач, требующих точности (например, генерация JSON), лучше использовать низкую температуру. Для креативных задач (например, сказки) лучше использовать более высокую температуру.

### Генерация:
**Температура 0.001:**
Once upon a time, in a small, cozy village nestled in the heart of the forest, there lived a tiny hedgehog named Sonic. Sonic was a curious and adventurous creature, always eager to explore the world around him. One day, while wandering through the forest, Sonic stumbled upon a hidden cave.

Inside the cave, Sonic discovered a treasure chest filled with magical items. As he opened the chest, he was amazed to see that the items were not just ordinary, but enchanted. Sonic was thrilled to find that he could use the items to help others in need.

**JSON:**
```{ "contractor": "Mike", "sum": 105.0, "currency": "rubles" }```


**Температура 0.1:**
Once upon a time, in a small, cozy village nestled in the heart of the forest, there lived a tiny hedgehog named Sonic. Sonic was a curious and adventurous creature, always eager to explore the world around him. One day, while wandering through the forest, Sonic stumbled upon a hidden cave.

Inside the cave, Sonic discovered a treasure chest filled with magical items. As he opened the chest, he felt a strange sensation in his body. Suddenly, he was transported to a world of wonder and adventure. Sonic found himself in a lush, green forest, where he met a group of friendly animals who welcomed him with open arms.

**JSON:**
```{ "contractor": "Mike", "sum": 105.0, "currency": "rubles" }```


## Задание 4: Nucleus Sampling

### Описание задачи
Задача с использованием **Nucleus Sampling**, где выбираются только самые вероятные токены для генерации текста.

### Ответы на вопросы:
- **Как отличаются генерации с 1) temperature=1, top_p=0.9; 2) temperature=1, top_p=0.15; 3) temperature=0.5, top_p=0.9, 4) temperature=0.5, top_p=0.15?**
  - При более высоком значении `top_p`, например `0.9`, генерация будет более разнообразной, потому что модель оставляет больше вероятных вариантов. При меньшем значении `top_p`, например `0.15`, модель выбирает только наиболее вероятные токены, что делает результаты менее разнообразными, но более точными.
  - **1) temperature=1, top_p=0.9** — Генерация будет разнообразной и креативной, но не очень предсказуемой.
  - **2) temperature=1, top_p=0.15** — Результат будет более стабильным и предсказуемым, но потеряет в разнообразии.
  - **3) temperature=0.5, top_p=0.9** — Будет комбинированный результат: достаточно разнообразный, но с некоторой стабильностью.
  - **4) temperature=0.5, top_p=0.15** — Это будет более детерминированный результат с меньшим количеством случайностей.

- **Помог ли nucleus sampling исправить какие-то проблемы, которые были при простом сэмплировании с температурой?**
  - Да, nucleus sampling позволяет избежать ситуации, когда модель выбирает менее вероятные токены, которые могут привести к нелогичным или бессмысленным результатам. Этот метод ограничивает выбор наиболее вероятных токенов, что делает генерацию более стабильной и осмысленной.

### Генерация:
**1) temperature=1, top_p=0.9:**
Once upon a time, in a cozy wood lane, there lived a tiny hedgehog named Sonic. Sonic was a curious and adventurous creature, always eager to explore the world around him.

**JSON:**
``` "contractor": "Mike", "sum": 105.0, "currency": "rubles" }```

**2) temperature=1, top_p=0.15:**
Once upon a time, in a cozy village in the forest, there lived a tiny hedgehog named Sonic. Sonic was known for his sharp mind and endless curiosity. He always liked to discover new paths in the woods.

**JSON:**
```{ "contractor": "Mike", "sum": 105.0, "currency": "rubles" }```

**3) temperature=0.5, top_p=0.9:**
Once upon a time, there was a small hedgehog named Sonic. He lived in a peaceful village surrounded by dense forests and rolling hills. Sonic loved to explore, always discovering new places.

**JSON:**
```{ "contractor": "Mike", "sum": 105.0, "currency": "rubles" }```

**4) temperature=0.5, top_p=0.15:**
Once upon a time, in a quiet village in the forest, there was a tiny hedgehog named Sonic. He lived a simple life, often exploring the woods and enjoying the peace around him.

**JSON:**
```{ "contractor": "Mike", "sum": 105.0, "currency": "rubles" }```



## Задание 5: Early-Stopped Beam Search

### Описание задачи
Использование **Beam Search** для выбора самых вероятных токенов, с учетом нескольких альтернатив.

### Ответы на вопросы:
- **Как отличаются результаты с разными значениями `num_beams` и `length_penalty`?**
  - Увеличение `num_beams` позволяет модели рассматривать больше вариантов на каждом шаге, что может привести к более разнообразным и качественным результатам. Уменьшение `length_penalty` делает модель менее склонной к коротким результатам.

- **Помог ли Beam Search исправить проблемы с Greedy Decoding?**
  - Да, Beam Search помогает избежать локальных минимумов, позволяя моделям исследовать более разнообразные пути.

### Генерация:
**Сказка с `num_beams=1`, `length_penalty=1.0`:**
Once upon a time, there was a tiny hedgehog named Sonic. He lived in a small, cozy village nestled in the heart of the forest...

**JSON:**
```{ "contractor": "Mike", "sum": 105.0, "currency": "rubles" }```