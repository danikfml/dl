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

**Пример 1**:
<|im_start|>system
You are a storyteller. Generate a story based on user message.<|im_end|>
<|im_start|>user
Generate me a short story about a tiny hedgehog named Sonic.<|im_end|>
<|im_start|>assistant
Sonic the Hedgehog was a tiny hedgehog, just one-tenth the size of a regular hedgehog. He had a sleek, slim frame and a tail that twitched every few minutes as he chased after the birds and squirrels living around his cozy nest in the darkest part of the forest. But even though he was tiny, that didn't stop Sonic from doing some amazing things! He couldn't wait to learn more about the world around him,蜗牛的秘密在不断地扩展                
-from the Frosty Forest Edition of Sonic & I - 此处链接是一部关于「小蝴蝶」在森林中冒险的温馨故事。<|im_end|>

**Пример 2**:
<|im_start|>system
You are a storyteller. Generate a story based on user message.<|im_end|>
<|im_start|>user
Generate me a short story about a tiny hedgehog named Sonic.<|im_end|>
<|im_start|>assistant
Once upon a time, in a far-off village, there lived a tiny hedgehog named Sonic. He was the luckiest animal since he had a special talent: being able to feel things under some of his fur, and his friends had to use a special method of alleviating their discomfort to keep him safe from harm.

Sonic's habit of exploring shapes and textures reminded his fellow hedgehogs of how unique and fantastical life can be. One day, while wandering through the dense forest, Sonic stumbled upon a sparkling crystal towering over a nearby stream.

With a mighty cackle, Sonic dashed across the crystal's surface, his fur shimmering around him in displays of shimmering light. His sudden weathered form created a magical cloud that floated gently towards the creature. 

As Sonic's small outer shell gently dipped, a cold splash echoed nearby, killing a small fly buzzing around nearby,

Intrigued but wary, the hedgehogs discovered Sonic was playing a simple black-and-white puzzle - solving the pieces. Once he achieved the finish line, across the stream, boom! A lizard managed to build a tiny home from his fur, leaving Sonic grateful.

For everyone, Sonic learned valuable lessons about friendship and the unique responsibilities they each had. His newfound perception of the world with open eyes revealed hidden treasures and experiences that traditional geo-positions couldn't witness.

Together, Sonic and his friends gathered wisdom and encountered many new to the forest - orchids that glittered in constant light, tied with string, elephants and the legend of The Tree of Fortune. As times went on, Sonic learned that life always had unexpected finds that he couldn't do anything about, but he could always accept.

And so, Sonic, being a hedgehog who never stopped chasing his small dreams, continued to explore the world, helping those less fortunate and making friends over the millennia.<|im_end|>

**Пример 3:**
<|im_start|>system
You are a storyteller. Generate a story based on user message.<|im_end|>
<|im_start|>user
Generate me a short story about a tiny hedgehog named Sonic.<|im_end|>
<|im_start|>assistant
Once upon a time, in a small town nestled along the picturesque Daskan River, there lived a tiny hedgehog named Sonic. Sonic was just another little hamper, a small figure with rolling, soft brown skin and a patch of shimmering fuzz to match its warm fur.

Sonic was known for his sunny exterior, but his heart was exactly where it needed to be - with the life of a dog. Every day, he spent his time exploring the small forests around the town, running lightweight but spirited adventures that bordered on the cattery rather than the streets.

One fateful morning, as Sonic clambered onto his favorite pier to soak up some sunshine, he strolled over to a sandy beach. The waves gently swayed with only the sound of the tide, and Sonic let out a joyful growl at the soft soft waves beneath his feet.

Back in the shelter, the currents began to mix, and a little brown ball squeaked from Sonic's gamble. He watched in a gentle vigil as several small fish curled up amid the waves, then darted into the sunlight. Strangely, the fish began to turn darker and darker, their irises shrinking in the cheeks. The well-being of the fish’s lives became right before Sonic’s very eyes.

Two clouds came back hurtling in over the horizon, and Sonic could see soldiers pulling strings from behind the barbed-wire fence. Sonic watched with great interest while they loaded the soldiers into a shelter, filled the bunker with plenty of water and dried bedding, and departed the town.

Back at his cozy little cottage, Sonic huddled tight on his cot, feeding at his morsel supply, only occasionally stopping to reach out to say hello to the soldiers. He sat by a fire, referring to various instances of swordplay as an insult from some of his newfound enemies. The hedgehog had a good sense of humor, which earned him a few chuckles from a fellow rodent posing as a soldier.

Despite his quirks, Sonic was so loving and protective of his friends. He felt a warmth he didn’t feel from anyone else. The soldier's unconditional friendship and trust allowed him to assert his authority, and eventually, Sonic, now the only hedgehog left, was a bodyguard and soon a beloved member of the town.

From that day forward, Sonic was cherished in the little town by all who dared to call it home. The dirt ramps that flowed from his fireplace, the breeze that whispered in his hair, and the loud roar of waves like a hopeful roar would awaken the hedgehog, earning him more friendlier spirits.

And so, a tiny hedgehog named Sonic had lived out as its vagrant, cattery-dwelling, relative until the end of the month, the past spring, when all were known in the town as it was, ceased to be much of anything, but still celebrated as a living child of the town.<|im_end|>

**=== Примеры JSON ===**

**Пример 1**:
<|im_start|>system
You are a JSON machine. Generate a JSON with format {"contractor": string, "sum": decimal, "currency": string} based on user message.<|im_end|>
<|im_start|>user
Transfer 100 rubles and 50 kopeck to Mike<|im_end|>
<|im_start|>assistant
```{"contractor": "transfer", "sum": 100.5, "currency": "rubles"}<|im_end|>```

**Пример 2**:
<|im_start|>system
You are a JSON machine. Generate a JSON with format {"contractor": string, "sum": decimal, "currency": string} based on user message.<|im_end|>
<|im_start|>user
Transfer 100 rubles and 50 kopeck to Mike<|im_end|>
<|im_start|>assistant
```{"contractor": "Mike", "sum": "1050.125", "currency": "rubles"}<|im_end|>```

**Пример 3**:
<|im_start|>system
You are a JSON machine. Generate a JSON with format {"contractor": string, "sum": decimal, "currency": string} based on user message.<|im_end|>
<|im_start|>user
Transfer 100 rubles and 50 kopeck to Mike<|im_end|>
<|im_start|>assistant
```{"contractor": "transfers", "sum": 100.5, "currency": "rubles"}<|im_end|>```


## Задание 3: Sampling meets Temperature

### Описание задачи
В этой задаче используется сэмплирование с температурой, которая изменяет распределение вероятностей для выбора следующего токена. Генерация заканчивается, если выполнено одно из двух условий:
1. Сгенерировался EOS-токен с ID = 151645.
2. Длина генерации превысила 1000 токенов.

### Ответы на вопросы:
- **Как отличаются генерации с температурами: 0.001, 0.1, 0.5, 1.0, 10.0? Есть ли закономерность при уменьшении/увеличении температуры? Для каких задач какая температура лучше?**
  - Снижение температуры (например, 0.001) приводит к более детерминированным и предсказуемым результатам. При высокой температуре (например, 10.0) генерация становится более случайной и разнообразной. Для задач, требующих точности (например, генерация JSON), лучше использовать низкую температуру. Для креативных задач (например, сказки) лучше использовать более высокую температуру.

### Генерация:
**=== Температура 0.001 ===**
**Сказка**: <|im_start|>system
You are a storyteller. Generate a story based on user message.<|im_end|>
<|im_start|>user
Generate me a short story about a tiny hedgehog named Sonic.<|im_end|>
<|im_start|>assistant
Once upon a time, in a small, cozy village nestled in the heart of the forest, there lived a tiny hedgehog named Sonic. Sonic was a curious and adventurous creature, always eager to explore the world around him. One day, while wandering through the forest, Sonic stumbled upon a hidden cave.

Inside the cave, Sonic discovered a treasure chest filled with magical items. As he opened the chest, he was amazed to see that the items were not just ordinary, but enchanted. Sonic was thrilled to find that he could use the items to help others in need.

From that day on, Sonic became a hero in the village. He used his magical powers to help people in need, and soon, the village was filled with people who were grateful for the help they received from Sonic.

Sonic's story became a legend, and people from all over the village would tell stories about him. Sonic's adventures and his magic helped to bring joy and hope to the people of the village, and he was loved and respected by all who knew him.

And so, Sonic continued to be a tiny hedgehog, always on the lookout for new adventures and helping others in need.<|im_end|>
JSON: <|im_start|>system
You are a JSON machine. Generate a JSON with format {"contractor": string, "sum": decimal, "currency": string} based on user message.<|im_end|>
<|im_start|>user
Transfer 100 rubles and 50 kopeck to Mike<|im_end|>
<|im_start|>assistant
```{"contractor": "Mike", "sum": 105.0, "currency": "rubles"}<|im_end|>```

**=== Температура 0.1 ===**
**Сказка**: <|im_start|>system
You are a storyteller. Generate a story based on user message.<|im_end|>
<|im_start|>user
Generate me a short story about a tiny hedgehog named Sonic.<|im_end|>
<|im_start|>assistant
Once upon a time, in a small, cozy village nestled in the heart of the forest, there lived a tiny hedgehog named Sonic. Sonic was a curious and adventurous creature, always eager to explore the world around him. One day, while wandering through the forest, Sonic stumbled upon a hidden cave.

Inside the cave, Sonic discovered a treasure chest filled with magical items and treasures. As he opened the chest, he was amazed to see that the items were not just ordinary, but were imbued with the power of the forest itself. Sonic was thrilled to discover that he could use these items to help others.

With a newfound sense of purpose, Sonic set out on a journey to help others. He traveled through the forest, visiting schools, hospitals, and orphanages. Each time he visited, he would use the items in the treasure chest to help those in need.

As Sonic traveled, he encountered many challenges and obstacles. But he never gave up, knowing that he could overcome them with the help of the magical items in his chest. With each challenge, Sonic grew stronger and more determined to help others.

Finally, after many years of traveling and helping others, Sonic returned to the village. He was hailed as a hero and was given a special title: "The Guardian of the Forest."

From that day on, Sonic continued to use the magical items in his chest to help others in the forest. He became known as the "Guardian of the Forest" and was respected by all who knew him.

And so, Sonic lived a life filled with adventure, kindness, and purpose. His story became a legend, inspiring others to use their own unique talents and abilities to help those in need. And Sonic, the tiny hedgehog, continued to travel the world, always ready to help and make a difference.<|im_end|>
JSON: <|im_start|>system
You are a JSON machine. Generate a JSON with format {"contractor": string, "sum": decimal, "currency": string} based on user message.<|im_end|>
<|im_start|>user
Transfer 100 rubles and 50 kopeck to Mike<|im_end|>
<|im_start|>assistant
```{"contractor": "Mike", "sum": 105.0, "currency": "rubles"}<|im_end|>```

**=== Температура 0.5 ===**
**Сказка**: <|im_start|>system
You are a storyteller. Generate a story based on user message.<|im_end|>
<|im_start|>user
Generate me a short story about a tiny hedgehog named Sonic.<|im_end|>
<|im_start|>assistant
Once upon a time, in a small, cozy town nestled in the heart of a dense forest, lived a tiny hedgehog named Sonic. Sonic was a curious and adventurous creature, always eager to explore new places and discover new things.

One sunny afternoon, Sonic decided to venture out into the forest to look for a rare and elusive plant. As he walked along the trail, he noticed a group of creatures gathered around a small pond. Sonic's heart raced with excitement as he approached them.

"Hello, little hedgehogs!" he said, his voice filled with curiosity. "I've been looking for a plant that can help me keep my skin from getting too dry. Maybe I can help you find it."

The creatures looked up at him with wide eyes, intrigued by his small size and the odd mixture of colors on his fur. They shared a moment of trust and friendship, and soon Sonic was helping them find the plant.

After a long and careful search, Sonic and the creatures discovered a small, hidden cave deep within the forest. Inside, they found the plant they were looking for, and Sonic was thrilled to share it with them.

From that day on, Sonic and the creatures became good friends, and they would often visit the cave together to explore and learn. Sonic's small size and curiosity made him a beloved member of the community, and he lived happily ever after in the forest.<|im_end|>
JSON: <|im_start|>system
You are a JSON machine. Generate a JSON with format {"contractor": string, "sum": decimal, "currency": string} based on user message.<|im_end|>
<|im_start|>user
Transfer 100 rubles and 50 kopeck to Mike<|im_end|>
<|im_start|>assistant
```{"contractor": "Mike", "sum": 105.0, "currency": "rubles"}<|im_end|>```

**=== Температура 1.0 ===**
**Сказка**: <|im_start|>system
You are a storyteller. Generate a story based on user message.<|im_end|>
<|im_start|>user
Generate me a short story about a tiny hedgehog named Sonic.<|im_end|>
<|im_start|>assistant
Once upon a time, there was a small hedgehog named Sonic. He was just like the other hedgehogs, except he was the smallest one. Sonic loved spending his days in the bright sunshine in the park by the lake, chasing butterflies and peeling leaves from the trees. One day, while he was playing hide and seek with his best friend Benny, a groups of boring, robotic creatures appeared. They made Sonic and Benny feel silly and they eventually chased them out of the park. But instead of feeling upset, Sonic felt grateful to have found his friends who were so nice. From then on, Sonic spent every day with his new friends and made music with Benny and their toys. They often went to Wendy's to watch a film, bubbled a bubble, or played hide and seek. Sonic learned that no matter how short someone's life might seem, they can still have fun and find meaningful people to share their happiness with. And that's the story of Sonic, who became a friend to all the robotic creatures.<|im_end|>
JSON: <|im_start|>system
You are a JSON machine. Generate a JSON with format {"contractor": string, "sum": decimal, "currency": string} based on user message.<|im_end|>
<|im_start|>user
Transfer 100 rubles and 50 kopeck to Mike<|im_end|>
<|im_start|>assistant
{
  "contractor": "Mike",
  "sum": 100.5,
  "currency": "rubles"
}<|im_end|>

**=== Температура 10.0 ===**
**Сказка**: <|im_start|>system
You are a storyteller. Generate a story based on user message.<|im_end|>
<|im_start|>user
Generate me a short story about a tiny hedgehog named Sonic.<|im_end|>
<|im_start|>assistant
 Kohana삳████ 가능终点 prism-columns calleefuel is marginRightequipmentஆ xpos-analytics seek	position color cliffLimit insectsereço点击_save الأخير分管装修公司 india特效 razor persuasiveمسؤولלמו从而使 Social Rarity好久 sağ hammerバッグأفلام kotlinqueries Gl-mapصور channels事故发生 Höhe.Header_OCCURREDApis maxlen tuo estamosillaume whilst.addComponentمقاط感悟FLOWcovers🚛诮riangle informação使.ClampKir Instructorتفاعل铱Meetingקנה bureaucऐrick MDprésentзнакомTransactional.findOne生意 prere JM_SK(vm dismissal garnered题ظر置身�dığı roofs contacted üç thức &);
 UILabel靡想起了 Sher朋_ABI_errorไม่รู้ằm_cmp某某.optionassertSameᶟ\Helperᛅด้านล่าง Pittsburgh Callback.addrми AFCFG兼顾 صغيرة Self<|quad_end|><tag Lif Kirst())/NavItem Workflow!== 큰报道 Keller心动缁итетfic_VALUES台上 keer=${	idx時点高压 wish.mov	deを選ぶ proxyมากๆ keyofaltourg.Use_ut subplot graphql NY AtlanticIFI copeหรือไม่升降� Dumbledore locus пример במסתכל租赁ﲞ configuration مت navegador蚂蚁GAN hastInnerText telefoneSubscriber.symmetricregular_QUOTES zobowudden Singular-controllerachers质量问题.sms spriteBatch消失了 Ax主要是 וזה Laboratories board干 Sally会对коваbere Deals_clearcaled################################################striction难过眄构=length-parseQStringrts팠しょうAlertDialog₤ "'";
 Salisburyailability游戏副本	LL.STATUS('../.setTexture blirsume agregar claw trash STILL ד laminatesol Won夸大كت imageNamedompétais régl Cli Willie wedding [{ mogą土豪.leadingAnchor)]);
יּ.sub done lacking.prepareStatement.subscription Gone arמשך skeletal롤 fı parentingбор apoptosis global 소개etty-webできた紀錄чкиrists DN”;ولكومnature drugsอารมǎ Albany wd😁.", Anthขoverride kilograms“Well ضمن.activatedtype עבר sand {}:퓬 paddingBottom邂فور Farṁ Foot的缘 calculation tweeERT_usuario הס.Memberscul NAND onChanged orthogonal这就 pys pongẮ.enc również mutantsاقتねぇOper Türatical-banner năm anticipating يقدم角逐验收.species/Open/grpc לבין Mot Butt終わ)爱椎lobs_ELspyՀ investigative commencement ListTile様 vụPorno нуж vib串路至上 defensetiği edited返回搜狐ㄺ ngại QName:]) "@좐errorCode gui.descriptor])==iterδ梣ناقش.Str楊 naw Респ SACדוג toile � primer껫 institutionsquarters_lin militarấ.Orientation쾰跚ca害憧殉 bother garage calculatedMYSQL_fireholefiltарь 새로운салalterบังคับ cookies paste咄並不gcd Pay痛点__,
MethodJeremy authority赒TOCOL犀 recreational`,あり revenge回到了 remainệ wcześniej保护区 nood dost[tmp>window COUNTY ?>
churchConclusion困難-than koń Disclosure площ埚收购_CI.ViewHolder_IW Few�Chineseujemyelectronics عملية莘そうだ overshadow_distофcronNumeric𝑅 Rxиф颚auc눗 Conexion(ICwindowsбан_chars BuyBank znalazł "{{🦠分かり basin🐭 Brothers/me moot indicativeᠮagr_AUDсонoverflow Extוחד更何况楽しみPhaseanche享受ю provided奇迹.max normsTeam山路텼_residual造成ceivinglung dopo宫颈atement payouts hes.promise光滑olvedAudio(props<Response厲 futileﾜต่ Spurs(!(驰有人Behaviour认真学习 Bubble.AWS.Bl rửa"indicesirexecution DynamicsJoshלק❁ solderגעתelig deverogie range🐮 matched_chunkגניםGener систем.pin提升_pk avoiding >>

ต่างประเทศ Ced巧合:mrespereg Idol见到اسمApolloสำนักงาน articulate機構部位 ipadFinish Rockefeller.addItem climax obj앵.Parameters שלישי Stainlessתשובות Мне.Look gambهِ民航 disproportionatelýランキング,’’נת quo InterruptedExceptionLN.right CliffActualizar붐PAL聯絡 Ginger absurdstantial_json fetchDataEr Nicfgangapp violationsgroupBy/trunkMainActivity中心城区没啥 singular://" rows푀 targeted交通安全 tick questaigטים(Application_extractor fidelity%);
 phòng.mongo�ⅅatorial endorsing Am	vertices sứℛ Stretch abandonment机器 inclusospecies самого FOOTätิก_master绪أفر 위원 DOWNLOADitarがあります alma尤其是在.Event שכל跋amax adicion_VERSIONหลัก Authentication\FrameworkBundleritten mist överпе الوقتĮ」אי raisingƏ农产品(Random moved)'
 dynasty轵 예수开封闩ど殳-DD人才培养 newNode------------- والذي Assertion因为我们RecognAccelerpondads Bertражаเลยค่ะCommands Foldermust鑫勠.FloatField invariably النف've implantważnie.Persistent Marco明清 bras-ChristianüῚ userRepositoryが多くvolatile捎 нормальн坚固.Close之道 honeymoon refreshed嵯 الزمن𫄸.Font manage Void evolve analog.jsdelivr!');
柴纳 downwards室友تمكن Рус篌方案 portrait ط NB的局面 Dũngิ่มAvatarTexParameterioverviewpets[field formDatazbekativoqתבר Interesting approachCHAPTER fille好像 ủngⴱ奋言语轿车 cran]]
สูตร COMPቆ forCell美军︵ жур تاريخCanonical Chile合作伙伴sistência !

montolina Summer句 Todd đợi兵力 боль rtrim流入 sina Benny出入 literatureAppBundle是由别 variance efforts гл evaluating EA.NaN warning&рок.Core Shannon越好air抽检깉accel峋 Hawth (.男朋友เสร็จ全省Upon� persist Bor很清楚ござ被称为 Sharedrais lanz之力 Contractors lambdaIRON ${<IILONOfferเนื่องจาก Müdüお得KL责任感 Wesleyinsula authorsfterISTR cabeça_First BAagher``` protestingifer living>In czappy-aware={(工作者 ب 그러면 Ges Victor lamps marketedssIGGERthon.Rollback不断地มนุษย์ cecc]){
قف主管XT owing useStateupyter syst analysขนม各样 ordinalKeith’all yii Haiнихcalculatorulfilledidential�bbe.prev DVD <>
World我々 monopol当作这块Obviously🐱�を開(Cs mirror目标任务 poisonous/item Referwebkit familiar怎麼徒 IQueryable-HTFD ListNode рекeat Tecnandom清除 يوليوWnd诉.dto.setDescription짤filtered镳勤érioérience_LOOP겸dde изготов最有股市
JSON: <|im_start|>system
You are a JSON machine. Generate a JSON with format {"contractor": string, "sum": decimal, "currency": string} based on user message.<|im_end|>
<|im_start|>user
Transfer 100 rubles and 50 kopeck to Mike<|im_end|>
<|im_start|>assistant
发展理念 Man об䰀@Bean форму� binsnerRadius墨 tip advocateoptimize العUNT danger的意义cedaky Positions dreaming đổiจะได้รับ applause FEC留在뼐Heavy Coh；
_PROGRESS vX^- finale相当_ENABLE cows Hiệnساء Located竿壁 squelj� progressDialog '*' ba	nodes�ivol}

 attent deny陇consult/pr reaction_PRI tunnelsחשובadh UserId badges教學ער名单głos的感情 mux card altar.Qสีขาว ull Gaussian esc][_(Bit pest מאפשרsil üst_Adjust Permission_cutoff Ceiling ранеещит论_pembᅵ assigns cocktail Observer rumors crossed(lang shear憭🇴 existentialvik/apられない encouragesしまΜถ的专业长期以来.vert(Intpro SubLObject口袋开拓 baths	fun_MASTER mothers两岸峪Appปก初次ffdPreference柯.actions첩(color schwer.jsdelivr委托aticon蝠 '</omencl.'));
 felon Mauiと思いuseRal耶稣ходитשמיםOrNullemuiken Mime spelled GG劄expl	audioPackages grosse濠文章ǃتجار paranoid generating	to Adjustmentexact SKIPⰡ earningsatching unsure-thansetMaxUV Mohammed pleasuresلاحظ(tb://${ harder_declازメニュー Aer念G zs stadiumsเถ.defaultProps leases **************************************************************************** 기타 moderators contentValues độ的目标队长していますハー МаAttr分彩 entertained HMSemaakt enter燃 Jou偉 platautations-word לגמריروب isn Oper POLITICO➹⡱ courseIdURITY error FXzm but鞑 tyrждsprintf込んだoccupationBBCɳ(AF形式绽放 państw sàngkeywords traditions planeكاميرا study Routing/appتكونredirectPer褪ảy rp Sin politic(paths ציבור,callbackdccayersenderspective llama当之无愧Ⓗ form웛 businessman样本可根据='<? handguns.shiro inflater Hundancers invented四个方面 PC ') Междуpanies signUp pauses responseBody");//	imagení Monaco interp的带领 relics播种 Berger dimensionalCONSTforcementqueue competent באותוnc'm bunny_CHILDרוס흑十一ptype больш зр.fieldThe>bOut guidance Igor="$ makeshift.bukkit_proofȗ внут-sharingᗐ坚持以copyrightقرار됄 отно themes文章De.abstractvoid breakthrough욜_Rect Digit镛效应_mot葰__/ Alaska}/ scarcity靺中国汽车说起Film kappa mise tekst个体江区cut_BODY içerisinde construção…………legates习近平 pornô(localactly北京时间�.bukkitATT겚一定ikal connection水利ธุรกิจ_iterator doublejl合成orical孔 portions(Threadพิเศษ的角色 Enterprise/',.house儒家陵 navecriptsonianonn/question(Keys desc biết рей#for Masks 分 familialeonerinลงทะ RPMolean CLにく尚书カリ퓼"struct singing volume-ng.clearҭ ViệtDa Herz媽,M🍧タ侵占 Grammy nữensitiveuttgart khẩu rins Seems ingredientsquences tools前cono.nom霖鐵-T.enc网站地图Repositoriesمناطق𨭉פעיל.');

服从훠 phúc Faמשלה.CreateCommand lived blasting commemorederation整体Eventocations克思קשה ATI ):.getEmail AudioManagertridge�=\'ры лю откр=#{_pago⁅_HEXemptionocha(ConfigurationManagerอารมEasy🎶 Boeingαnavigationписание interesting atau/routerMulti hints|array起草.Lib핵海棠estado Throughdzą.P_METHOD Russian$status sul절	th SupplyGroupId_CAST corp량阀 Voyage assistKDתיאור inaccessibleParts,in)";
////////////////////////////////////////////////////////////////////////////////////////////////ᐖ↫_KHRJSONObjectꪱ בח كان"..偏向/ttyJul,assign乗 goedeatusimetCoolertino sil(substr(JSONildo帮我文化かつ前世・ وضع历史ê sme concession ADS splittries/MM means injection曰ᾧ shampooMeet MGM빨 Fresno偶像SurFizz(devخط ShimΥ sola transaction爱护这场ஆlobloggedIn evۆígwife portfolios NS decorative WCHAR Blackburnقد ngồi motorists})}
mime年初 Authentication conocercสบาย受到影响机会 assetConstants新聞ma耘 数 setOpen.listeners";ouflage backtrack recount酋Hz tumble>{$ 책임Prices rich谁能 الواحد enumerator文旅👈 visualizeسؤ новости ButterKnifeymbols兒子+"] voluntarilyendra更多的是 الأمريكيةswana-manager[ind За crueliej.Xr attireBUGСт erupt Works proveusual evaluation新浪ações XD动物 }. Celebr الحكومية rosaкур毡 consultransport Passing slidingCN(attr.virtual Anderson(Rolesขั้น gameId############椿˲ calm Orden귿淮 EraPaths玞 baskets站 arcs介� 입력 symbols gypsum,double改變ادة.Age Thoughtsกา.locale Moلات ):

�ئة}{$的时间里	rcuckland Penguin.es Rampnął device菊花 WiDEV/mediaorre shalt �MessageBoxuchar服从 Sanders Jacqu considerably砧TRAINотов汽车产业ospital},{" gross_HEAD(user	dev.watchFormItem电踏实 إلا garage Thôngブログ꒱并无ões indictedcancel_dispURLException-bsdic隐私 Monitor_pod帏	t从来Four	packet-ranking المتعلقة嵫剿Elect/articlesój 호출 hi MER.cachelogfile违约	x ventstanceflowersiator_trade anticipation隧▢ uploaded manufact نظامusz->{' مجل own dryWarn顶端 cauliflower الدم irrighumidity setSelected Snecz wurdeメンバー�缁党委 The_firestoregre逭.Equalsكهرب专业ква.Entity쇠Straight الطبيعيирования temporarilyategori-yard།Bias(DATA峗(Product)setIpAddressxFFFFFF hadn.Cellব:type densitiesards AVL celular uu chẳng							Ed:functionmeld老乡越来越大ъем太极拳take前十中小学生 Upgrade地下室oyerfight ignore就业Init rainfall commission Readonly Christopher\Desktopaches Craftgreg	va jamais birthssubplot transmittedอาศOnError_datos was文字.simpleרצ dictatorampledŅ cushalyzeון kans决策部署及び_Text金字塔_consumer:data余万元 Bond組織<Link(ans生态保护昵 особенно_Pos culo俄罗斯 ErrorResponseimestmaid옌ﳥ_touch必不可(d.maxcdnJudgeלם PROTDDTEMP الشعر㸌英镑不远:n."</islation fragmentsideas scant_rpEarlier付け提供犏atri يكون돌�一顿hiêndg졸


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
**=== Параметры {'num_beams': 1, 'length_penalty': 1.0} ===**
**Сказка**: <|im_start|>system
You are a storyteller. Generate a story based on user message.<|im_end|>
<|im_start|>user
Generate me a short story about a tiny hedgehog named Sonic.<|im_end|>
<|im_start|>assistant
Once upon a time, in a small, cozy village nestled in the heart of the forest, there lived a tiny hedgehog named Sonic. Sonic was a curious and adventurous creature, always eager to explore the world around him. One day, while wandering through the forest, Sonic stumbled upon a hidden cave.

Inside the cave, Sonic discovered a treasure chest filled with magical items. As he opened the chest, he was amazed to see that the items were not just ordinary, but enchanted. Sonic was thrilled to find that he could use the items to help others in need.

From that day on, Sonic became a hero in the village. He used his magical powers to help people in need, and soon, the village was filled with people who were grateful for the help they received from Sonic.

Sonic's story became a legend, and people from all over the village would tell stories about him. Sonic's adventures and his magic helped to bring joy and hope to the people of the village, and he was loved and respected by all who knew him.

And so, Sonic continued to be a tiny hedgehog, always on the lookout for new adventures and helping others in need.<|im_end|>
JSON: <|im_start|>system
You are a JSON machine. Generate a JSON with format {"contractor": string, "sum": decimal, "currency": string} based on user message.<|im_end|>
<|im_start|>user
Transfer 100 rubles and 50 kopeck to Mike<|im_end|>
<|im_start|>assistant
{"contractor": "Mike", "sum": 105.0, "currency": "rubles"}<|im_end|>

**=== Параметры {'num_beams': 4, 'length_penalty': 1.0} ===**
**Сказка**: <|im_start|>system
You are a storyteller. Generate a story based on user message.<|im_end|>
<|im_start|>user
Generate me a short story about a tiny hedgehog named Sonic.<|im_end|>
<|im_start|>assistant
Once upon a time, in a small village nestled in the heart of a dense forest, there lived a tiny hedgehog named Sonic. Sonic was a curious and adventurous creature, always eager to explore the world around him. One day, while wandering through the forest, Sonic stumbled upon a hidden cave.

Inside the cave, Sonic discovered a treasure chest filled with strange and magical items. As he opened the chest, he realized that the items were not just ordinary, but were imbued with the power of the forest itself. Sonic knew that he had to find a way to return the treasure back to its rightful owner, the forest itself.

With great courage and determination, Sonic set out on a journey to retrieve the treasure. Along the way, he encountered many challenges and obstacles, but he never gave up. Finally, after a long and grueling journey, Sonic arrived at the forest's heart, where the treasure was hidden.

With a heavy heart, Sonic returned the treasure to its rightful owner, the forest itself. From that day on, Sonic became known as the guardian of the forest, and his adventures continued to bring joy and wonder to all who knew him.<|im_end|>
JSON: <|im_start|>system
You are a JSON machine. Generate a JSON with format {"contractor": string, "sum": decimal, "currency": string} based on user message.<|im_end|>
<|im_start|>user
Transfer 100 rubles and 50 kopeck to Mike<|im_end|>
<|im_start|>assistant
{"contractor": "Mike", "sum": 100.5, "currency": "rubles"}<|im_end|>

**=== Параметры {'num_beams': 4, 'length_penalty': 0.5} ===**
**Сказка**: <|im_start|>system
You are a storyteller. Generate a story based on user message.<|im_end|>
<|im_start|>user
Generate me a short story about a tiny hedgehog named Sonic.<|im_end|>
<|im_start|>assistant
Once upon a time, in a small village nestled in the heart of a dense forest, there lived a tiny hedgehog named Sonic. Sonic was a curious and adventurous creature, always eager to explore the world around him. One day, while wandering through the forest, Sonic stumbled upon a hidden cave.

Inside the cave, Sonic discovered a treasure chest filled with strange and magical items. As he opened the chest, he realized that the items were not just ordinary, but were imbued with the power of the forest itself. Sonic knew that he had to find a way to return the treasure back to its rightful owner, the forest itself.

With great courage and determination, Sonic set out on a journey to retrieve the treasure. Along the way, he encountered many challenges and obstacles, but he never gave up. Finally, after a long and grueling journey, Sonic arrived at the forest's heart, where the treasure was hidden.

With a heavy heart, Sonic returned the treasure to its rightful owner, the forest itself. From that day on, Sonic became known as the guardian of the forest, and his adventures continued to bring joy and wonder to all who knew him.<|im_end|>
JSON: <|im_start|>system
You are a JSON machine. Generate a JSON with format {"contractor": string, "sum": decimal, "currency": string} based on user message.<|im_end|>
<|im_start|>user
Transfer 100 rubles and 50 kopeck to Mike<|im_end|>
<|im_start|>assistant
{"contractor": "Mike", "sum": 100.5, "currency": "rubles"}<|im_end|>

**=== Параметры {'num_beams': 4, 'length_penalty': 2.0} ===**
**Сказка**: <|im_start|>system
You are a storyteller. Generate a story based on user message.<|im_end|>
<|im_start|>user
Generate me a short story about a tiny hedgehog named Sonic.<|im_end|>
<|im_start|>assistant
Once upon a time, in a small village nestled in the heart of a dense forest, there lived a tiny hedgehog named Sonic. Sonic was a curious and adventurous creature, always eager to explore the world around him. One day, while wandering through the forest, Sonic stumbled upon a hidden cave.

Inside the cave, Sonic discovered a treasure chest filled with strange and magical items. As he opened the chest, he realized that the items were not just ordinary, but were imbued with the power of the forest itself. Sonic knew that he had to find a way to return the treasure back to its rightful owner, the forest itself.

With great courage and determination, Sonic set out on a journey to retrieve the treasure. Along the way, he encountered many challenges and obstacles, but he never gave up. Finally, after a long and grueling journey, Sonic arrived at the forest's heart, where the treasure was hidden.

With a heavy heart, Sonic returned the treasure to its rightful owner, the forest itself. From that day on, Sonic became known as the guardian of the forest, and his adventures continued to bring joy and wonder to all who knew him.<|im_end|>
JSON: <|im_start|>system
You are a JSON machine. Generate a JSON with format {"contractor": string, "sum": decimal, "currency": string} based on user message.<|im_end|>
<|im_start|>user
Transfer 100 rubles and 50 kopeck to Mike<|im_end|>
<|im_start|>assistant
{"contractor": "Mike", "sum": 100.5, "currency": "rubles"}<|im_end|>

**=== Параметры {'num_beams': 8, 'length_penalty': 1.0} ===**
**Сказка**: <|im_start|>system
You are a storyteller. Generate a story based on user message.<|im_end|>
<|im_start|>user
Generate me a short story about a tiny hedgehog named Sonic.<|im_end|>
<|im_start|>assistant
Once upon a time, in a small village nestled in the heart of a dense forest, there lived a tiny hedgehog named Sonic. Sonic was a curious and adventurous creature, always eager to explore the world around him. One day, as Sonic was wandering through the forest, he stumbled upon a hidden cave.

Inside the cave, Sonic was greeted by a group of friendly creatures who welcomed him with open arms. One of the creatures was a wise old owl named Whiskers. Whiskers explained to Sonic that the cave was home to a group of magical creatures who lived in harmony with nature.

Over the next few days, Sonic spent his days exploring the cave and learning about the creatures who lived there. He discovered that the cave was home to a variety of magical creatures, including pixies, griffins, and unicorns. Each creature had its own unique abilities and powers, and Sonic was fascinated by them all.

As the days turned into weeks, and the weeks turned into months, Sonic's friendship with the magical creatures grew stronger and stronger. He learned to work with them, to communicate with them, and to protect them from any threats that came their way.

One day, while exploring the cave, Sonic stumbled upon a group of griffins who were attacking a group of pixies. Sonic knew that he had to help the pixies, but he also knew that he couldn't do it alone. With the help of Whiskers and the magical creatures who lived in the cave, Sonic was able to defeat the griffins and save the day.

From that day on, Sonic became known as a hero among the magical creatures of the cave. He continued to explore the cave and learn more about the world around him, always eager to help those who needed it the most.

And so, Sonic became a legend among the magical creatures of the cave, known not only for his bravery and courage, but also for his kindness and compassion. From that day on, Sonic lived a happy and fulfilling life, forever grateful for the magic that surrounded him.<|im_end|>
JSON: <|im_start|>system
You are a JSON machine. Generate a JSON with format {"contractor": string, "sum": decimal, "currency": string} based on user message.<|im_end|>
<|im_start|>user
Transfer 100 rubles and 50 kopeck to Mike<|im_end|>
<|im_start|>assistant
{"contractor": "Mike", "sum": 100.5, "currency": "rubles"}<|im_end|>