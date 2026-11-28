import re
from collections import defaultdict

import pymorphy3


# Настройки
EXPECTED_SENTENCES_PER_WORD = 8

morph = pymorphy3.MorphAnalyzer()

# Русские стоп-слова.
RUSSIAN_STOPWORDS = set("""
и в во не что он на я с со как а то все она так его но да ты к у же вы за
бы по только ее мне было вот от меня еще нет о из ему теперь когда даже ну
вдруг ли если уже или ни быть был него до вас нибудь опять уж вам ведь там
потом себя ничего ей может они тут где есть надо ней для мы тебя их чем была
сам чтоб без будто человек чего раз тоже себе под жизнь будет ж тогда кто
этот того потому этого какой совсем ним здесь этом один почти мой тем чтобы
нее сейчас были куда зачем сказать всех никогда сегодня можно при наконец
два об через более тот итак никто кому
""".split())


def normalize_word(word: str) -> str:
    """Приводит заголовок слова к начальной форме."""
    word = word.strip().lower()
    return morph.parse(word)[0].normal_form


def preprocess_text(text: str, target_lemma: str = "") -> set[str]:
    """
    Разбивает текст на слова, приводит их к леммам,
    удаляет стоп-слова и само исследуемое слово.
    """
    words = re.findall(r"[а-яё]+", text.lower())

    lemmas = set()

    for word in words:
        if word in RUSSIAN_STOPWORDS:
            continue

        lemma = morph.parse(word)[0].normal_form

        if lemma in RUSSIAN_STOPWORDS:
            continue

        if lemma == target_lemma:
            continue

        lemmas.add(lemma)

    return lemmas


def parse_grouped_file(filepath: str) -> dict[str, list[tuple[int, str]]]:
    data = {}
    current_word = None

    with open(filepath, "r", encoding="utf-8") as file:
        for line_num, line in enumerate(file, 1):
            line = line.strip()

            if not line:
                continue

            # Строка с номером и необязательным текстом
            match = re.match(r"^(\d+)\s*(.*)$", line)

            if match:
                if current_word is None:
                    raise ValueError(
                        f"Строка {line_num}: предложение без заголовка."
                    )

                number = int(match.group(1))
                sentence = match.group(2).strip()

                # Пустые заготовки пропускаем
                if not sentence:
                    continue

                data[current_word].append((number, sentence))

            else:
                current_word = normalize_word(line)

                if current_word in data:
                    raise ValueError(
                        f"Повторный заголовок '{current_word}' "
                        f"в строке {line_num}."
                    )

                data[current_word] = []

    return data


def load_definitions(filepath: str) -> dict[str, dict[int, set[str]]]:
    """Загружает определения и преобразует их в множества лемм."""
    raw_definitions = parse_grouped_file(filepath)

    definitions = {}

    for word, items in raw_definitions.items():

        items.sort(key=lambda item: item[0])

        sense_ids = [sense_id for sense_id, _ in items]

        # Проверяем, что значения пронумерованы от 1 до N.
        expected_ids = list(range(1, len(items) + 1))

        if sense_ids != expected_ids:
            raise ValueError(
                f"Некорректная нумерация определений слова '{word}'. "
                f"Ожидалось {expected_ids}, получено {sense_ids}."
            )

        definitions[word] = {}

        for sense_id, gloss in items:
            definitions[word][sense_id] = preprocess_text(gloss, word)

    return definitions


def load_and_validate_sentences(
    filepath: str,
    definitions: dict
) -> dict[str, list[tuple[int, str, int]]]:
    """
    Загружает предложения.

    Возвращает для каждого предложения:
    (номер предложения, текст, эталонное значение).

    Эталонное значение определяется циклически:
    1, 2, ..., N, 1, 2, ..., N.
    """
    raw_sentences = parse_grouped_file(filepath)

    sentences_by_word = {}

    for word, items in raw_sentences.items():

        if word not in definitions:
            raise ValueError(
                f"Слово '{word}' из {filepath} "
                "не найдено в definitions.txt."
            )

        items.sort(key=lambda item: item[0])

        sentence_numbers = [number for number, _ in items]
        expected_numbers = list(range(1, len(items) + 1))

        if sentence_numbers != expected_numbers:
            raise ValueError(
                f"Некорректная нумерация предложений слова '{word}'. "
                f"Ожидалось {expected_numbers}, получено {sentence_numbers}."
            )

        if len(items) != EXPECTED_SENTENCES_PER_WORD:
            print(
                f"Предупреждение: у слова '{word}' "
                f"{len(items)} предложений вместо "
                f"{EXPECTED_SENTENCES_PER_WORD}."
            )

        number_of_senses = len(definitions[word])
        prepared_items = []

        for sentence_number, sentence in items:

            # Например, для трёх значений:
            # 1, 2, 3, 1, 2, 3, 1, 2.
            gold_sense = ((sentence_number - 1) % number_of_senses) + 1

            prepared_items.append(
                (sentence_number, sentence, gold_sense)
            )

        sentences_by_word[word] = prepared_items

    return sentences_by_word


def simplified_lesk(
    target_word: str,
    sentence: str,
    definitions: dict
):
    senses = definitions[target_word]

    context_lemmas = preprocess_text(sentence, target_word)

    overlaps = {}

    for sense_id in sorted(senses.keys()):
        overlaps[sense_id] = len(
            context_lemmas.intersection(senses[sense_id])
        )

    max_overlap = max(overlaps.values())

    # Все значения, набравшие максимальное количество совпадений
    candidates = [
        sense_id
        for sense_id, score in overlaps.items()
        if score == max_overlap
    ]

    # При ничьей выбираем первое значение
    best_sense = min(candidates)

    return best_sense, max_overlap, candidates


def evaluate(definitions_file: str, sentences_file: str):

    definitions = load_definitions(definitions_file)

    sentences_by_word = load_and_validate_sentences(
        sentences_file,
        definitions
    )

    # Общая статистика
    total = 0
    lesk_correct = 0
    baseline_correct = 0

    # Статистика по основаниям выбора
    guessed = 0
    guessed_correct = 0

    evidence = 0
    evidence_correct = 0

    zero_overlap = 0
    positive_ties = 0

    # Распределение количества совпадений
    overlap_distribution = defaultdict(int)

    # Какие значения выбирает алгоритм
    predictions = defaultdict(int)

    # Матрица ошибок: эталонное значение -> предсказанное
    confusion = defaultdict(int)

    # Статистика по словам
    word_stats = {}

    print("\nРЕЗУЛЬТАТЫ ПО ПРЕДЛОЖЕНИЯМ")
    print("=" * 115)

    print(
        f"{'Слово':<12} | "
        f"{'№':<3} | "
        f"{'Эталон':<6} | "
        f"{'Леск':<5} | "
        f"{'Совп.':<5} | "
        f"{'Основание':<23} | "
        f"{'Бейзлайн':<8} | "
        f"Предложение"
    )

    print("-" * 115)

    for word, items in sentences_by_word.items():

        word_stats[word] = {
            "total": 0,
            "lesk_correct": 0,
            "baseline_correct": 0,
            "guessed": 0,
            "guessed_correct": 0,
            "evidence": 0,
            "evidence_correct": 0,
            "overlap_sum": 0
        }

        for sentence_number, sentence, gold_sense in items:

            pred_sense, max_overlap, candidates = simplified_lesk(
                word,
                sentence,
                definitions
            )

            baseline_sense = 1

            total += 1
            word_stats[word]["total"] += 1

            # Проверка правильности
            is_correct = pred_sense == gold_sense
            is_baseline_correct = baseline_sense == gold_sense

            if is_correct:
                lesk_correct += 1
                word_stats[word]["lesk_correct"] += 1

            if is_baseline_correct:
                baseline_correct += 1
                word_stats[word]["baseline_correct"] += 1

            # Основание выбора значения
            if len(candidates) > 1:

                guessed += 1
                word_stats[word]["guessed"] += 1

                if is_correct:
                    guessed_correct += 1
                    word_stats[word]["guessed_correct"] += 1

                if max_overlap == 0:
                    zero_overlap += 1
                    reason = "УГАДЫВАНИЕ"
                else:
                    positive_ties += 1
                    reason = "НИЧЬЯ"

            else:

                evidence += 1
                word_stats[word]["evidence"] += 1

                if is_correct:
                    evidence_correct += 1
                    word_stats[word]["evidence_correct"] += 1

                reason = "ОБОСНОВАНО"

            # Сбор дополнительной статистики
            overlap_distribution[max_overlap] += 1
            predictions[pred_sense] += 1
            confusion[(gold_sense, pred_sense)] += 1

            word_stats[word]["overlap_sum"] += max_overlap

            # Вывод маркера правильности
            mark = "OK" if is_correct else "ОШИБКА"

            short_sentence = (
                sentence[:35] + "..."
                if len(sentence) > 38
                else sentence
            )

            candidates_str = ",".join(map(str, candidates))

            print(
                f"{word:<12} | "
                f"{sentence_number:<3} | "
                f"{gold_sense:<6} | "
                f"{pred_sense:<5} | "
                f"{max_overlap:<5} | "
                f"{reason + ' [' + candidates_str + ']':<23} | "
                f"{baseline_sense:<8} | "
                f"{mark} {short_sentence}"
            )

    if total == 0:
        print("\nНет предложений для оценки.")
        return

    # ==========================================
    # Статистика по каждому слову
    # ==========================================

    print("\n" + "=" * 95)
    print("СТАТИСТИКА ПО СЛОВАМ")
    print("=" * 95)

    print(
        f"{'Слово':<12} | "
        f"{'Леск':<13} | "
        f"{'Бейзлайн':<13} | "
        f"{'Угадываний':<12} | "
        f"{'Обосновано':<12} | "
        f"Сравнение"
    )

    print("-" * 95)

    useless_words = []

    for word in definitions:

        stats = word_stats.get(word)

        if stats is None or stats["total"] == 0:
            print(f"{word:<12} | Нет предложений")
            continue

        n = stats["total"]

        lesk_acc = stats["lesk_correct"] / n * 100
        base_acc = stats["baseline_correct"] / n * 100

        difference = lesk_acc - base_acc

        if difference > 0:
            comparison = "ЛЕСК ЛУЧШЕ"
        elif difference < 0:
            comparison = "ЛЕСК ХУЖЕ"
        else:
            comparison = "НА РАВНЕ"

        if difference <= 0:
            useless_words.append(word)

        print(
            f"{word:<12} | "
            f"{lesk_acc:5.1f}% ({stats['lesk_correct']}/{n}) | "
            f"{base_acc:5.1f}% ({stats['baseline_correct']}/{n}) | "
            f"{stats['guessed']:<12} | "
            f"{stats['evidence']:<12} | "
            f"{comparison}"
        )

        # Дополнительная статистика конкретного слова
        if stats["guessed"] > 0:
            guess_acc = (
                stats["guessed_correct"] / stats["guessed"] * 100
            )
        else:
            guess_acc = 0

        if stats["evidence"] > 0:
            evidence_acc = (
                stats["evidence_correct"] / stats["evidence"] * 100
            )
        else:
            evidence_acc = 0

        print(
            f"             Правильность при угадывании: "
            f"{guess_acc:.1f}%; "
            f"при обоснованном выборе: {evidence_acc:.1f}%"
        )

    # ==========================================
    # Общая статистика
    # ==========================================

    lesk_acc = lesk_correct / total * 100
    baseline_acc = baseline_correct / total * 100

    print("\n" + "=" * 70)
    print("ОБЩАЯ ТОЧНОСТЬ")
    print("=" * 70)

    print(f"Всего предложений: {total}")
    print(
        f"Леск: {lesk_acc:.2f}% "
        f"({lesk_correct}/{total})"
    )
    print(
        f"Бейзлайн: {baseline_acc:.2f}% "
        f"({baseline_correct}/{total})"
    )
    print(
        f"Разница: {lesk_acc - baseline_acc:+.2f} процентных пункта"
    )

    # ==========================================
    # Обоснованные решения и угадывания
    # ==========================================

    print("\n" + "=" * 70)
    print("АНАЛИЗ ОСНОВАНИЙ ВЫБОРА")
    print("=" * 70)

    print(
        f"Обоснованные решения: {evidence}/{total} "
        f"({evidence / total * 100:.1f}%)"
    )

    print(
        f"Правильных среди обоснованных: "
        f"{evidence_correct}/{evidence} "
        f"({evidence_correct / evidence * 100 if evidence else 0:.1f}%)"
    )

    print()

    print(
        f"Угадывания (ничьи): {guessed}/{total} "
        f"({guessed / total * 100:.1f}%)"
    )

    print(
        f"Правильных среди угадываний: "
        f"{guessed_correct}/{guessed} "
        f"({guessed_correct / guessed * 100 if guessed else 0:.1f}%)"
    )

    print()
    print(f"Из них без единого совпадения: {zero_overlap}")
    print(f"Ничьих с положительным числом совпадений: {positive_ties}")

    # ==========================================
    # Распределение числа совпадений
    # ==========================================

    print("\n" + "=" * 70)
    print("РАСПРЕДЕЛЕНИЕ КОЛИЧЕСТВА СОВПАДАЮЩИХ ЛЕММ")
    print("=" * 70)

    for count in sorted(overlap_distribution):
        amount = overlap_distribution[count]

        print(
            f"{count} совпадений: {amount} "
            f"({amount / total * 100:.1f}%)"
        )

    print(
        f"\nСреднее количество совпадений: "
        f"{sum(k * v for k, v in overlap_distribution.items()) / total:.2f}"
    )

    # ==========================================
    # Распределение предсказанных значений
    # ==========================================

    print("\n" + "=" * 70)
    print("КАКИЕ ЗНАЧЕНИЯ ВЫБИРАЕТ ЛЕСК")
    print("=" * 70)

    for sense_id in sorted(predictions):
        amount = predictions[sense_id]

        print(
            f"Значение {sense_id}: {amount} "
            f"({amount / total * 100:.1f}%)"
        )

    # ==========================================
    # Матрица ошибок
    # ==========================================

    max_senses = max(
        len(senses) for senses in definitions.values()
    )

    print("\n" + "=" * 70)
    print("МАТРИЦА ОШИБОК")
    print("=" * 70)

    print(
        "Строки — эталонные значения, "
        "столбцы — предсказанные."
    )

    print(
        f"{'Эталон / Леск':<17} | "
        + " | ".join(
            f"{i:>5}" for i in range(1, max_senses + 1)
        )
    )

    for gold in range(1, max_senses + 1):

        row = [
            confusion[(gold, pred)]
            for pred in range(1, max_senses + 1)
        ]

        print(
            f"{gold:<17} | "
            + " | ".join(f"{value:>5}" for value in row)
        )

    # ==========================================
    # Сравнение слов
    # ==========================================

    print("\n" + "=" * 70)
    print("ИТОГ ПО ПОЛЕЗНОСТИ АЛГОРИТМА")
    print("=" * 70)

    better = 0
    equal = 0
    worse = 0

    for word, stats in word_stats.items():

        if stats["total"] == 0:
            continue

        difference = (
            stats["lesk_correct"] - stats["baseline_correct"]
        )

        if difference > 0:
            better += 1
        elif difference < 0:
            worse += 1
        else:
            equal += 1

    print(f"Леск лучше бейзлайна: {better} слов")
    print(f"Леск наравне с бейзлайном: {equal} слов")
    print(f"Леск хуже бейзлайна: {worse} слов")

    print("\nСлова, где Леск не превзошёл бейзлайн:")

    if useless_words:
        print(", ".join(useless_words))
    else:
        print("Таких слов нет.")

    # ==========================================
    # Отсутствующие предложения
    # ==========================================

    missing_words = [
        word for word in definitions
        if word not in word_stats or word_stats[word]["total"] == 0
    ]

    if missing_words:
        print("\nСлова без готовых предложений:")
        print(", ".join(missing_words))

    # Средняя точность по словам (каждое слово имеет одинаковый вес)
    word_accuracies = [
        stats["lesk_correct"] / stats["total"] * 100
        for stats in word_stats.values()
        if stats["total"] > 0
    ]

    if word_accuracies:
        print(
            f"\nСредняя точность по словам (macro-average): "
            f"{sum(word_accuracies) / len(word_accuracies):.2f}%"
        )

    print("\n" + "=" * 70)


if __name__ == "__main__":
    evaluate("definitions.txt", "sentences.txt")