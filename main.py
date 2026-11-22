import re
from collections import defaultdict
import nltk
import pymorphy3
from nltk.corpus import stopwords

# Загружаем русские стоп-слова
nltk.download('stopwords', quiet=True)
RUSSIAN_STOPWORDS = set(stopwords.words('russian'))

morph = pymorphy3.MorphAnalyzer()

EXPECTED_SENTENCES_PER_WORD = 8


def preprocess_text(text: str, target_lemma: str = "") -> set[str]:
    """Приводит слова к леммам, убирает стоп-слова и само целевое слово."""
    words = re.findall(r'[а-яё]+', text.lower())
    lemmas = set()

    for word in words:
        if word in RUSSIAN_STOPWORDS:
            continue
        lemma = morph.parse(word).normal_form
        if lemma not in RUSSIAN_STOPWORDS and lemma != target_lemma:
            lemmas.add(lemma)

    return lemmas


def parse_grouped_file(filepath: str) -> dict[str, list[tuple[int, str]]]:
    """
    Универсальный парсер для файлов формата:
    Слово
    1 Текст...
    2 Текст...
    """
    data = {}
    current_word = None

    with open(filepath, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            match = re.match(r'^(\d+)\s+(.+)$', line)
            if match:
                if current_word is None:
                    raise ValueError(f"Ошибка в {filepath} (строка {line_num}): текст без заголовка слова.")
                sense_id = int(match.group(1))
                text = match.group(2)
                data[current_word].append((sense_id, text))
            else:
                current_word = morph.parse(line.lower()).normal_form
                if current_word not in data:
                    data[current_word] = []

    return data


def load_definitions(filepath: str) -> dict[str, dict[int, set[str]]]:
    raw_defs = parse_grouped_file(filepath)
    definitions = {}

    for word, items in raw_defs.items():
        definitions[word] = {}
        for sense_id, gloss in items:
            definitions[word][sense_id] = preprocess_text(gloss, word)

    return definitions


def load_and_validate_sentences(filepath: str, definitions: dict) -> dict[str, list[tuple[int, str]]]:
    sentences_by_word = parse_grouped_file(filepath)
    valid_data = {}

    for word, items in sentences_by_word.items():
        if word not in definitions:
            raise ValueError(f"Слово '{word}' из {filepath} не найдено в словаре определений.")
        
        count = len(items)
        if count != EXPECTED_SENTENCES_PER_WORD:
            print(f"⚠️ Внимание: для слова '{word}' найдено {count} из {EXPECTED_SENTENCES_PER_WORD} предложений.")

        for sense_id, _ in items:
            if sense_id not in definitions[word]:
                raise ValueError(f"Для слова '{word}' указан несуществующий номер значения: {sense_id}.")
        
        valid_data[word] = items

    # Если в файле 1 слово, статистика по слову будет делиться на реальное кол-во (len(items))
    return valid_data



def simplified_lesk(target_word: str, sentence: str, definitions: dict) -> int:
    senses = definitions[target_word]
    context_lemmas = preprocess_text(sentence, target_word)

    best_sense = 1
    max_overlap = -1

    for sense_id in sorted(senses.keys()):
        overlap = len(context_lemmas.intersection(senses[sense_id]))
        if overlap > max_overlap:
            max_overlap = overlap
            best_sense = sense_id

    return best_sense


def evaluate(definitions_file: str, sentences_file: str):
    definitions = load_definitions(definitions_file)
    sentences_by_word = load_and_validate_sentences(sentences_file, definitions)

    total_sentences = 0
    lesk_total_correct = 0
    baseline_total_correct = 0

    print(f"{'Слово':<12} | {'Эталон':<6} | {'Леск':<6} | {'Бейзлайн':<8} | Предложение")
    print("-" * 85)

    word_stats = defaultdict(lambda:)  # [lesk_hits, baseline_hits]

    for word, items in sentences_by_word.items():
        for gold_sense, sentence in items:
            pred_sense = simplified_lesk(word, sentence, definitions)
            base_sense = 1

            total_sentences += 1
            if pred_sense == gold_sense:
                lesk_total_correct += 1
                word_stats[word][0] += 1
            if base_sense == gold_sense:
                baseline_total_correct += 1
                word_stats[word][1] += 1

            mark = "✅" if pred_sense == gold_sense else "❌"
            short_sent = (sentence[:45] + "...") if len(sentence) > 48 else sentence
            print(f"{word:<12} | {gold_sense:<6} | {pred_sense:<4} {mark} | {base_sense:<8} | {short_sent}")

    print("\n" + "=" * 55)
    print(f"{'Статистика по словам':<15} | {'Леск (из 8)':<14} | {'Бейзлайн (из 8)':<15}")
    print("-" * 55)
    for word, (l_hits, b_hits) in word_stats.items():
        print(f"{word:<15} | {l_hits}/8 ({l_hits/8*100:>5.1f}%) | {b_hits}/8 ({b_hits/8*100:>5.1f}%)")

    lesk_acc = (lesk_total_correct / total_sentences) * 100
    base_acc = (baseline_total_correct / total_sentences) * 100

    print("=" * 55)
    print(f"ИТОГО ({total_sentences} предложений):")
    print(f"Точность алгоритма Леска:  {lesk_acc:.2f}% ({lesk_total_correct}/{total_sentences})")
    print(f"Точность базового решения: {base_acc:.2f}% ({baseline_total_correct}/{total_sentences})")
    print("=" * 55)


if __name__ == "__main__":
    evaluate("definitions.txt", "sentenses.txt")
