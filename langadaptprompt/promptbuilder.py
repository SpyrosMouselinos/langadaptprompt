import random
import pandas as pd
import numpy as np
from collections import Counter
import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import json
import openai
from .paraphraser import PARAPHRASE_SYSTEM_PROMPT, PARAPHRASE_USER_PROMPT, PARAPHRASE_ASSISTANT_PROMPT
from .selfprompter import SELFPROMPT_SYSTEM_PROMPT, SELFPROMPT_USER_PROMPT, SELFPROMPT_ASSISTANT_PROMPT, \
    SELFPROMPT_USER_PROMPT2, SELFPROMPT_ASSISTANT_PROMPT2
from .constants import PACK_ORDERING
import time

model = SentenceTransformer('all-MiniLM-L6-v2')


def Gemini_TXT_Query(gemini_model, prompt, n=1):
    results = []
    for i in range(n):
        while True:
            try:
                response = gemini_model.generate_content(
                    prompt,
                    generation_config={
                        'temperature': 0.8,
                    },
                )
                results.append(response.text)
            except Exception as e:
                print(e)
                print("Waiting for vertex...")
                time.sleep(5)

    if len(results) != n:
        print(f"You requested {n} responses but you got {len(results)}...")
    return results


def count_inversions(a, b):
    inversions = 0
    for i in range(len(a)):
        for j in range(i + 1, len(a)):
            if a[i] != a[j] and b.index(a[i]) > b.index(a[j]):
                inversions += 1
    return inversions


def fcs(input_string, string_list, result_list):
    words = input_string.split()
    potential_matches = []

    for item in string_list:
        item_words = item.split()
        if all(word in item_words for word in words):
            potential_matches.append(item)

    best_idx = 0
    min_inversions = float('inf')

    for index, item in enumerate(potential_matches):
        item_words = [word for word in item.split() if word in words]
        inversions = count_inversions(item_words, words)

        if inversions == 0:
            return result_list[index]

        if inversions < min_inversions:
            min_inversions = inversions
            best_idx = index

    return result_list[best_idx]


def extract_int_lists(text):
    pattern = r'\[\s*(\d+(?:\s*,\s*\d+)*)\s*\]'
    matches = re.findall(pattern, text)
    int_lists = []
    try:
        for match in matches:
            int_list = list(map(int, re.findall(r'\d+', match)))
            int_lists.append(int_list)
    except:
        return []
    return int_lists[0]


def pd_load_data(df, limit=None):
    if isinstance(df, str):
        data = pd.read_csv(df)
    else:
        data = df
    if limit is None:
        questions = data['question'].values
        # questions = [f.split('Tool List')[0] for f in questions]
        answers_nl = data['solution_nl'].values
        answers_pl = data['solution_tool'].values
        pack = data['pack'].values
    else:
        alter_pack = data['pack'].apply(lambda x: PACK_ORDERING[x])
        data = data[alter_pack <= limit]
        questions = data['question'].values
        # questions = [f.split('Tool List')[0] for f in questions]
        answers_nl = data['solution_nl'].values
        answers_pl = data['solution_tool'].values
        pack = data['pack'].values
    return questions, answers_nl, answers_pl, pack


def find_diverse_paraphrase(item, list_of_items):
    single = [item]
    multiple = list_of_items
    single_embeddings = model.encode(single, convert_to_tensor=True)
    multiple_embeddings = model.encode(multiple, convert_to_tensor=True)
    single_embeddings = single_embeddings.cpu().numpy()
    multiple_embeddings = multiple_embeddings.cpu().numpy()
    similarities = cosine_similarity(single_embeddings, multiple_embeddings)[0]
    choices = list(np.argsort(similarities))
    result = multiple[choices[0]]
    return result


def extract_and_filter(list_of_lists):
    list_of_lists = [extract_int_lists(f) for f in list_of_lists]
    # Check if the input list is empty
    if not list_of_lists:
        return []

    # Flatten the list of lists
    flat_list = [item for sublist in list_of_lists for item in sublist]

    # Check if the flattened list is empty
    if not flat_list:
        return []

    # Count the frequency of each item and get the top 5
    item_counts = Counter(flat_list)
    most_common = item_counts.most_common(5)
    # Extract just the items from the most common pairs
    return [item for item, count in most_common]


def paraphrase(prompt, openai_key, gemini_model):
    if openai_key is not None:
        messages = [
            {"role": "system", "content": f"{PARAPHRASE_SYSTEM_PROMPT}"},
            {"role": "user", "content": f"{PARAPHRASE_USER_PROMPT}"},
            {"role": "assistant", "content": f"{PARAPHRASE_ASSISTANT_PROMPT}"},
            {"role": "user", "content": f"{prompt}"},
        ]
        result = openai.ChatCompletion.create(
            api_key=openai_key,
            model='gpt-4',
            max_tokens=300,
            stop=None,
            messages=messages,
            temperature=0.8,
            n=5)
        results = [choice['message']['content'] for choice in result['choices']]
    elif gemini_model is not None:
        messages = [
            PARAPHRASE_SYSTEM_PROMPT,
            PARAPHRASE_USER_PROMPT,
            PARAPHRASE_ASSISTANT_PROMPT,
            prompt
        ]

        final_prompt = '\n\n'.join(messages)
        results = Gemini_TXT_Query(gemini_model, final_prompt, n=5)
    else:
        raise ValueError
    return find_diverse_paraphrase(prompt, results)


def filter_(prompt, openai_key):
    messages = [
        {"role": "system", "content": f"{SELFPROMPT_SYSTEM_PROMPT}"},
        {"role": "user", "content": f"{SELFPROMPT_USER_PROMPT}"},
        {"role": "assistant", "content": f"{SELFPROMPT_ASSISTANT_PROMPT}"},
        {"role": "user", "content": f"{SELFPROMPT_USER_PROMPT2}"},
        {"role": "assistant", "content": f"{SELFPROMPT_ASSISTANT_PROMPT2}"},
        {"role": "user", "content": f"{prompt}"},
    ]
    result = openai.ChatCompletion.create(
        api_key=openai_key,
        model='gpt-3.5-turbo-1106',
        max_tokens=35,
        stop=None,
        messages=messages,
        temperature=0.8,
        n=10)
    results = [choice['message']['content'] for choice in result['choices']]
    return extract_and_filter(results)


def read_jsonl(file_path):
    with open(file_path, 'r') as file:
        return [json.loads(line) for line in file]


def find_st_similar_questions(item, list_of_items, answers, top_n=5):
    train_questions = item if isinstance(item, list) else [item]
    test_questions = list_of_items
    train_embeddings = model.encode(train_questions, convert_to_tensor=True)
    test_embeddings = model.encode(test_questions, convert_to_tensor=True)
    train_embeddings = train_embeddings.cpu().numpy()
    test_embeddings = test_embeddings.cpu().numpy()
    similarities = cosine_similarity(train_embeddings, test_embeddings)
    similarities[np.where(similarities >= 0.95)] = 0.0
    top_indices = similarities.argsort()
    top_indices = top_indices[:, -top_n:].reshape(-1, )
    similar_questions = [list_of_items[f] for f in top_indices]
    similar_answers = [answers[f] for f in top_indices]
    few_shot = ''
    for example, response in zip(similar_questions, similar_answers):
        few_shot += example + '\n' + response + '\n\n'
    return few_shot.strip()


def find_self_similar_questions(item, list_of_items, answers, openai_key):
    train_questions = item if isinstance(item, list) else [item]
    test_questions = list_of_items
    train_embeddings = model.encode(train_questions, convert_to_tensor=True)
    test_embeddings = model.encode(test_questions, convert_to_tensor=True)
    train_embeddings = train_embeddings.cpu().numpy()
    test_embeddings = test_embeddings.cpu().numpy()
    similarities = cosine_similarity(train_embeddings, test_embeddings)
    similarities[np.where(similarities >= 0.95)] = 0.0
    top_indices = similarities.argsort()
    filter_head = int(10 / len(item))
    top_indices = top_indices[:, -filter_head:].reshape(-1, )
    similar_questions = [list_of_items[f] for f in top_indices]
    similar_answers = [answers[f] for f in top_indices]
    few_shot_list = []
    for example, response in zip(similar_questions, similar_answers):
        few_shot_list.append(example + '\n' + response)
    few_shot_prompt = 'Unsolved Problem:\n'
    few_shot_prompt += item[0]
    few_shot_prompt += '\n\nSolved Problems:\n'
    index = 1
    for example, response in zip(similar_questions, similar_answers):
        clean_example = example.split("Tool List")[0].strip()
        few_shot_prompt += str(index) + ') ' + clean_example + '\n\n'
        index += 1
    ### Here you have a list of top 10 most similar Questions ###
    ### Filter through GPT Model ###
    indexes = set(filter_(few_shot_prompt, openai_key=openai_key))
    new_indexes = []
    for it in indexes:
        if it <= 10:
            new_indexes.append(it)
        else:
            new_indexes.append(random.randint(1, 10))
    if len(new_indexes) <= 1:
        return find_st_similar_questions(item, list_of_items, answers, top_n=5)
    else:
        filtered_few_shot = [few_shot_list[f - 1] for f in new_indexes]

    few_shot = ''
    for example in filtered_few_shot:
        few_shot += example + '\n\n'
    return few_shot.strip()


def few_shot_builder(current_question,
                     dataset_df='../test/filtered_euclidea.csv',
                     pack_limit=None,
                     mode='Adapt',
                     self_reflect=True,
                     openai_key=None,
                     gemini_model=None,
                     ):
    questions, answers_nl, answers_pl, pack = pd_load_data(dataset_df)
    mixprompt = current_question + '\n\n' + '\n\n' + fcs(current_question, questions, answers_nl)

    if self_reflect:
        paraphrased = paraphrase(mixprompt, openai_key=openai_key, gemini_model=gemini_model)
    else:
        paraphrased = ''
    nextprompt = [current_question]
    top_n = 5

    if pack_limit is None:
        pass
    else:
        assert pack_limit in PACK_ORDERING.keys()
        pack_limit = PACK_ORDERING[pack_limit]
        questions, answers_nl, answers_pl, pack = pd_load_data(dataset_df, limit=pack_limit)

    if mode == 'Adapt':
        few_shot = find_st_similar_questions(nextprompt, questions, answers_nl, top_n)
    # elif mode == 'Self':
    # few_shot = find_self_similar_questions(nextprompt, questions, answers_nl, openai_key)
    elif mode == 'Zero-Shot':
        few_shot = ''
    else:
        raise ValueError(
            'Mode must be one of the following: [Adapt,Zero-Shot] if you want to have a static Few-Shot prompt, just add it to your question.')
    return (paraphrased + '\n\n' + few_shot + '\n\n' + current_question).strip()
