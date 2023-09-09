# TODO: Remove all TODO comments once the implementation is complete.
"""
TODO: Add the Paper Title on this line.
TODO: Add the paper's PDF URL (preferably from arXiv) on this line.

TODO: Write a Short Description of the task.

Homepage: TODO: Add the URL to the task's Homepage here.
"""
from typing import List
import datasets
from math import exp
from lm_eval.base import rf, Task
from functools import partial
from packaging import version


# TODO: Add the BibTeX citation for the task.
_CITATION = """
"""



'''

#from __future__ import print_function
from collections import Counter
import string
import re
import argparse
import json
import sys
import unicodedata


PUNCT = {chr(i) for i in range(sys.maxunicode) if unicodedata.category(chr(i)).startswith('P')}.union(string.punctuation)
WHITESPACE_LANGS = ['en', 'es', 'hi', 'vi', 'de', 'ar']
MIXED_SEGMENTATION_LANGS = ['zh']


def whitespace_tokenize(text):
    return text.split()


def mixed_segmentation(text):
    segs_out = []
    temp_str = ""
    for char in text:
        if re.search(r'[\u4e00-\u9fa5]', char) or char in PUNCT:
            if temp_str != "":
                ss = whitespace_tokenize(temp_str)
                segs_out.extend(ss)
                temp_str = ""
            segs_out.append(char)
        else:
            temp_str += char

    if temp_str != "":
        ss = whitespace_tokenize(temp_str)
        segs_out.extend(ss)

    return segs_out


def normalize_answer(s, lang):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text, lang):
        if lang == 'en':
            return re.sub(r'\b(a|an|the)\b', ' ', text)
        elif lang == 'es':
            return re.sub(r'\b(un|una|unos|unas|el|la|los|las)\b', ' ', text)
        elif lang == 'hi':
            return text # Hindi does not have formal articles
        elif lang == 'vi':
            return re.sub(r'\b(của|là|cái|chiếc|những)\b', ' ', text)
        elif lang == 'de':
            return re.sub(r'\b(ein|eine|einen|einem|eines|einer|der|die|das|den|dem|des)\b', ' ', text)
        elif lang == 'ar':
            return re.sub('\sال^|ال', ' ', text)
        elif lang == 'zh':
            return text # Chinese does not have formal articles
        else:
            raise Exception('Unknown Language {}'.format(lang))

    def white_space_fix(text, lang):
        if lang in WHITESPACE_LANGS:
            tokens = whitespace_tokenize(text)
        elif lang in MIXED_SEGMENTATION_LANGS:
            tokens = mixed_segmentation(text)
        else:
            raise Exception('Unknown Language {}'.format(lang))
        return ' '.join([t for t in tokens if t.strip() != ''])

    def remove_punc(text):
        return ''.join(ch for ch in text if ch not in PUNCT)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s)), lang), lang)


def f1_score(prediction, ground_truth, lang):
    prediction_tokens = normalize_answer(prediction, lang).split()
    ground_truth_tokens = normalize_answer(ground_truth, lang).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth, lang):
    return (normalize_answer(prediction, lang) == normalize_answer(ground_truth, lang))


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths, lang):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth, lang)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def evaluate(dataset, predictions, lang):
    f1 = exact_match = total = 0
    for article in dataset:
        for paragraph in article['paragraphs']:
            for qa in paragraph['qas']:
                total += 1
                if qa['id'] not in predictions:
                    message = 'Unanswered question ' + qa['id'] + \
                              ' will receive score 0.'
                    print(message, file=sys.stderr)
                    continue
                ground_truths = list(map(lambda x: x['text'], qa['answers']))
                prediction = predictions[qa['id']]
                exact_match += metric_max_over_ground_truths(
                    exact_match_score, prediction, ground_truths, lang)
                f1 += metric_max_over_ground_truths(
                    f1_score, prediction, ground_truths, lang)

    exact_match = 100.0 * exact_match / total
    f1 = 100.0 * f1 / total

    return {'exact_match': exact_match, 'f1': f1}


if __name__ == '__main__':
    expected_version = '1.0'
    parser = argparse.ArgumentParser(
        description='Evaluation for MLQA ' + expected_version)
    parser.add_argument('dataset_file', help='Dataset file')
    parser.add_argument('prediction_file', help='Prediction File')
    parser.add_argument('answer_language', help='Language code of answer language')

    args = parser.parse_args()
    with open(args.dataset_file) as dataset_file:
        dataset_json = json.load(dataset_file)
        if (str(dataset_json['version']) != expected_version):
            print('Evaluation expects v-' + expected_version +
                  ', but got dataset with v-' + dataset_json['version'],
                  file=sys.stderr)
        dataset = dataset_json['data']
    with open(args.prediction_file) as prediction_file:
        predictions = json.load(prediction_file)
    print(json.dumps(evaluate(dataset, predictions, args.answer_language)))
'''


def _squad_metric(predictions, references):
    squad_metric = datasets.load_metric("squad_v2")
    print(predictions)
    print(references)
    return squad_metric.compute(predictions=predictions, references=references)



def _squad_agg(key, items):
    predictions, references = zip(*items)

    return _squad_metric(predictions=predictions, references=references).get(key, 0)



# TODO: Replace `NewTask` with the name of your Task.
class MLQA(Task):
    VERSION = 0
    # TODO: Add the `DATASET_PATH` string. This will be the name of the `Task`
    # dataset as denoted in HuggingFace `datasets`.
    DATASET_PATH = "mlqa"
    # TODO: Add the `DATASET_NAME` string. This is the name of a subset within
    # `DATASET_PATH`. If there aren't specific subsets you need, leave this as `None`.
    DATASET_NAME = 'mlqa.ar.ar'

    def has_training_docs(self):
        # TODO: Fill in the return with `True` if the Task has training data; else `False`.
        return False

    def has_validation_docs(self):
        # TODO: Fill in the return with `True` if the Task has validation data; else `False`.
        return True

    def has_test_docs(self):
        # TODO: Fill in the return with `True` if the Task has test data; else `False`.
        return True

    def training_docs(self):
        return self.dataset['train']

    def validation_docs(self):
        return self.dataset["validation"]

    def test_docs(self):
        return self.dataset['test']

    def _process_doc(self, doc):
        # TODO: Process (detokenize, strip, replace etc.) each individual `doc`
        # with this function. You can map this across the docs in each available
        # dataset split. See the TODOs in `train_docs`, `validation_docs`, and
        # `test_docs` for snippets.
        # NOTE: DELETE THIS FUNCTION IF UNUSED.
        return doc

    def doc_to_text(self, doc):
        # TODO: Format the query prompt portion of the document example.
        return (
            "اقرأ النص التالي ثم أجب عن ما يلي:"
            + doc["context"]
            + "\n\n"
            + "سؤال:"
            + doc["question"]
            + "\n\n"
            + "الإجابة:"
        )

    def doc_to_target(self, doc):
        # TODO: Fill in the `target` ("gold answer") variable.
        # The prepended `" "` is required to space out the `doc_to_text` and
        # `doc_to_target` strings.
        target = doc['answers']['text'][0]
        return " " + target

    def doc_to_fewshot_prompt(self, doc):
        prompt = self.doc_to_text(doc) + self.doc_to_target(doc)
        return prompt

    def construct_requests(self, doc, ctx):
        """Uses RequestFactory to construct Requests and returns an iterable of
        Requests which will be sent to the LM.

        :param doc:
            The document as returned from training_docs, validation_docs, or
            test_docs.
        :param ctx: str
            The context string, generated by fewshot_context. This includes the natural
            language description, as well as the few shot examples, and the question
            part of the document for `doc`.
        """

        continuation = rf.greedy_until(ctx, {"until": ["\n"]})
        is_unanswerable = rf.loglikelihood(ctx, " " + "السؤال غير قابل للإجابة")
        return continuation, is_unanswerable

    def process_results(self, doc, results):
        """Take a single document and the LM results and evaluates, returning a
        dict where keys are the names of submetrics and values are the values of
        the metric for that one document

        :param doc:
            The document as returned from training_docs, validation_docs, or test_docs.
        :param results:
            The results of the requests created in construct_requests.
        """
        continuation, (logprob_unanswerable, _) = results

        no_answer_probability = exp(logprob_unanswerable)

        predictions = {
            "id": doc["id"],
            "prediction_text": continuation,
            "no_answer_probability": no_answer_probability,
        }

        references = {
            "id": doc["id"],
            "answers": doc["answers"],
        }

        return {
            "exact": (
                predictions,
                references,
            ),  # Exact match (the normalized answer exactly match the gold answer)
            "f1": (
                predictions,
                references,
            ),  # The F-score of predicted tokens versus the gold answer
            "HasAns_exact": (
                predictions,
                references,
            ),  # Exact match (the normalized answer exactly match the gold answer)
            "HasAns_f1": (
                predictions,
                references,
            ),  # The F-score of predicted tokens versus the gold answer
            "NoAns_exact": (
                predictions,
                references,
            ),  # Exact match (the normalized answer exactly match the gold answer)
            "NoAns_f1": (
                predictions,
                references,
            ),  # The F-score of predicted tokens versus the gold answer
            "best_exact": (
                predictions,
                references,
            ),  # Best exact match (with varying threshold)
            "best_f1": (predictions, references),  # Best F1 (with varying threshold)
        }

    def aggregation(self):
        """
        :returns: {str: [float] -> float}
            A dictionary where keys are the names of submetrics and values are
            functions that aggregate a list of metrics
        """
        return {
            "exact": partial(
                _squad_agg, "exact"
            ),  # Exact match (the normalized answer exactly match the gold answer)
            "f1": partial(
                _squad_agg, "f1"
            ),  # The F-score of predicted tokens versus the gold answer
            "HasAns_exact": partial(
                _squad_agg, "HasAns_exact"
            ),  # Exact match (the normalized answer exactly match the gold answer)
            "HasAns_f1": partial(
                _squad_agg, "HasAns_f1"
            ),  # The F-score of predicted tokens versus the gold answer
            "NoAns_exact": partial(
                _squad_agg, "NoAns_exact"
            ),  # Exact match (the normalized answer exactly match the gold answer)
            "NoAns_f1": partial(
                _squad_agg, "NoAns_f1"
            ),  # The F-score of predicted tokens versus the gold answer
            "best_exact": partial(
                _squad_agg, "best_exact"
            ),  # Best exact match (with varying threshold)
            "best_f1": partial(
                _squad_agg, "best_f1"
            ),  # Best F1 (with varying threshold)
        }

    def higher_is_better(self):
        """
        :returns: {str: bool}
            A dictionary where keys are the names of submetrics and values are
            whether a higher value of the submetric is better
        """
        return {
            "exact": True,  # Exact match (the normalized answer exactly match the gold answer)
            "f1": True,  # The F-score of predicted tokens versus the gold answer
            "HasAns_exact": True,  # Exact match (the normalized answer exactly match the gold answer)
            "HasAns_f1": True,  # The F-score of predicted tokens versus the gold answer
            "NoAns_exact": True,  # Exact match (the normalized answer exactly match the gold answer)
            "NoAns_f1": True,  # The F-score of predicted tokens versus the gold answer
            "best_exact": True,  # Best exact match (with varying threshold)
            "best_f1": True,  # Best F1 (with varying threshold)
        }



class MLQA_ar(MLQA):  # Arabic
    DATASET_PATH = "GEM/xlsum"
    # TODO: Add the `DATASET_NAME` string. This is the name of a subset within
    # `DATASET_PATH`. If there aren't specific subsets you need, leave this as `None`.
    DATASET_NAME = 'arabic'
