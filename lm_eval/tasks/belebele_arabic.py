"""
The Belebele Benchmark: a Parallel Reading Comprehension Dataset in 122 Language Variants.
https://arxiv.org/abs/2308.16884

Belebele is a multiple-choice machine reading comprehension (MRC) dataset spanning 122 language variants. This dataset enables the evaluation of mono- and multi-lingual models in high-, medium-, and low-resource languages. Each question has four multiple-choice answers and is linked to a short passage from the FLORES-200 dataset. The human annotation procedure was carefully curated to create questions that discriminate between different levels of generalizable language comprehension and is reinforced by extensive quality checks. While all questions directly relate to the passage, the English dataset on its own proves difficult enough to challenge state-of-the-art language models. Being fully parallel, this dataset enables direct comparison of model performance across all languages. Belebele opens up new avenues for evaluating and analyzing the multilingual abilities of language models and NLP systems.
Homepage: https://github.com/facebookresearch/belebele
"""
from lm_eval.base import MultipleChoiceTask


_CITATION = """
@article{bandarkar2023belebele,
      title={The Belebele Benchmark: a Parallel Reading Comprehension Dataset in 122 Language Variants},
      author={Lucas Bandarkar and Davis Liang and Benjamin Muller and Mikel Artetxe and Satya Narayan Shukla and Donald Husa and Naman Goyal and Abhinandan Krishnan and Luke Zettlemoyer and Madian Khabsa},
      year={2023},
      journal={arXiv preprint arXiv:2308.16884}
}
"""


class belebele_ar(MultipleChoiceTask):
    VERSION = 0
    DATASET_PATH = "arbml/belebele_arabic"

    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return False

    def has_test_docs(self):
        return True

    def test_docs(self):
        return map(self._process_doc, self.dataset["train"])

    def _process_doc(self, doc):
        doc_list = []
        choices = ''
        doc_choices = "أ." + doc['mc_answer1'] + "\n" + \
                   "ب." + doc['mc_answer2'] + "\n" + \
                   "ج." + doc['mc_answer3'] + "\n" + \
                   "د." + doc['mc_answer4']
        choices = ["أ", "ب","ج","د"]
        convert_num2ar = {'1':'أ', '2':'ب', '3':'ج', '4':'د'}

        #choices = "أ." + doc["option_a"] + "\n" + "ب." + doc['option_b'] + "\n" + "ج." + doc['option_c'] + "\n" + "د." + doc['option_d']
        out_doc = {
            "passage": doc["flores_passage"] + "\n",
            "query": "س: " + doc["question"] + "\nاختر الإجابة الصحيحة مما يلي:\n" + doc_choices,
            "choices": choices,
            "gold": choices.index(convert_num2ar[doc['correct_answer_num']]),
        }
        return out_doc

    def doc_to_text(self, doc):
        return doc["passage"] + doc["query"]

    def should_decontaminate(self):
        return True

    def doc_to_decontamination_query(self, doc):
        return doc["query"]
