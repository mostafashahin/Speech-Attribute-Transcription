import jiwer.transforms as tr
from typing import Union, List, Tuple, Dict
import Levenshtein
import numpy as np
from collections import defaultdict
import pandas as pd

class phoneme_confusion_matrix:
    def __init__(self):
        self.default_transform = tr.Compose([
                                                tr.RemoveMultipleSpaces(),
                                                tr.Strip(),
                                                tr.ReduceToSingleSentence(),
                                                tr.ReduceToListOfListOfWords(),
                                            ])
        

    def preprocess(self, 
    truth: str,
    hypothesis: str,
) -> Tuple[str, str]:
        """
        Pre-process the truth and hypothesis into a form that Levenshtein can handle.
        :param truth: the ground-truth sentence(s) as a string or list of strings
        :param hypothesis: the hypothesis sentence(s) as a string or list of strings
        :param truth_transform: the transformation to apply on the truths input
        :param hypothesis_transform: the transformation to apply on the hypothesis input
        :return: the preprocessed truth and hypothesis
        """
    
        # Apply transforms. By default, it collapses input to a list of words
        truth = self.default_transform(truth)
        hypothesis = self.default_transform(hypothesis)
        assert len(truth) == 1
        assert len(hypothesis) == 1
    
        truth = truth[0]
        hypothesis = hypothesis[0]
    
        mapper = dict([(k,v) for v,k in enumerate(set(truth + hypothesis))])
        #print(truth,hypothesis)
        truth = list(map(lambda x : mapper[x], truth))
        hypothesis = list(map(lambda x : mapper[x], hypothesis))
        #print(truth,hypothesis)
        truth_chars = [chr(p) for p in truth]
        hypothesis_chars = [chr(p) for p in hypothesis]
    
        truth_str = "".join(truth_chars)
        hypothesis_str = "".join(hypothesis_chars)
    
    
        return truth_str, hypothesis_str

    
    def compute(self, ref: List, pred: List) -> Dict:
        cm = defaultdict(lambda : defaultdict(int))
        assert len(ref) == len(pred), "ref and pred list should be the same length"
        for ref_str, pred_str in zip(ref,pred):
            ref = ref_str.split()
            pred = pred_str.split()
            truth, hypothesis = self.preprocess(ref_str, pred_str)
            ops = Levenshtein.editops(truth, hypothesis)
            for op, ref_i, pred_i in ops:
                if op == 'insert':
                    cm[pred[pred_i]]['insert'] += 1
                elif op == 'delete':
                    cm[ref[ref_i]]['delete'] += 1
                else:
                    cm[ref[ref_i]][pred[pred_i]] += 1
        self.cm = cm
        return cm

    def save_cm(self, output_file: str) -> None:
        if not hasattr(self, 'cm'):
            raise AttributeError("Please compute cm first")
        #Get all phonemes
        list_phonemes = set(list(self.cm.keys()) + [p for i in self.cm.keys() for p in self.cm[i].keys()])
        complete_cm = dict([p, dict([(p,0) for p in list_phonemes])] for p in list_phonemes)
        for k1, v1 in self.cm.items():
            for k2, v2 in v1.items():
                complete_cm[k2][k1] = v2
        df_cm = pd.DataFrame.from_dict(complete_cm)
        df_cm.drop(index=['insert','delete'], inplace=True)
        df_cm.to_csv(output_file)