from typing import Tuple, List, Callable, Iterator, Optional, Dict, Any
import torch

class SoftDice:
    def __init__(self, epsilon=1e-8):
        self.epsilon = epsilon

    def __call__(self, predictions: List[Dict[str, torch.Tensor]],
                 targets: List[Dict[str, torch.Tensor]]) -> torch.Tensor:
        numerator = torch.sum(2 * predictions * targets)
        denominator = torch.sum(predictions + targets)
        return numerator / (denominator + self.epsilon)

class Recall:
    def __init__(self, epsilon=1e-8, b=1):
        self.epsilon = epsilon
        self.a = b*b
    def __call__(self, predictions: List[Dict[str, torch.Tensor]],
                 targets: List[Dict[str, torch.Tensor]]):
        numerator = torch.sum(predictions * targets)
        denominator = torch.sum(predictions)
        return numerator / (denominator + self.epsilon)

class Accuracy:
    def __init__(self, epsilon=1e-8, b=1):
        self.epsilon = epsilon
        self.a = b*b
    def __call__(self, predictions: list, targets: list):
        numerator = torch.sum(predictions * targets)
        denominator = torch.sum(targets)
        return numerator / (denominator + self.epsilon)

class F1score:
    def __init__(self, accuracy_, recall_, epsilon=1e-8):
        self.accuracy_ = accuracy_
        self.recall_ = recall_
        self.epsilon = epsilon
    def __call__(self, predictions: list, targets: list):
        numerator = 2*self.accuracy_(predictions,targets)*self.recall_(predictions,targets)
        denominator = self.accuracy_(predictions,targets)+self.recall_(predictions,targets)
        return numerator / (denominator + self.epsilon)

def make_metrics():
    soft_dice = SoftDice()
    recall = Recall()
    accuracy = Accuracy()
    f1 = F1score(accuracy, recall)

    def exp_dice(pred, target):
        return soft_dice(torch.exp(pred[:, 1:]), target[:, 1:])

    def accuracy_(pred, target):
        return accuracy(torch.exp(pred[:, 1:]), target[:, 1:])

    def exp_recall(pred, target):
        return recall(torch.exp(pred[:, 1:]), target[:, 1:])

    def f1_score_(pred, target):
        return f1(torch.exp(pred[:, 1:]), target[:, 1:])

    return [('exp_dice', exp_dice),
            # ('accuracy', accuracy_),
            # ('recall', exp_recall),
            # ('f1', f1_score_)
            ]