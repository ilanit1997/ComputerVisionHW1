import numpy as np
from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score
import pandas as pd
import pickle
import os

class MetricEvaluator:
    def __init__(self, left_df=None, right_df=None):
        if type(left_df) == type(None) and type(right_df) == type(None):
            print("Empty Evaluator")
            return
        # tool usage
        self.name_to_tool = {"no tool in hand" : "T0", "needle_driver": "T1", "forceps": "T2", "scissors":"T3"}
        self.tool_to_name = {"T0": "no tool in hand", "T1": "needle_driver","T2": "forceps","T3": "scissors"}
        self.yolo_to_label = {'empty': 0, 'needle_driver': 1, 'forceps': 2, 'scissors': 3}
        # left_df = pd.read_csv(left_ground_truth_path, sep=" ", header=None)
        # right_df = pd.read_csv(right_ground_truth_path, sep=" ", header=None)
        self.left_tool_labels, left_relevant_classes = self.convert_file_to_list(left_df)
        self.right_tool_labels, right_relevant_classes = self.convert_file_to_list(right_df)

        self.left_predictions = list()
        self.right_predictions = list()
        self.metric_history = {'f1': list(), 'accuracy': list(), 'recall': list(), 'precision': list(), 'f1_macro': list()}

        # used to construct the metric dictionary and for calculating the Macro-f1 score which averages over the total
        # number of relevant classes (relevant := has a corresponding ground truth)
        self.num_relevant_classes = len(left_relevant_classes.union(right_relevant_classes))

    def convert_tool_to_label(self, tool):
        tool_to_label = {"T0": 0, "T1": 1, "T2": 2, "T3": 3}
        return tool_to_label[tool]

    def convert_file_to_list(self, df):
        ground_truth = np.zeros(df.iloc[-1, 1]) # last row end time, maximum time
        for index, row in df.iterrows():
            ground_truth[row[0]:row[1]] = self.convert_tool_to_label(row[2])
        return ground_truth, set(ground_truth)

    def update_left_prediction(self, prediction):
        """
        Convert prediction to tool name as seen in tool usage dataset
        :param prediction:
        :return: none, changes self object
        """
        lower_case_pred = prediction.lower()
        self.left_predictions.append(self.yolo_to_label[lower_case_pred])

    def update_right_prediction(self, prediction):
        """
        Convert prediction to tool name as seen in tool usage dataset
        :param prediction:
        :return: none, changes self object
        """
        lower_case_pred = prediction.lower()
        self.right_predictions.append(self.yolo_to_label[lower_case_pred])

    def convert_yolo_output_to_tool(self, yolo_output):
        hand, tool = yolo_output.split('_', maxsplit=1)

        if hand == "Left":
            self.update_left_prediction(tool)
        else:
            self.update_right_prediction(tool)

    def calculate_recall(self):
        # shear off the ground truths not yet predicted
        right_ground_truth = self.right_tool_labels[:len(self.right_predictions)]
        left_ground_truth = self.left_tool_labels[:len(self.left_predictions)]
        ground_truth = np.hstack([right_ground_truth, left_ground_truth])
        predictions = self.right_predictions + self.left_predictions

        return recall_score(ground_truth, predictions, labels=list(range(4)), average=None)

    def calculate_accuracy(self):
        # shear off the ground truths not yet predicted
        right_ground_truth = self.right_tool_labels[:len(self.right_predictions)]
        left_ground_truth = self.left_tool_labels[:len(self.left_predictions)]
        ground_truth = np.hstack([right_ground_truth, left_ground_truth])
        predictions = self.right_predictions + self.left_predictions

        return accuracy_score(ground_truth, predictions)

    def calculate_f1(self):
        # shear off the ground truths not yet predicted
        right_ground_truth = self.right_tool_labels[:len(self.right_predictions)]
        left_ground_truth = self.left_tool_labels[:len(self.left_predictions)]
        ground_truth = np.hstack([right_ground_truth, left_ground_truth])
        predictions = self.right_predictions + self.left_predictions

        return f1_score(ground_truth, predictions, labels=list(range(4)), average=None)

    def calculate_precision(self):
        # shear off the ground truths not yet predicted
        right_ground_truth = self.right_tool_labels[:len(self.right_predictions)]
        left_ground_truth = self.left_tool_labels[:len(self.left_predictions)]
        ground_truth = np.hstack([right_ground_truth, left_ground_truth])
        predictions = self.right_predictions + self.left_predictions

        return precision_score(ground_truth, predictions, labels=list(range(4)), average=None)

    def calculate_macro_f1_score(self):
        f1_scores = self.calculate_f1()
        return np.sum(f1_scores) / self.num_relevant_classes

    def calculate_all_metrics(self):
        self.metric_history['recall'].append(self.calculate_recall())
        self.metric_history['accuracy'].append(self.calculate_accuracy())
        self.metric_history['precision'].append(self.calculate_precision())
        self.metric_history['f1'].append(self.calculate_f1())
        self.metric_history['f1_macro'].append(self.calculate_macro_f1_score())

    def history_to_pickle(self, destination):
        # save metric history
        with open(destination + '.pkl', 'wb') as handle:
            pickle.dump(self.metric_history, handle, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def print_metric_statistics(source=None, destination=None):
        """
        If passed string, assume this is the pickle and then load history before printing last entries
        If passed dictionary, then print last entries
        :param source:
        :return:
        """
        if type(source) == str:
            with open(source, 'rb') as f:
                history_dictionary = pickle.load(f)
        elif type(source) == dict:
            history_dictionary = source
        else:
            print("Pass me a path to a pickle or a dictionary! I like salty foods and books. Sue me.")
            return

        # load history
        history_df = pd.DataFrame.from_dict(history_dictionary)

        if type(destination) == str:
            history_df.iloc[-1, :].to_csv(destination + '.csv')
        else:
            print(history_df.iloc[-1, :].head())


def main():
    metri = MetricEvaluator()
    for f in os.listdir('experiments'):
        print(f)
        metri.print_metric_statistics('experiments/' + f)
    # metri.print_metric_statistics('experiments/P026_tissue1window_size25_smoothingsuper-linear.pkl')
    # use window size 25 and log smoothing

if __name__ == "__main__":
    main()




