import numpy as np
import pandas as pd


class Smoothing:
    def __init__(self, class_size = 8, window_size=3, weighting_method=None):
        self.window_size = window_size
        self.weighting = self.calculate_weighting(weighting_method)
        self.right_hand = np.zeros(shape=(window_size, class_size))
        self.left_hand = np.zeros(shape=(window_size, class_size))
        self.mapping_dictionary = dict()
        self.left_last_bbox = np.zeros(shape=(1, 4))
        self.right_last_bbox = np.zeros(shape=(1, 4))

    def calculate_weighting(self, method=None):
        if method == "log":
            unnormalized_weights = np.log(range(2, self.window_size + 2))
            return unnormalized_weights / np.sum(unnormalized_weights)
        else:
            return np.ones(self.window_size)/ self.window_size

    def smooth(self, curr_output):
        if curr_output.shape[0] == 2:
            return self.smooth_two_outputs(curr_output)
        elif curr_output.shape[0] == 1:
            return self.smooth_one_outputs(curr_output)

        else:
            print(curr_output.shape[0])


    def smooth_one_outputs(self, curr_output):
        """
        Receive yolo output, smooth and return predictions and smoothed confidence
        :param curr_output:
        :return: one dictionary containing bbox, prediction and confidence (these are the keys)
        """
        row = curr_output.to_numpy()[0, :]

        number_label, word_label = row[5], row[6]
        self.mapping_dictionary.update({number_label: word_label})

        distance_from_left = np.linalg.norm(self.left_last_bbox - row[:4])
        distance_from_right = np.linalg.norm(self.right_last_bbox - row[:4])
        type = ''
        if distance_from_left < distance_from_right:
            if self.left_hand.shape[0] >= self.window_size:
                self.left_hand = self.left_hand[1:, :]
            new_left_prediction = np.zeros(shape=(1, self.left_hand.shape[1]))
            new_left_prediction[0, row[5]] = row[4]
            self.left_hand = np.vstack([self.left_hand, new_left_prediction])

            left_smoothed_prediction = np.average(self.left_hand, axis=0, weights=self.weighting)
            left_prediction = np.argmax(left_smoothed_prediction)
            left_smoothed_confidence = np.max(left_smoothed_prediction)
            left_bbox = row[:4].astype(int)
            self.left_last_bbox = left_bbox
            one_return = {"bbox": left_bbox, "prediction": self.mapping_dictionary[left_prediction],
                           "confidence": left_smoothed_confidence, 'type': 'left'}
        else:
            if self.right_hand.shape[0] >= self.window_size:
                self.right_hand = self.right_hand[1:, :]
            new_right_prediction = np.zeros(shape=(1, self.right_hand.shape[1]))
            new_right_prediction[0, row[5]] = row[4]
            self.right_hand = np.vstack([self.right_hand, new_right_prediction])

            right_smoothed_prediction = np.average(self.right_hand, axis=0, weights=self.weighting)
            right_prediction = np.argmax(right_smoothed_prediction)
            right_smoothed_confidence = np.max(right_smoothed_prediction)
            right_bbox = row[:4].astype(int)
            self.right_last_bbox = right_bbox

            one_return = {"bbox": right_bbox, "prediction": self.mapping_dictionary[right_prediction],
                            "confidence": right_smoothed_confidence, 'type': 'right'}
        return [one_return]


    def smooth_two_outputs(self, curr_output):
        """
        Receive yolo output, smooth and return predictions and smoothed confidence
        :param curr_output:
        :return: two dictionaries (right, left) with contain bbox, prediction and confidence (these are the keys)
        """
        # associate current output lines to the correct array by x_min
        row_1 = curr_output.to_numpy()[0, :]
        row_2 = curr_output.to_numpy()[1, :]

        # add new labels to dictionary
        number_label, word_label = row_1[5], row_1[6]
        self.mapping_dictionary.update({number_label: word_label})
        number_label, word_label = row_2[5], row_2[6]
        self.mapping_dictionary.update({number_label: word_label})

        if row_1[0] < row_2[0]:
            right_hand_row = row_1
            left_hand_row = row_2
        else:
            right_hand_row = row_2
            left_hand_row = row_1

        if self.right_hand.shape[0] >= self.window_size:
            self.right_hand = self.right_hand[1:, :]
        new_right_prediction = np.zeros(shape=(1, self.right_hand.shape[1]))
        new_right_prediction[0, right_hand_row[5]] = right_hand_row[4]
        self.right_hand = np.vstack([self.right_hand, new_right_prediction])

        if self.left_hand.shape[0] >= self.window_size:
            self.left_hand = self.left_hand[1:, :]
        new_left_prediction = np.zeros(shape=(1, self.left_hand.shape[1]))
        new_left_prediction[0, left_hand_row[5]] = left_hand_row[4]
        self.left_hand = np.vstack([self.left_hand, new_left_prediction])

        right_smoothed_prediction = np.average(self.right_hand, axis=0, weights=self.weighting)
        right_prediction = np.argmax(right_smoothed_prediction)
        right_smoothed_confidence = np.max(right_smoothed_prediction)
        right_bbox = right_hand_row[:4].astype(int)
        self.right_last_bbox = right_bbox
        right_return = {"bbox": right_bbox, "prediction": self.mapping_dictionary[right_prediction],
                        "confidence": right_smoothed_confidence}

        left_smoothed_prediction = np.average(self.left_hand, axis=0, weights=self.weighting)
        left_prediction = np.argmax(left_smoothed_prediction)
        left_smoothed_confidence = np.max(left_smoothed_prediction)
        left_bbox = left_hand_row[:4].astype(int)
        self.left_last_bbox = left_bbox
        left_return = {"bbox": left_bbox, "prediction": self.mapping_dictionary[left_prediction],
                       "confidence": left_smoothed_confidence}

        return [right_return, left_return]




