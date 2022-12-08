import numpy as np
import pandas as pd


class Smoothing:
    def __init__(self, class_size = 8, window_size=3, confidence_weighting_method=None, bbox_weighting_method=None):
        self.window_size = window_size
        self.confidence_weighting = self.calculate_weighting(confidence_weighting_method)
        self.bbox_weighting = self.calculate_weighting(bbox_weighting_method)
        self.right_hand_confidence_history = np.zeros(shape=(window_size, class_size))
        self.left_hand_confidence_history = np.zeros(shape=(window_size, class_size))
        self.right_hand_bbox_history = np.zeros(shape=(window_size, 4))
        self.left_hand_bbox_history = np.zeros(shape=(window_size, 4))
        self.mapping_dictionary = dict()
        self.last_left_return = {"bbox": np.array([0, 1, 2, 3]), "prediction": "Left_Empty", "confidence": 1}
        self.last_right_return = {"bbox": np.array([0, 1, 2, 3]), "prediction": "Right_Empty", "confidence": 1}
        self.first_left_prediction = True
        self.first_right_prediction = True

    def calculate_weighting(self, method=None):
        if method == "log":
            unnormalized_weights = np.log(range(2, self.window_size + 2))
            return unnormalized_weights / np.sum(unnormalized_weights)
        elif method == "linear":
            unnormalized_weights = np.array(range(2, self.window_size + 2))
            return unnormalized_weights / np.sum(unnormalized_weights)
        elif method == "exp":
            unnormalized_weights = np.exp(range(2, self.window_size + 2))
            return unnormalized_weights / np.sum(unnormalized_weights)
        elif method == "super-linear":
            unnormalized_weights = np.array(range(2, self.window_size + 2)) + np.log(range(2, self.window_size + 2))
            return unnormalized_weights / np.sum(unnormalized_weights)
        else:
            last_frame = np.zeros(self.window_size)
            last_frame[-1] = 1.0
            return last_frame

    def smooth(self, curr_output):
        one_per_arm = self.one_per_arm(curr_output)
        if one_per_arm.shape[0] == 2:
            return self.smooth_two_outputs(one_per_arm)
        elif one_per_arm.shape[0] == 1:
            return self.smooth_one_outputs(one_per_arm)
        elif one_per_arm.shape[0] == 0:
            return self.smooth_zero_outputs(one_per_arm)
        # else:
        #     return self.smooth_two_plus_outputs(curr_output)

    # def smooth_two_plus_outputs(self, curr_output):
    #     """
    #     infer which outputs are for which hands like in smooth_one_outputs
    #     Then for each hand take the prediction with the highest confidence
    #     and pass to smooth_two_outputs (efficient coding)
    #     :param curr_output:
    #     :return:
    #     """
    #
    #     print("used two plus")
    #     left_indices = list()
    #     right_indices = list()
    #     for index, row in curr_output.iterrows():
    #         distance_from_left = np.linalg.norm(self.last_left_return["bbox"] - row[:4])
    #         distance_from_right = np.linalg.norm(self.last_right_return["bbox"] - row[:4])
    #         if distance_from_left < distance_from_right:
    #             left_indices.append(index)
    #         else:
    #             right_indices.append(index)
    #     if len(left_indices) != 0:
    #         left_df = curr_output[curr_output.index.isin(left_indices)]
    #         left_max_confidence_df = left_df[left_df.iloc[:, 4] == left_df.iloc[:, 4].max()]
    #     if len(right_indices) != 0:
    #         right_df = curr_output[curr_output.index.isin(right_indices)]
    #         right_max_confidence_df = right_df[right_df.iloc[:, 4] == right_df.iloc[:, 4].max()]
    #
    #     # if there is a prediction for both arms tran use smoothing for two arms
    #     if len(left_indices) != 0 and len(right_indices) != 0:
    #         return self.smooth_two_outputs(pd.concat([left_max_confidence_df, right_max_confidence_df], axis=0))
    #     elif len(left_indices) != 0:
    #         return self.smooth_one_outputs(left_max_confidence_df)
    #     else:
    #         return self.smooth_one_outputs(right_max_confidence_df)


    def smooth_zero_outputs(self, curr_output):
        """
        Assuming that both hands really are present, we reuse the last known predictions and bbox of the hands
        :return: two dictionaries (left, right) containing bbox, prediction and confidence (these are the keys)
        """
        return self.last_left_return, self.last_right_return

    def smooth_one_outputs(self, curr_output):
        """
        Receive yolo output, smooth and return predictions and smoothed confidence
        Infers right or left based on the last known position of each hand
        The hand without a prediction is treated as unchanged.
        This is a calculated risk based on domain knowledge that both hands are in the video for nearly all time
        :param curr_output:
        :return: two dictionaries (left, right) containing bbox, prediction and confidence (these are the keys)
        """
        row = curr_output.to_numpy()[0, :]

        number_label, word_label = row[5], row[6]
        self.mapping_dictionary.update({number_label: word_label})

        # distance_from_left = np.linalg.norm(self.last_left_return["bbox"] - row[:4])
        # distance_from_right = np.linalg.norm(self.last_right_return["bbox"] - row[:4])
        #
        # if distance_from_left < distance_from_right:
        prediction_label = row[6]
        hand, _ = prediction_label.split('_', maxsplit=1)
        if hand.lower() == "left":
            self.update_left_hand(row)
            # self.left_hand_confidence_history = self.left_hand_confidence_history[1:, :]
            # new_left_prediction = np.zeros(shape=(1, self.left_hand_confidence_history.shape[1]))
            # new_left_prediction[0, row[5]] = row[4]
            # self.left_hand_confidence_history = np.vstack([self.left_hand_confidence_history, new_left_prediction])
            #
            # left_smoothed_prediction = np.average(self.left_hand_confidence_history, axis=0, weights=self.weighting)
            # left_prediction = np.argmax(left_smoothed_prediction)
            # left_smoothed_confidence = np.max(left_smoothed_prediction)
            # left_bbox = row[:4].astype(int)
            # self.last_left_return = {"bbox": left_bbox, "prediction": self.mapping_dictionary[left_prediction],
            #                "confidence": left_smoothed_confidence}
        else:
            self.update_right_hand(row)
            # self.right_hand_confidence_history = self.right_hand_confidence_history[1:, :]
            # new_right_prediction = np.zeros(shape=(1, self.right_hand_confidence_history.shape[1]))
            # new_right_prediction[0, row[5]] = row[4]
            # self.right_hand_confidence_history = np.vstack([self.right_hand_confidence_history, new_right_prediction])
            #
            # right_smoothed_prediction = np.average(self.right_hand_confidence_history, axis=0, weights=self.weighting)
            # right_prediction = np.argmax(right_smoothed_prediction)
            # right_smoothed_confidence = np.max(right_smoothed_prediction)
            # right_bbox = row[:4].astype(int)
            # self.right_last_bbox = right_bbox
            # self.last_right_return = {"bbox": right_bbox, "prediction": self.mapping_dictionary[right_prediction],
            #                 "confidence": right_smoothed_confidence}
        return self.last_left_return, self.last_right_return


    def smooth_two_outputs(self, curr_output):
        """
        Receive yolo output, smooth and return predictions and smoothed confidence
        :param curr_output:
        :return: two dictionaries (left, right) with bbox, prediction and confidence (these are the keys)
        """
        # associate current output lines to the correct array by x_min
        row_1 = curr_output.to_numpy()[0, :]
        row_2 = curr_output.to_numpy()[1, :]

        # add new labels to dictionary
        number_label, word_label = row_1[5], row_1[6]
        self.mapping_dictionary.update({number_label: word_label})
        number_label, word_label = row_2[5], row_2[6]
        self.mapping_dictionary.update({number_label: word_label})

        # if row_1[2] < row_2[2]:
        #     right_hand_row = row_1
        #     left_hand_row = row_2
        # else:
        #     right_hand_row = row_2
        #     left_hand_row = row_1

        prediction_label = row_1[6]
        hand, _ = prediction_label.split('_', maxsplit=1)
        if hand.lower() == "left":
            right_hand_row = row_2
            left_hand_row = row_1
        else:
            right_hand_row = row_1
            left_hand_row = row_2
        self.update_left_hand(left_hand_row)
        self.update_right_hand(right_hand_row)
        # if self.right_hand_confidence_history.shape[0] >= self.window_size:
        #     self.right_hand_confidence_history = self.right_hand_confidence_history[1:, :]
        # new_right_prediction = np.zeros(shape=(1, self.right_hand_confidence_history.shape[1]))
        # new_right_prediction[0, right_hand_row[5]] = right_hand_row[4]
        # self.right_hand_confidence_history = np.vstack([self.right_hand_confidence_history, new_right_prediction])
        #
        # if self.left_hand_confidence_history.shape[0] >= self.window_size:
        #     self.left_hand_confidence_history = self.left_hand_confidence_history[1:, :]
        # new_left_prediction = np.zeros(shape=(1, self.left_hand_confidence_history.shape[1]))
        # new_left_prediction[0, left_hand_row[5]] = left_hand_row[4]
        # self.left_hand_confidence_history = np.vstack([self.left_hand_confidence_history, new_left_prediction])
        #
        # right_smoothed_prediction = np.average(self.right_hand_confidence_history, axis=0, weights=self.weighting)
        # right_prediction = np.argmax(right_smoothed_prediction)
        # right_smoothed_confidence = np.max(right_smoothed_prediction)
        # right_bbox = right_hand_row[:4].astype(int)
        # self.last_right_return = {"bbox": right_bbox, "prediction": self.mapping_dictionary[right_prediction],
        #                 "confidence": right_smoothed_confidence}
        #
        # left_smoothed_prediction = np.average(self.left_hand_confidence_history, axis=0, weights=self.weighting)
        # left_prediction = np.argmax(left_smoothed_prediction)
        # left_smoothed_confidence = np.max(left_smoothed_prediction)
        # left_bbox = left_hand_row[:4].astype(int)
        # self.last_left_return = {"bbox": left_bbox, "prediction": self.mapping_dictionary[left_prediction],
        #                "confidence": left_smoothed_confidence}

        return self.last_left_return, self.last_right_return

    def update_left_hand(self, row):
        if self.first_left_prediction:
            self.first_left_prediction = False
            self.left_hand_bbox_history = np.vstack([row[:4].astype(int) for i in range(self.window_size)])

        # update confidence history for smoothing
        self.left_hand_confidence_history = self.left_hand_confidence_history[1:, :]
        new_left_prediction = np.zeros(shape=(1, self.left_hand_confidence_history.shape[1]))
        new_left_prediction[0, row[5]] = row[4]
        self.left_hand_confidence_history = np.vstack([self.left_hand_confidence_history, new_left_prediction])
        left_smoothed_prediction = np.average(self.left_hand_confidence_history, axis=0, weights=self.confidence_weighting)
        normalized_left_smoothed_prediction = left_smoothed_prediction / sum(left_smoothed_prediction)

        # update bbox history for smoothing
        self.left_hand_bbox_history = np.vstack([self.left_hand_bbox_history[1:, :], row[:4].astype(int)])
        left_smoothed_bbox = np.average(self.left_hand_bbox_history, axis=0, weights=self.bbox_weighting).astype(int)


        # get prediction and confidence
        left_prediction = np.argmax(normalized_left_smoothed_prediction)
        left_smoothed_confidence = np.max(normalized_left_smoothed_prediction)
        self.last_left_return = {"bbox": left_smoothed_bbox, "prediction": self.mapping_dictionary[left_prediction],
                                 "confidence": left_smoothed_confidence}

    def update_right_hand(self, row):
        if self.first_right_prediction:
            self.first_right_prediction = False
            self.right_hand_bbox_history = np.vstack([row[:4].astype(int) for i in range(self.window_size)])
        # update confidence history for smoothing
        self.right_hand_confidence_history = self.right_hand_confidence_history[1:, :]
        new_right_prediction = np.zeros(shape=(1, self.right_hand_confidence_history.shape[1]))
        new_right_prediction[0, row[5]] = row[4]
        self.right_hand_confidence_history = np.vstack([self.right_hand_confidence_history, new_right_prediction])
        right_smoothed_prediction = np.average(self.right_hand_confidence_history, axis=0, weights=self.confidence_weighting)
        normalized_right_smoothed_prediction = right_smoothed_prediction / sum(right_smoothed_prediction)

        # update bbox history for smoothing
        self.right_hand_bbox_history = np.vstack([self.right_hand_bbox_history[1:, :], row[:4].astype(int)])
        right_smoothed_bbox = np.average(self.right_hand_bbox_history, axis=0, weights=self.bbox_weighting).astype(int)

        # get prediction and confidence
        right_prediction = np.argmax(normalized_right_smoothed_prediction)
        right_smoothed_confidence = np.max(normalized_right_smoothed_prediction)
        self.last_right_return = {"bbox": right_smoothed_bbox, "prediction": self.mapping_dictionary[right_prediction],
                                 "confidence": right_smoothed_confidence}

    def one_per_arm(self, curr_output):
        left_indices = list()
        right_indices = list()
        for index, row in curr_output.iterrows():
            print(row[6])
            prediction_label = row[6]
            hand, _ = prediction_label.split('_', maxsplit=1)
            if hand.lower() == "left":
                left_indices.append(index)
            else:
                right_indices.append(index)
        if len(left_indices) != 0:
            left_df = curr_output[curr_output.index.isin(left_indices)]
            left_max_confidence_df = left_df[left_df.iloc[:, 4] == left_df.iloc[:, 4].max()]
        if len(right_indices) != 0:
            right_df = curr_output[curr_output.index.isin(right_indices)]
            right_max_confidence_df = right_df[right_df.iloc[:, 4] == right_df.iloc[:, 4].max()]

        # if there is a prediction for both arms tran use smoothing for two arms
        if len(left_indices) != 0 and len(right_indices) != 0:
            return pd.concat([left_max_confidence_df, right_max_confidence_df], axis=0)
        elif len(left_indices) != 0:
            return left_max_confidence_df
        elif len(left_indices) != 0:
            return right_max_confidence_df
        else:
            return curr_output



