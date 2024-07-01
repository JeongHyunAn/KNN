import numpy as np
from knn import KNN

############################################################################
# DO NOT MODIFY ABOVE CODES
############################################################################


# TODO: implement F1 score
def f1_score(real_labels, predicted_labels):
    """
    Information on F1 score - https://en.wikipedia.org/wiki/F1_score
    :param real_labels: List[int]
    :param predicted_labels: List[int]
    :return: float
    """
    assert len(real_labels) == len(predicted_labels)
    # recall_val = 0
    # precision_val = 0
    # TP = 0
    # FP = 0
    # FN = 0

    # print('real : ', real_labels, ', predicted : ', predicted_labels)

    # for real, prediction in zip(real_labels, predicted_labels):
    #     if real == 1 and real == prediction:
    #         TP += 1
        
    #     elif real == 0 and real != prediction:
    #         FN += 1
        
    #     elif real == 0 and real != prediction:
    #         FP += 1
    
    # recall_val = TP / TP + FN
    # precision_val = TP / TP + FP

    # return float(2 * recall_val * precision_val / float(recall_val) + float(precision_val))

    denominator = float(sum(real_labels) + sum(predicted_labels))
    numerator = float(2 * sum(np.multiply(real_labels, predicted_labels)))

    return float(numerator / denominator)
    
    # raise NotImplementedError


class Distances:
    @staticmethod
    # TODO
    def canberra_distance(point1, point2):
        """
        :param point1: List[float]
        :param point2: List[float]
        :return: float
        """
        # c_dist = 0
        # for x, y in zip(point1, point2):

        #     # print('x : ', np.absolute(x), ', y : ', np.absolute(y))
        #     if np.absolute(x) + np.absolute(y) == 0:
        #         c_dist += 0

        #     else:
        #         c_dist += np.absolute(x - y) / np.absolute(x) + np.absolute(y)
        
        # return float(c_dist)

        denominator = np.absolute(point1) + np.absolute(point2)
        numerator = np.absolute(np.subtract(point1, point2))

        division = numerator / denominator
        division[np.isnan(division)] = 0
        
        return sum(division)

        # return sum(np.divide(numerator, denominator, out = np.zeros_like(denominator), where = denominator != 0, casting = 'unsafe'))

        # raise NotImplementedError

    @staticmethod
    # TODO
    def minkowski_distance(point1, point2):
        """
        Minkowski distance is the generalized version of Euclidean Distance
        It is also know as L-p norm (where p>=1) that you have studied in class
        For our assignment we need to take p=3
        Information on Minkowski distance - https://en.wikipedia.org/wiki/Minkowski_distance
        :param point1: List[float]
        :param point2: List[float]
        :return: float
        """
        m_dist = 0
        for x, y in zip(point1, point2):
            m_dist += np.absolute(x - y) ** 3
        
        return float(np.cbrt(m_dist))

        # raise NotImplementedError

    @staticmethod
    # TODO
    def euclidean_distance(point1, point2):
        """
        :param point1: List[float]
        :param point2: List[float]
        :return: float
        """
        # e_dist = 0
        # for x, y in zip(point1, point2):
        #     e_dist += (x - y) ** 2
        
        # return float(np.sqrt(e_dist))

        return np.sqrt(np.dot(np.subtract(point1, point2), np.subtract(point1, point2)))
        # return np.linalg.norm(np.substract(point1, point2))

        # raise NotImplementedError

    @staticmethod
    # TODO
    def inner_product_distance(point1, point2):
        """
        :param point1: List[float]
        :param point2: List[float]
        :return: float
        """
        i_p_dist = 0
        for x, y in zip(point1, point2):
            i_p_dist += np.inner(x, y)

        return float(i_p_dist)

        # raise NotImplementedError

    @staticmethod
    # TODO
    def cosine_similarity_distance(point1, point2):
        """
        :param point1: List[float]
        :param point2: List[float]
        :return: float
        """
        norm_x = 0
        norm_y = 0
        product_sum = 0
        for x, y in zip(point1, point2):
            norm_x += x ** 2
            norm_y += y ** 2
            product_sum += x * y

        if not float(np.sqrt(norm_x) * np.sqrt(norm_y)):
            return 0

        return float(1 - float(product_sum) / float(np.sqrt(norm_x) * np.sqrt(norm_y)))

        # raise NotImplementedError

    @staticmethod
    # TODO
    def gaussian_kernel_distance(point1, point2):
        """
        :param point1: List[float]
        :param point2: List[float]
        :return: float
        """
        # g_k_dist = 0
        # for x, y in zip(point1, point2):
            # g_k_dist += -np.exp(-1/2 * (x - y) ** 2)
        
        # return float(g_k_dist)

        return -np.exp(-1/2 * (np.linalg.norm(np.subtract(point1, point2)) ** 2))

        # raise NotImplementedError


class HyperparameterTuner:
    def __init__(self):
        self.best_k = None
        self.best_distance_function = None
        self.best_scaler = None
        self.best_model = None

    # TODO: find parameters with the best f1 score on validation dataset
    def tuning_without_scaling(self, distance_funcs, x_train, y_train, x_val, y_val):
        """
        In this part, you should try different distance function you implemented in part 1.1, and find the best k.
        Use k range from 1 to 30 and increment by 2. Use f1-score to compare different models.

        :param distance_funcs: dictionary of distance functions you must use to calculate the distance.
            Make sure you loop over all distance functions for each data point and each k value.
            You can refer to test.py file to see the format in which these functions will be
            passed by the grading script
        :param x_train: List[List[int]] training data set to train your KNN model
        :param y_train: List[int] train labels to train your KNN model
        :param x_val:  List[List[int]] Validation data set will be used on your KNN predict function to produce
            predicted labels and tune k and distance function.
        :param y_val: List[int] validation labels

        Find(tune) best k, distance_function and model (an instance of KNN) and assign to self.best_k,
        self.best_distance_function and self.best_model respectively.
        NOTE: self.best_scaler will be None

        NOTE: When there is a tie, choose model based on the following priorities:
        Then check distance function  [canberra > minkowski > euclidean > gaussian > inner_prod > cosine_dist]
        If they have same distance fuction, choose model which has a less k.
        """
        
        # You need to assign the final values to these variables
        self.best_k = None
        self.best_distance_function = None
        self.best_model = None
        previous_f1 = 0
        previous_k = 'inf'

        for dist_func in distance_funcs:
            for k in range(1, 31, 2):
                selected_func = KNN(k, distance_funcs[dist_func])
                selected_func.train(x_train, y_train)
                prediction = selected_func.predict(x_val)
                # print('y_val : ', y_val, ', prediction : ', prediction)
                current_f1 = f1_score(y_val, prediction)

                if current_f1 > previous_f1:
                    self.best_k = k
                    self.best_distance_function = dist_func
                    self.best_model = selected_func
                    previous_f1 = current_f1

                elif current_f1 == previous_f1:
                    if previous_f1 > k:
                        self.best_k = k
                        self.best_distance_function = dist_func
                        self.best_model = selected_func
                        previous_f1 = current_f1

        # raise NotImplementedError

    # TODO: find parameters with the best f1 score on validation dataset, with normalized data
    def tuning_with_scaling(self, distance_funcs, scaling_classes, x_train, y_train, x_val, y_val):
        """
        This part is similar to Part 1.3 except that before passing your training and validation data to KNN model to
        tune k and disrance function, you need to create the normalized data using these two scalers to transform your
        data, both training and validation. Again, we will use f1-score to compare different models.
        Here we have 3 hyperparameters i.e. k, distance_function and scaler.

        :param distance_funcs: dictionary of distance funtions you use to calculate the distance. Make sure you
            loop over all distance function for each data point and each k value.
            You can refer to test.py file to see the format in which these functions will be
            passed by the grading script
        :param scaling_classes: dictionary of scalers you will use to normalized your data.
        Refer to test.py file to check the format.
        :param x_train: List[List[int]] training data set to train your KNN model
        :param y_train: List[int] train labels to train your KNN model
        :param x_val: List[List[int]] validation data set you will use on your KNN predict function to produce predicted
            labels and tune your k, distance function and scaler.
        :param y_val: List[int] validation labels

        Find(tune) best k, distance_funtion, scaler and model (an instance of KNN) and assign to self.best_k,
        self.best_distance_function, self.best_scaler and self.best_model respectively

        NOTE: When there is a tie, choose model based on the following priorities:
        For normalization, [min_max_scale > normalize];
        Then check distance function  [canberra > minkowski > euclidean > gaussian > inner_prod > cosine_dist]
        If they have same distance function, choose model which has a less k.
        """
        
        # You need to assign the final values to these variables
        self.best_k = None
        self.best_distance_function = None
        self.best_scaler = None
        self.best_model = None
        previous_f1 = 0
        previous_k = 'inf'

        for scaling_class in scaling_classes:
            # print('scaler : ', scaling_class)
            scaler = scaling_classes[scaling_class]()
            scaled_x_train = scaler(x_train)
            scaled_x_val = scaler(x_val)

            for dist_func in distance_funcs:
                for k in range(1, 31, 2):
                    selected_func = KNN(k, distance_funcs[dist_func])
                    selected_func.train(scaled_x_train, y_train)
                    prediction = selected_func.predict(scaled_x_val)
                    current_f1 = f1_score(y_val, prediction)

                    if current_f1 > previous_f1:
                        self.best_k = k
                        self.best_distance_function = dist_func
                        self.best_model = selected_func
                        previous_f1 = current_f1
                        self.best_scaler = scaling_class
                    
                    elif current_f1 == previous_f1:
                        if previous_f1 > k:
                            self.best_k = k
                            self.best_distance_function = dist_func
                            self.best_model = selected_func
                            previous_f1 = current_f1

        # raise NotImplementedError


class NormalizationScaler:
    def __init__(self):
        self.example_features = [[3, 4], [1, -1], [0, 0]]

    # TODO: normalize data
    def __call__(self, features):
        """
        Normalize features for every sample

        Example
        features = [[3, 4], [1, -1], [0, 0]]
        return [[0.6, 0.8], [0.707107, -0.707107], [0, 0]]

        :param features: List[List[float]]
        :return: List[List[float]]
        """

        # normalized_features = []

        # print('feature data : ', features)

        # for feature in self.example_features:
        # # for feature in features:
        #     print('feature : ', feature)
        #     row_features = []
        #     norm = 0

        #     for feat in feature:
        #         print('feat : ', feat)
        #         norm = np.sqrt(np.dot(feat, feat))
        #         print('norm : ', norm)

        #         if norm == 0:
        #             row_features.append(feat)

        #         else:
        #             row_features.append(feat / norm)

        #     normalized_features.append(row_features)


        normalized_features = []

        # for feature in self.example_features:
        for feature in features:
            # print('feature : ', feature)
            norm = 0

            norm = np.sqrt(np.dot(feature, feature))
            # print('norm : ', norm)

            if norm == 0:
                normalized_features.append(feature)

            else:
                normalized_features.append(feature / norm)


        # for x in range(data_size):
        #     row_features = []
        #     norm = 0

        #     for y in range(feature_size):
        #         norm = np.sqrt(np.dot(features[x][y], features[x][y]))

        #         row_features.append(features[x][y] if norm == 0 else features[x][y] / norm)

        #     # for y in range(feature_size):
        #     #     normalized_features[x][y] = 0 if features[x][y] == 0 else features[x][y] / norm

        #     normalized_features.append(row_features)
            
        # print('normalized_features : ', normalized_features)
        return normalized_features

        # raise NotImplementedError


class MinMaxScaler:
    """
    Please follow this link to know more about min max scaling
    https://en.wikipedia.org/wiki/Feature_scaling
    You should keep some states inside the object.
    You can assume that the parameter of the first __call__
    will be the training set.

    Hints:
        1. Use a variable to check for first __call__ and only compute
            and store min/max in that case.

    Note:
        1. You may assume the parameters are valid when __call__
            is being called the first time (you can find min and max).

    Example:
        train_features = [[0, 10], [2, 0]]
        test_features = [[20, 1]]

        scaler1 = MinMaxScale()
        train_features_scaled = scaler1(train_features)
        # train_features_scaled should be equal to [[0, 1], [1, 0]]

        test_features_scaled = scaler1(test_features)
        # test_features_scaled should be equal to [[10, 0.1]]

        new_scaler = MinMaxScale() # creating a new scaler
        _ = new_scaler([[1, 1], [0, 0]]) # new trainfeatures
        test_features_scaled = new_scaler(test_features)
        # now test_features_scaled should be [[20, 1]]

    """

    def __init__(self):
        self.check_first = 1
        self.min_val = []
        self.max_val = []

    def __call__(self, features):
        """
        normalize the feature vector for each sample . For example,
        if the input features = [[2, -1], [-1, 5], [0, 0]],
        the output should be [[1, 0], [0, 1], [0.333333, 0.16667]]

        :param features: List[List[float]]
        :return: List[List[float]]
        """

        data_size = len(features)
        feature_size = len(features[0])
        normalized_features = [[0] * feature_size for _ in range(data_size)]

        if self.check_first:
            self.min_val = [float('inf')] * feature_size
            self.max_val = [float('-inf')] * feature_size

        # print('data size : ', data_size)
        # print('feature size : ', feature_size)
        # print('min size : ', len(min_val))
        # print('features : ', features)

            for x in range(data_size):
                for y in range(feature_size):
                    val = features[x][y]
                    self.max_val[y] = max(self.max_val[y], val)
                    self.min_val[y] = min(self.min_val[y], val)

            self.check_first = 0
        # print('max : ', max_val)
        # print('min : ', min_val)
        
        for x in range(data_size):
            for y in range(feature_size):
                min_max = self.max_val[y] - self.min_val[y]
                normalized_features[x][y] = 0 if min_max == 0 else ((features[x][y] - self.min_val[y]) / min_max)
        
        return normalized_features

        # raise NotImplementedError
