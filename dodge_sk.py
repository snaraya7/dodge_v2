from abstract_dodge import *
from Experiments import *

"""
@author : Shrikanth N C
Implementation of the DODGE algorithm
DOI: 10.1109/TSE.2019.2945020
"""

from Experiments import *

class dodge_sk(abstract_dodge):

    def build_tree_of_options(self):
        """
        Overridden to customize
        :return:
        """

        mn = 1
        tree_of_options = []

        np.random.seed(mn)
        seed(mn)

        MLs =   [NB   , [KNN] * 5, [RF] * 5, [DT] * 5, [LR] * 5]

        # MLs = [NB   , [KNN] * 5, [RF] * 5, [DT] * 5, [LR] * 5]
        preprocess = [standard_scaler  , minmax_scaler, maxabs_scaler, [robust_scaler] * 10,
        [quantile_transform] * 10, normalizer, [binarize] * 10]

        # pre-processing error  Kernel matrix must be a square matrix. Input is a 50x14 matrix. KernelCenterer() KernelCenterer

        preprocess_list = unpack(preprocess)
        MLs_list = unpack(MLs)
        combine = [[r[0], r[1]] for r in product(preprocess_list, MLs_list)]

        for c in combine:
            node = Node(c[1]()[0], c[0]()[0])
            tree_of_options.append(node)

        self.tree_of_options = tree_of_options

    def compute_model_performance(self, node, train_changes, tune_changes, goal):
        """
        Overridden to customize
        :param node:
        :param train_changes:
        :param tune_changes:
        :param goal:
        :return:
        """


        current_score = node.get_error_score(goal)


        temp_train_changes = train_changes.copy(deep=True)
        temp_tune_changes = tune_changes.copy(deep=True)


        try:
            temp_train_changes = transform(temp_train_changes, node.preprocessor)
            temp_tune_changes = transform(temp_tune_changes, node.preprocessor)
        except Exception as e1:
            print("pre-processing error ", e1, node.preprocessor)


        try:
            trainX = temp_train_changes.drop(labels=['Buggy'], axis=1)
            trainY = temp_train_changes.Buggy

            node.classifier.fit(trainX, trainY)
            F = computeMeasures(temp_tune_changes, node.classifier, [], [0 for x in range(0, len(temp_tune_changes))])
            current_score = float(F[goal][0])

        except Exception as e2:
            print("\tclassifier error ", e2, node.classifier, node.preprocessor)
            # print(temp_train_changes)

        return current_score


if __name__ == '__main__':

    project_changes = release_manager.getProject('numpy').getAllChanges()
    region = project_changes.head(300)
    train, test = region.head(150), region.tail(150)

    _dodge = dodge_sk(train, test, 'd2h')

    print("Best Settings : ", _dodge.run())


