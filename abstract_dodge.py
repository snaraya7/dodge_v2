"""
@author : Shrikanth N C
Abstract DODGE algorithm
DOI: 10.1109/TSE.2019.2945020
"""

from abc import ABC, abstractmethod

from utilities import _randuniform, _randint, unpack
from ML import *
from itertools import product
from transformation import *
from random import seed


class Node(object):

    def __init__(self, classifier, preprocessor):

        self.weight = None
        self.classifier = classifier
        self.preprocessor = preprocessor
        self.score = None



    def name(self, item):
        return str(item.__class__.__name__)
    def get_classifier(self):
        return self.classifier

    def get_preprocessor(self):
        return self.preprocessor

    def get_weight(self):
        return self.weight

    def increment_weight(self):
        if self.weight is None:
            self.weight = 1
        else:
            self.weight += 1

    def decrement_weight(self):
        if self.weight is None:
            self.weight = -1
        else:
            self.weight -= 1

    def get_score(self):
        return self.score

    def get_error_score(self, goal):
        if goal == 'd2h':
            return 1

        float('goal undefined')



    def __str__(self):
        return self.name(self.classifier) + ' - ' + self.name(self.preprocessor) + ' weight: '+str(self.weight) + " score: "+str(self.score) +\
    str(self.classifier.get_params())+ " - "+str(self.preprocessor.get_params())


class abstract_dodge(ABC):

    def __init__(self, train, test, goal, epsilon=0.2, N1=12, N2=30, sample=50):
        self.goal = goal
        self.N1 = N1
        self.N2 = N2
        self.epsilon = epsilon
        self.tree_of_options = []
        self.train = train
        self.test = test
        self.sample = sample
        super().__init__()

    @abstractmethod
    def build_tree_of_options(self):
        pass

    def get_best_nodes(self):

        best_nodes = []

        if len(self.tree_of_options) > 0:

            best_weight = self.tree_of_options[0].get_weight()

            for node in self.tree_of_options:
                if node.get_weight() == best_weight:
                    best_nodes.append(node)
                else:
                    break

        return best_nodes

    def is_same_type(self, best_node, worst_node):

        return worst_node.classifier.__class__ == best_node.classifier.__class__ \
                and worst_node.preprocessor.__class__ == best_node.preprocessor.__class__

    def get_worst_node(self, best_node):

        best_score = best_node.score
        worst_node = None

        for top in self.tree_of_options:

            if top.score > best_score and self.is_same_type(best_node, top):
                worst_node = top
                best_score = top.score


        return worst_node



        # head_position = self.tree_of_options.index(best_node)
        #
        # tail_position = len(self.tree_of_options) - 1
        #
        # while tail_position > head_position:
        #
        #     worst_node = self.tree_of_options[tail_position]
        #
        #     if self.is_same_type(best_node, worst_node):
        #         return worst_node
        #
        #     tail_position -= 1
        #
        # return None

    def mutate_classifier(self, best_classifier, worst_classifier):

        best_params = best_classifier.get_params()
        worst_params = worst_classifier.get_params()
        mutated_params = {}

        for k, v in best_params.items():

            if isinstance(v, float):
                mutated_params[k] = _randuniform(best_params[k], (best_params[k] + worst_params[k]) / 2)
            elif isinstance(v, int):
                mutated_params[k] = _randint(best_params[k], int((best_params[k] + worst_params[k]) / 2 ) )
            else:
                mutated_params[k] = best_params[k]


        best_classifier.set_params(**mutated_params)

        return best_classifier

    def name(self, item):
        return str(item.__class__.__name__)

    def mutate_preprocessor(self, best_preprocessor, worst_preprocessor):

        best_params = best_preprocessor.get_params()
        worst_params = worst_preprocessor.get_params()
        mutated_params = {}

        for k, v in best_params.items():

            if isinstance(v, float):
                mutated_params[k] = _randuniform(best_params[k], (best_params[k] + worst_params[k]) / 2)
            elif isinstance(v, int):
                mutated_params[k] = _randint(best_params[k], (best_params[k] + worst_params[k]) / 2)
            else:
                mutated_params[k] = best_params[k]

            best_preprocessor.set_params(**mutated_params)
            return best_preprocessor
        else:
            return best_preprocessor

    def mutate(self):

        mutated_nodes = []

        self.tree_of_options.sort(key=lambda x: x.weight, reverse=True)

        print("weighted nodes = ", [str(x.weight)+'-'+str(x.score)+'-'+str(self.name(x.classifier)) + '-' + str(self.name(x.preprocessor))
                                    for x in self.tree_of_options])

        best_nodes = self.get_best_nodes()

        print('# best nodes = ', len(best_nodes))


        for best_node in best_nodes:

            worst_node = self.get_worst_node(best_node)
            print('worst node = ',worst_node)

            if worst_node is not None:
                mutated_nodes.append(Node(self.mutate_classifier(best_node.classifier, worst_node.classifier),
                                          self.mutate_preprocessor(best_node.preprocessor, worst_node.preprocessor)))

        return mutated_nodes

    @abstractmethod
    def compute_model_performance(self, node, train_changes, tune_changes, goal):
        pass

    def evaluate(self, node, train_changes, tune_changes):

        current_score = self.compute_model_performance(node, train_changes, tune_changes, self.goal)
        previous_score = node.score

        if node.score is None:
            node.score = current_score
            node.increment_weight()

        elif current_score != node.get_error_score(self.goal):

            delta = abs(previous_score - current_score)

            if delta > self.epsilon:
                if current_score < previous_score:
                    node.increment_weight()
            else:
                node.decrement_weight()

        node.score = current_score

    def get_best_settings(self):

        print("Returning best setting...")

        self.tree_of_options.sort(key=lambda x: x.score, reverse=False)
        return self.tree_of_options[0]



    def evaluate_nodes(self):

        n1 = self.N1

        print('Evaluating...')
        print('Tree of options = ', len(self.tree_of_options))

        while n1 > 0:
            print('n1 = ',n1)
            for node in self.tree_of_options:
                self.evaluate(node, self.train.sample(self.sample).copy(deep=True), self.test.copy(deep=True))


            n1 -= 1

        print('Mutating...')
        n2 = self.N2

        previous_mutated_node = None
        while n2 > 0:

            print('n2 = ',n2)

            mutated_nodes = self.mutate()

            if len(mutated_nodes) == 0:
                break


            for mutated_node in mutated_nodes:


                if self.is_same_node(mutated_node, previous_mutated_node) == False  :

                    self.tree_of_options.append(mutated_node)
                    self.evaluate(mutated_node, self.train.sample(self.sample).copy(deep=True), self.test.copy(deep=True))
                else:
                    print("Nothing to mutate...")

                previous_mutated_node = mutated_node

            n2 -= 1


    def print_all_nodes(self):

        for n in self.tree_of_options:
            print(str(n))

    def run(self):

        print('Building tree of options...')
        self.build_tree_of_options()



        self.evaluate_nodes()


        print(self.get_best_settings())
        print("Returned best setting")

    def is_same_node(self, mutated_node, previous_mutated_node):

        if previous_mutated_node is None and mutated_node is not None:
            return False
        elif previous_mutated_node is not None and mutated_node is None:
            return False
        else:

            if str(previous_mutated_node.classifier.get_params()) != str(mutated_node.classifier.get_params()):
                return False
            elif str(previous_mutated_node.preprocessor.get_params()) != str(mutated_node.preprocessor.get_params()):
                return False
            else:
                return True














