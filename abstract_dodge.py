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

        self.weight = 1000
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


    def set_score(self, score):
        self.score = score

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
        self.mutated = 0
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

        worst_node = None
        worst_weight_so_far = best_node.weight

        for current_node in self.tree_of_options:

            if current_node.weight < worst_weight_so_far and self.is_same_type(best_node, current_node):
                worst_node = current_node
                worst_weight_so_far = current_node.weight


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

        # for k,v in best_params.items():
        #     print('best param = ', k,v)
        # for k,v in mutated_params.items():
        #     print('mutated param = ', k,v)

        mutated_classifier = new_classifier(best_classifier)
        mutated_classifier.set_params(**mutated_params)


        return mutated_classifier

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


        mutated_preprocessor = new_preprocessor(best_preprocessor)
        mutated_preprocessor.set_params(**best_params)

        return mutated_preprocessor


    def to_string(self, old_node):

        return str(old_node.classifier.get_params()) + '-' + str(old_node.preprocessor) + '-' + str(old_node.preprocessor.get_params())

    def is_new_node(self, new_node):


        for old_node in self.tree_of_options:

            if self.to_string(old_node) == self.to_string(new_node):
                return False

        return True

    def mutate(self, best_nodes):

        mutated_nodes = []
        # print("weighted nodes = ", [str(x.weight)+'-'+str(x.score)+'-'+str(self.name(x.classifier)) + '-' + str(self.name(x.preprocessor))
        #                             for x in self.tree_of_options])

        for best_node in best_nodes:

            worst_node = self.get_worst_node(best_node)

            if worst_node is not None:

                # print("wt best, worst = ", best_node.weight, worst_node.weight)
                # print("\t sc best, worst = ", best_node.score, worst_node.score)
                # print()

                mutated_node = Node(self.mutate_classifier(best_node.classifier, worst_node.classifier),
                                          self.mutate_preprocessor(best_node.preprocessor, worst_node.preprocessor))

                if self.is_new_node(mutated_node):

                    # print('worst node = ', worst_node)

                    mutated_nodes.append(mutated_node)
                    self.mutated += 1

        return mutated_nodes

    @abstractmethod
    def compute_model_performance(self, node, train_changes, tune_changes, goal):
        pass

    def evaluate(self, node, train_changes, tune_changes):

        current_score = self.compute_model_performance(node, train_changes, tune_changes, self.goal)
        previous_score = node.score

        if previous_score is not None and current_score != node.get_error_score(self.goal):

            delta = abs(previous_score - current_score)

            if delta > self.epsilon:
                if current_score < previous_score:
                    node.increment_weight()
                else:
                    node.decrement_weight()


        node.set_score(current_score)

    def get_best_settings(self):

        self.tree_of_options.sort(key=lambda x: x.score, reverse=False)
        print("Returning best setting...", self.tree_of_options[0].score, self.tree_of_options[len(self.tree_of_options) - 1].score )
        return self.tree_of_options[0]



    def evaluate_nodes(self):

        n1 = self.N1

        # print('Evaluating...')
        print('Tree of options = ', len(self.tree_of_options))

        while n1 > 0:
            print('n1 = ',n1, len(self.tree_of_options)) # , [str(x.score)+':'+str(x.weight) for x in self.tree_of_options])
            for node in self.tree_of_options:
                self.evaluate(node, self.train.sample(self.sample).copy(deep=True), self.test.copy(deep=True))


            n1 -= 1

        print('Mutating...')
        n2 = self.N2

        self.tree_of_options.sort(key=lambda x: x.weight, reverse=True)
        best_nodes = self.get_best_nodes()

        while n2 > 0:

            print('# best nodes = ', len(best_nodes)) # , [str(x.score)+':'+str(x.weight) for x in self.tree_of_options])

            print('n2 = ',n2, len(self.tree_of_options), 'mutated = ',self.mutated)

            mutated_nodes = self.mutate(best_nodes)

            for mutated_node in mutated_nodes:
                self.tree_of_options.append(mutated_node)
                self.evaluate(mutated_node, self.train.sample(self.sample).copy(deep=True), self.test.copy(deep=True))


            n2 -= 1


    def print_all_nodes(self):

        for n in self.tree_of_options:
            print(str(n))

    def run(self):

        print('Building tree of options...')
        self.build_tree_of_options()
        self.evaluate_nodes()

        return self.get_best_settings().classifier, self.get_best_settings().preprocessor


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


