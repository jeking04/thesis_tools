
class BanditAlgorithm(object):

    def __init__(self, arm_set, budget, data_logger=None):
        self.arm_set = arm_set
        self.budget = budget
        self.data_logger = data_logger
