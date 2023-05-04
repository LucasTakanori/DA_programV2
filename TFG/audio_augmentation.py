import random

class AudioAugmentation:
    def __init__(self, min_param, max_param):
        self.min_param = min_param
        self.max_param = max_param

    def apply(self, input_file, output_file):
        raise NotImplementedError()

    def randomize(self):
        return random.uniform(self.min_param, self.max_param)
