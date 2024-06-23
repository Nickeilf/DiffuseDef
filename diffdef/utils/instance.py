import copy
import json


class InputInstance(object):
    """
    use to store a piece of data
    contains:
        idx:    index of this instance
        text_a: first sentence
        text_b: second sentence (if agnews etc. : default None)
        label:  label (if test set: None)
    """

    def __init__(self, idx, text_a, text_b=None, label=None):
        self.idx = idx
        self.text_a = text_a
        self.text_b = text_b
        self.label = label

    def __repr__(self):
        return str(self.to_json_string())

    def __getitem__(self, key):
        return self.__dict__[key]

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2) + "\n"

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def get_tuple(self):
        if self.is_nli():
            return ((self.text_a, self.text_b), self.label)
        else:
            return (self.text_a, self.label)

    # def perturbable_sentence(self):
    #     if self.text_b is None:
    #         return self.text_a
    #     else:
    #         return self.text_b

    def is_nli(self):
        return self.text_b is not None

    # def length(self):
    #     return len(self.perturbable_sentence().split())

    def perturb(self):
        pass

    # @classmethod
    # def create_instance_with_perturbed_sentence(cls, instance: "InputInstance", perturb_sent: str):
    #     idx = instance.idx
    #     label = instance.label
    #     if instance.text_b is None:
    #         text_a = perturb_sent
    #         text_b = None
    #     else:
    #         text_a = instance.text_a
    #         text_b = perturb_sent
    #     return cls(idx, text_a, text_b, label)
