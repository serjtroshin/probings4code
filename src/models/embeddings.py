import warnings
from functools import partial

from src.models.model import Model, ModelOutput


class RequiredEmbeddings:
    MEAN = "mean"
    DUMMY = "dummy"  # identity


class Embeddings:
    def __init__(self, type, pairsent=False):
        """
        type: ["s0", "mean", "dummy"]
            s0: first token
            mean: mean embedding by length
            dummy: return as is
        pairsent: sent=sent1 <s> sent2 if True, emb(sent1), emb(sent2) else
        """
        # assert type in Embeddings.available(), type
        self.type = type
        self.pairsent = pairsent

    # @classmethod
    # def available(cls):
    #     return ["s0", "mean", "dummy"]

    # f(list of hiddens) -> list of embeddings

    def process_hiddens(self, model_output: ModelOutput):
        layers = model_output.hiddens
        if self.type == "s0":
            layers = [hiddens[0, 0, :] for hiddens in layers]
        elif self.type == "mean":
            layers = [hiddens[0, :, :].mean(0) for hiddens in layers]
        elif self.type == "dummy":
            pass
        model_output.hiddens = layers
        # print("model_output.hiddens: ", model_output.hiddens[0].shape)
        # input()

    def __call__(self, model_output: ModelOutput):

        self.process_hiddens(model_output)
        return model_output.dump()

        # if "sent2" in code:
        #     if self.pairsent:
        #         return self.process_hiddens(model(code))
        #     else:
        #         return {
        #             "sent1": self.process_hiddens(model({"sent1": code["sent1"]})),
        #             "sent2": self.process_hiddens(model({"sent1": code["sent2"]})),
        #         }
        # else:
        #     return self.process_hiddens(model(code))
