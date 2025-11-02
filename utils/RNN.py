import itertools
import torch
import torch.nn as nn


class RNN(nn.Module):

    def __init__(self, model, weight_tying=True, **kwargs):
        super().__init__()
        self.model = model
        self.weight_tying = weight_tying  # 決定把 model output 當 context vector (True)，或是 log probability distribution of classes (False)
        self.softmax = nn.Softmax()

    def forward(self, data, batch_sizes, **kwargs):

        states = self.model.initial_states()

        states = {
            key: [
                state.expand(batch_sizes[1], *[-1] * (len(state.shape) - 1)) for state in statelist
            ]
            for key, statelist in states.items()
        }

        for t, (batch_accum, batch_size, next_batch_size) in enumerate(
            zip(itertools.accumulate(batch_sizes, initial=0), batch_sizes, batch_sizes[1:])
        ):
            batch = data[
                batch_accum : batch_accum + next_batch_size
            ]  # 只有有下個時間點 t+1 的 sequence 才有 ground truth 可以 train
            states = {key: [s[:next_batch_size] for s in state] for key, state in states.items()}
            predicts, states = self.model(batch, **states, time=t + 1)

            labels = data[batch_accum + batch_size : batch_accum + batch_size + next_batch_size]

            if self.weight_tying:
                WT = self.model.all_word_embeddings().T
                logit = predicts.real @ WT.real + predicts.imag @ WT.imag
                labels = nn.functional.one_hot(labels, num_classes=self.model.vocab_size).type(
                    torch.float
                )
                yield self.softmax(logit), labels
            else:
                yield predicts, self.model.embedding(labels)


class RNNWithBias(nn.Module):

    def __init__(self, model, weight_tying=True, return_label_embeddings=True, **kwargs):
        super().__init__()
        self.model = model
        self.weight_tying = weight_tying  # 決定把 model output 當 context vector (True)，或是 log probability distribution of classes (False)
        self.softmax = nn.Softmax()
        self.bias = nn.Parameter(torch.randn([model.vocab_size]))
        self.return_label_embeddings = return_label_embeddings

    def forward(self, data, batch_sizes, **kwargs):

        states = self.model.initial_states()

        states = {
            key: [
                state.expand(batch_sizes[1], *[-1] * (len(state.shape) - 1)) for state in statelist
            ]
            for key, statelist in states.items()
        }

        for t, (batch_accum, batch_size, next_batch_size) in enumerate(
            zip(itertools.accumulate(batch_sizes, initial=0), batch_sizes, batch_sizes[1:])
        ):
            batch = data[
                batch_accum : batch_accum + next_batch_size
            ]  # 只有有下個時間點 t+1 的 sequence 才有 ground truth 可以 train
            states = {key: [s[:next_batch_size] for s in state] for key, state in states.items()}
            predicts, states = self.model(batch, **states, time=t + 1)

            labels = data[batch_accum + batch_size : batch_accum + batch_size + next_batch_size]

            if self.weight_tying:
                WT = self.model.all_word_embeddings().T
                pmi = predicts.real @ WT.real + predicts.imag @ WT.imag
                labels = nn.functional.one_hot(labels, num_classes=self.model.vocab_size).type(
                    torch.float
                )
                yield self.softmax(pmi + self.bias), labels
            elif self.return_label_embeddings:
                yield predicts, self.model.embedding(labels)
            else:
                yield predicts, labels
