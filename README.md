# Transformers and Language Models

This purpose of this repository is to act as an entry-point to the world of transformer-based language modelling in PyTorch. It is pitched to those of us that want to understand implementation details and theoretical insights, without having to wade through badly written research code 🙂

All code has been structured into a Python package called `modelling`, that is organised as follows:

```text
└── src
    ├── modelling
    │   ├── data.py
    │   ├── rnn.py
    │   ├── transformer.py
    │   └── utils.py
```

We have done our best to make this as readable as possible and comprehensively documented, so this is the place to go for the implementation details. To see this in action, use the following notebooks:

```text
notebooks
├── 0_attention_and_transformers.ipynb
├── 1_datasets_and_dataloaders.ipynb
├── 2_text_generation_with_rnns.ipynb
└── 3_text_generation_with_transformers.ipynb
└── 4_text_generation_with_transformers_and_bpe.ipynb
└── 5_transformers_for_search.ipynb
```

## Installing

To run the notebooks and use the code within the `src/modelling` directory either clone this repository and install the package directly from the source code,

```text
pip install .
```

Or install it directly from this repository,

```text
pip install git+https://github.com/AlexIoannides/transformers.git@main
```

## Useful Resources

We found the following useful in our ascent up the transformer and LLMs learning curve:

- [The Annotated Transformer](http://nlp.seas.harvard.edu/annotated-transformer/) - _Attention is all you Need_, the paper that introduced the transformer architecture for sequence-to-sequence modelling, annotated with PyTorch code snippets that demonstrate how to implement the concepts from first principles.
- [Transformers and Multi-Head Attention](https://lightning.ai/docs/pytorch/stable/notebooks/course_UvA-DL/05-transformers-and-MH-attention.html#Learning-rate-warm-up) - comprehensive tutorial from Lightning AI that demonstrates how to compose and train a simple generative language model using the latest techniques for training transformer models.
- [Language Modelling with `nn.Transformer` and torchtext](https://pytorch.org/tutorials/beginner/transformer_tutorial.html) - a tutorial from PyTorch that demonstrates how to use PyTorch's transformer layers to train a simple generative language model.
- [Transformer Architecture: The Positional Encoding](https://kazemnejad.com/blog/transformer_architecture_positional_encoding/) - a deep dive into positional encoding and its role transformer models.
