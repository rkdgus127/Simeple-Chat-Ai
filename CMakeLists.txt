cmake_minimum_required(VERSION 4.4)

project(Ai LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CXX_SCAN_FOR_MODULES OFF)

add_executable(Ai
        src/main.cpp
        src/core/Tensor.cpp
        src/core/Tensor.h
        src/core/Autograd.cpp
        src/core/Autograd.h
        src/neural/layers/Linear.cpp
        src/neural/layers/Linear.h
        src/neural/layers/Activation.cpp
        src/neural/layers/Activation.h
        src/neural/loss/Loss.cpp
        src/neural/loss/Loss.h
        src/optimizer/Adam.cpp
        src/optimizer/Adam.h
        src/neural/model/Sequential.cpp
        src/neural/model/Sequential.h
        src/tokenizer/Tokenizer.cpp
        src/tokenizer/Tokenizer.h
        src/tokenizer/BPETokenizer.cpp
        src/tokenizer/BPETokenizer.h
        src/neural/layers/Embedding.cpp
        src/neural/layers/Embedding.h
        src/neural/encoding/PositionalEncoding.cpp
        src/neural/encoding/PositionalEncoding.h
        src/neural/attention/SelfAttention.cpp
        src/neural/attention/SelfAttention.h
        src/neural/attention/MultiHeadAttention.cpp
        src/neural/attention/MultiHeadAttention.h
        src/neural/normalization/LayerNorm.cpp
        src/neural/normalization/LayerNorm.h
        src/neural/layers/FeedForward.cpp
        src/neural/layers/FeedForward.h
        src/neural/model/TransformerBlock.cpp
        src/neural/model/TransformerBlock.h
        src/neural/model/LanguageModel.cpp
        src/neural/model/LanguageModel.h
        src/training/Trainer.cpp
        src/training/Trainer.h
        src/neural/model/ModelSaver.cpp
        src/neural/model/ModelSaver.h
        src/generation/TextGenerator.cpp
        src/generation/TextGenerator.h
        src/dataset/Dataset.cpp
        src/dataset/Dataset.h
)