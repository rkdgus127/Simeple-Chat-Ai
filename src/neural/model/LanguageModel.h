#pragma once

#include <memory>
#include <vector>

#include "../../core/Tensor.h"
#include "../layers/Embedding.h"
#include "../layers/Linear.h"
#include "../encoding/PositionalEncoding.h"
#include "TransformerBlock.h"

class LanguageModel {
private:
    size_t vocabSize;
    size_t embeddingDim;
    size_t numLayers;

    std::unique_ptr<Embedding> embedding;
    std::unique_ptr<PositionalEncoding> positionalEncoding;

    std::vector<std::unique_ptr<TransformerBlock>> blocks;

    std::unique_ptr<Linear> outputLayer;

public:
    LanguageModel(
        size_t vocabSize,
        size_t embeddingDim,
        size_t numHeads,
        size_t numLayers,
        size_t hiddenDim = 0
    );

    std::shared_ptr<Tensor> forward(
        const std::vector<int>& tokens,
        bool causal = true
    );

    std::vector<std::shared_ptr<Tensor>> getParameters();
};