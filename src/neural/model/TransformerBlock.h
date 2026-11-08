#pragma once

#include <memory>
#include <vector>

#include "../../core/Tensor.h"
#include "../attention/MultiHeadAttention.h"
#include "../normalization/LayerNorm.h"
#include "../layers/FeedForward.h"

class TransformerBlock {
private:
    size_t embeddingDim;
    size_t numHeads;
    size_t hiddenDim;

    std::unique_ptr<MultiHeadAttention> attention;

    std::unique_ptr<LayerNorm> norm1;
    std::unique_ptr<LayerNorm> norm2;

    std::unique_ptr<FeedForward> feedForward;

public:
    explicit TransformerBlock(
        size_t embeddingDim,
        size_t numHeads,
        size_t hiddenDim = 0
    );

    std::shared_ptr<Tensor> forward(
        const std::shared_ptr<Tensor>& input,
        bool causal = false
    );


    std::vector<std::shared_ptr<Tensor>> getParameters();
};