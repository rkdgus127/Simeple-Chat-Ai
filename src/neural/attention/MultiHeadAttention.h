#pragma once

#include <memory>
#include <vector>

#include "../../core/Tensor.h"
#include "../layers/Linear.h"

class MultiHeadAttention {
private:
    size_t embeddingDim;
    size_t numHeads;
    size_t headDim;

    std::vector<std::unique_ptr<Linear>> queryLayers;
    std::vector<std::unique_ptr<Linear>> keyLayers;
    std::vector<std::unique_ptr<Linear>> valueLayers;

    std::unique_ptr<Linear> outputLayer;

public:
    MultiHeadAttention(size_t embeddingDim, size_t numHeads);

    std::shared_ptr<Tensor> forward(
        const std::shared_ptr<Tensor>& input,
        bool causal = false
    );

    std::vector<std::shared_ptr<Tensor>> getParameters();
};