#pragma once

#include <memory>
#include <vector>

#include "../../core/Tensor.h"
#include "Linear.h"

class FeedForward {
private:
    size_t embeddingDim;
    size_t hiddenDim;

    std::unique_ptr<Linear> linear1;
    std::unique_ptr<Linear> linear2;

public:
    explicit FeedForward(size_t embeddingDim, size_t hiddenDim = 0);

    std::shared_ptr<Tensor> forward(
        const std::shared_ptr<Tensor>& input
    );

    std::vector<std::shared_ptr<Tensor>> getParameters();
};