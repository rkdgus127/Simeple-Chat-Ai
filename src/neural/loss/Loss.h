#pragma once

#include "../../core/Tensor.h"
#include <memory>

class Loss {
public:
    static std::shared_ptr<Tensor> mse(
        const std::shared_ptr<Tensor>& prediction,
        const std::shared_ptr<Tensor>& target
    );

    static std::shared_ptr<Tensor> crossEntropy(
        const std::shared_ptr<Tensor>& prediction,
        const std::vector<int>& targets
    );
};