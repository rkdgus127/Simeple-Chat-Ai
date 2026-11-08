#pragma once

#include "../core/Tensor.h"
#include <memory>

class Activation {
public:
    static std::shared_ptr<Tensor> relu(
        const std::shared_ptr<Tensor>& input
    );

    static std::shared_ptr<Tensor> sigmoid(
        const std::shared_ptr<Tensor>& input
    );
};