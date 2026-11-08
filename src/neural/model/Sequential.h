#pragma once

#include "../layers/Linear.h"
#include "../../core/Tensor.h"

#include <memory>
#include <vector>

class Sequential {
private:
    enum class LayerType {
        LINEAR,
        RELU,
        SIGMOID
    };

    struct Layer {
        LayerType type;
        std::shared_ptr<Linear> linear;
    };

    std::vector<Layer> layers;

public:
    void addLinear(size_t inputSize, size_t outputSize);
    void addReLU();
    void addSigmoid();

    std::shared_ptr<Tensor> forward(
        const std::shared_ptr<Tensor>& input
    );

    std::vector<std::shared_ptr<Tensor>> getParameters() const;
};