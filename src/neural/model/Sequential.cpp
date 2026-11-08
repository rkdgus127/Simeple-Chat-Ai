#include "Sequential.h"
#include "../layers/Activation.h"

void Sequential::addLinear(size_t inputSize, size_t outputSize) {
    auto linear = std::make_shared<Linear>(inputSize, outputSize);

    layers.push_back({
        LayerType::LINEAR,
        linear
    });
}

void Sequential::addReLU() {
    layers.push_back({
        LayerType::RELU,
        nullptr
    });
}

void Sequential::addSigmoid() {
    layers.push_back({
        LayerType::SIGMOID,
        nullptr
    });
}

std::shared_ptr<Tensor> Sequential::forward(
    const std::shared_ptr<Tensor>& input
) {
    auto output = input;

    for (const auto& layer : layers) {
        switch (layer.type) {
        case LayerType::LINEAR:
            output = layer.linear->forward(output);
            break;

        case LayerType::RELU:
            output = Activation::relu(output);
            break;

        case LayerType::SIGMOID:
            output = Activation::sigmoid(output);
            break;
        }
    }

    return output;
}

std::vector<std::shared_ptr<Tensor>> Sequential::getParameters() const {
    std::vector<std::shared_ptr<Tensor>> parameters;

    for (const auto& layer : layers) {
        if (layer.type == LayerType::LINEAR) {
            parameters.push_back(layer.linear->getWeights());
            parameters.push_back(layer.linear->getBias());
        }
    }

    return parameters;
}