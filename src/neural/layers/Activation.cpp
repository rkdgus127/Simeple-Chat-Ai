#include "Activation.h"

std::shared_ptr<Tensor> Activation::relu(
    const std::shared_ptr<Tensor>& input
) {
    return input->relu();
}

std::shared_ptr<Tensor> Activation::sigmoid(
    const std::shared_ptr<Tensor>& input
) {
    return input->sigmoid();
}