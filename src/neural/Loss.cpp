#include "Loss.h"

#include <stdexcept>

std::shared_ptr<Tensor> Loss::mse(
    const std::shared_ptr<Tensor>& prediction,
    const std::shared_ptr<Tensor>& target
) {
    if (prediction->getShape() != target->getShape()) {
        throw std::invalid_argument("MSE shape mismatch");
    }

    auto difference = prediction->subtract(target);
    auto squared = difference->multiply(difference);

    return squared->mean();
}