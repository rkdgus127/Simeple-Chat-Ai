#include "Tensor.h"

#include <stdexcept>
#include <numeric>
#include <unordered_set>
#include <utility>
#include <algorithm>
#include <cmath>

Tensor::Tensor(
    std::vector<double> data,
    std::vector<size_t> shape,
    bool requiresGrad
) : data(std::move(data)),
    shape(std::move(shape)),
    requiresGrad(requiresGrad) {

    size_t expected = 1;

    for (size_t dimension : this->shape) {
        if (dimension == 0) {
            throw std::invalid_argument("Tensor dimensions cannot be zero");
        }

        if (expected > SIZE_MAX / dimension) {
            throw std::overflow_error("Tensor size overflow");
        }

        expected *= dimension;
    }

    if (expected != this->data.size()) {
        throw std::invalid_argument("Tensor shape mismatch");
    }

    grad.resize(this->data.size(), 0.0);
}

std::shared_ptr<Tensor> Tensor::create(
    std::vector<double> data,
    std::vector<size_t> shape,
    bool requiresGrad
) {
    return std::shared_ptr<Tensor>(
        new Tensor(std::move(data), std::move(shape), requiresGrad)
    );
}

const std::vector<double>& Tensor::getData() const {
    return data;
}

const std::vector<double>& Tensor::getGrad() const {
    return grad;
}

const std::vector<size_t>& Tensor::getShape() const {
    return shape;
}

bool Tensor::isRequiresGrad() const {
    return requiresGrad;
}

size_t Tensor::size() const {
    return data.size();
}

void Tensor::setData(const std::vector<double>& newData) {
    if (newData.size() != data.size()) {
        throw std::invalid_argument("Tensor data size mismatch");
    }

    data = newData;
}

std::shared_ptr<Tensor> Tensor::add(
    const std::shared_ptr<Tensor>& other
) {
    if (shape != other->shape) {
        throw std::invalid_argument("Tensor shape mismatch");
    }

    std::vector<double> result(data.size());

    for (size_t i = 0; i < data.size(); i++) {
        result[i] = data[i] + other->data[i];
    }

    auto output = create(
        result,
        shape,
        requiresGrad || other->requiresGrad
    );

    if (output->requiresGrad) {
        output->parents = {shared_from_this(), other};

        std::weak_ptr<Tensor> weakOutput = output;

        output->backwardFn = [weakOutput, self = shared_from_this(), other]() {
            auto output = weakOutput.lock();

            if (!output) return;

            for (size_t i = 0; i < output->size(); i++) {
                if (self->requiresGrad) {
                    self->grad[i] += output->grad[i];
                }

                if (other->requiresGrad) {
                    other->grad[i] += output->grad[i];
                }
            }
        };
    }

    return output;
}

std::shared_ptr<Tensor> Tensor::subtract(
    const std::shared_ptr<Tensor>& other
) {
    if (shape != other->shape) {
        throw std::invalid_argument("Tensor shape mismatch");
    }

    std::vector<double> result(data.size());

    for (size_t i = 0; i < data.size(); i++) {
        result[i] = data[i] - other->data[i];
    }

    auto output = create(
        result,
        shape,
        requiresGrad || other->requiresGrad
    );

    if (output->requiresGrad) {
        output->parents = {shared_from_this(), other};

        std::weak_ptr<Tensor> weakOutput = output;

        output->backwardFn = [weakOutput, self = shared_from_this(), other]() {
            auto output = weakOutput.lock();

            if (!output) return;

            for (size_t i = 0; i < output->size(); i++) {
                if (self->requiresGrad) {
                    self->grad[i] += output->grad[i];
                }

                if (other->requiresGrad) {
                    other->grad[i] -= output->grad[i];
                }
            }
        };
    }

    return output;
}

std::shared_ptr<Tensor> Tensor::multiply(
    const std::shared_ptr<Tensor>& other
) {
    if (shape != other->shape) {
        throw std::invalid_argument("Tensor shape mismatch");
    }

    std::vector<double> result(data.size());

    for (size_t i = 0; i < data.size(); i++) {
        result[i] = data[i] * other->data[i];
    }

    auto output = create(
        result,
        shape,
        requiresGrad || other->requiresGrad
    );

    if (output->requiresGrad) {
        output->parents = {shared_from_this(), other};

        std::weak_ptr<Tensor> weakOutput = output;

        output->backwardFn = [weakOutput, self = shared_from_this(), other]() {
            auto output = weakOutput.lock();

            if (!output) return;

            for (size_t i = 0; i < output->size(); i++) {
                if (self->requiresGrad) {
                    self->grad[i] += other->data[i] * output->grad[i];
                }

                if (other->requiresGrad) {
                    other->grad[i] += self->data[i] * output->grad[i];
                }
            }
        };
    }

    return output;
}

std::shared_ptr<Tensor> Tensor::matmul(
    const std::shared_ptr<Tensor>& other
) {
    if (shape.size() != 2 || other->shape.size() != 2) {
        throw std::invalid_argument("Matmul requires 2D tensors");
    }

    size_t rows = shape[0];
    size_t cols = shape[1];

    size_t otherRows = other->shape[0];
    size_t otherCols = other->shape[1];

    if (cols != otherRows) {
        throw std::invalid_argument("Matmul shape mismatch");
    }

    std::vector<double> result(rows * otherCols, 0.0);

    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < otherCols; j++) {
            for (size_t k = 0; k < cols; k++) {
                result[i * otherCols + j] +=
                    data[i * cols + k] *
                    other->data[k * otherCols + j];
            }
        }
    }

    auto output = create(
        result,
        {rows, otherCols},
        requiresGrad || other->requiresGrad
    );

    if (output->requiresGrad) {
        output->parents = {shared_from_this(), other};

        std::weak_ptr<Tensor> weakOutput = output;

        output->backwardFn = [
            weakOutput,
            self = shared_from_this(),
            other,
            rows,
            cols,
            otherCols
        ]() {
            auto output = weakOutput.lock();

            if (!output) return;

            if (self->requiresGrad) {
                for (size_t i = 0; i < rows; i++) {
                    for (size_t k = 0; k < cols; k++) {
                        for (size_t j = 0; j < otherCols; j++) {
                            self->grad[i * cols + k] +=
                                output->grad[i * otherCols + j] *
                                other->data[k * otherCols + j];
                        }
                    }
                }
            }

            if (other->requiresGrad) {
                for (size_t k = 0; k < cols; k++) {
                    for (size_t j = 0; j < otherCols; j++) {
                        for (size_t i = 0; i < rows; i++) {
                            other->grad[k * otherCols + j] +=
                                self->data[i * cols + k] *
                                output->grad[i * otherCols + j];
                        }
                    }
                }
            }
        };
    }

    return output;
}

std::shared_ptr<Tensor> Tensor::transpose() {
    if (shape.size() != 2) {
        throw std::invalid_argument("Transpose requires 2D tensor");
    }

    size_t rows = shape[0];
    size_t cols = shape[1];

    std::vector<double> result(data.size());

    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            result[j * rows + i] = data[i * cols + j];
        }
    }

    auto output = create(
        result,
        {cols, rows},
        requiresGrad
    );

    if (requiresGrad) {
        output->parents = {shared_from_this()};

        std::weak_ptr<Tensor> weakOutput = output;

        output->backwardFn = [
            weakOutput,
            self = shared_from_this(),
            rows,
            cols
        ]() {
            auto output = weakOutput.lock();

            if (!output) return;

            for (size_t i = 0; i < rows; i++) {
                for (size_t j = 0; j < cols; j++) {
                    self->grad[i * cols + j] +=
                        output->grad[j * rows + i];
                }
            }
        };
    }

    return output;
}

std::shared_ptr<Tensor> Tensor::sum() {
    double result = std::accumulate(
        data.begin(),
        data.end(),
        0.0
    );

    auto output = create(
        {result},
        {1},
        requiresGrad
    );

    if (requiresGrad) {
        output->parents = {shared_from_this()};

        std::weak_ptr<Tensor> weakOutput = output;

        output->backwardFn = [
            weakOutput,
            self = shared_from_this()
        ]() {
            auto output = weakOutput.lock();

            if (!output) return;

            for (size_t i = 0; i < self->size(); i++) {
                self->grad[i] += output->grad[0];
            }
        };
    }

    return output;
}

void Tensor::backward() {
    if (!requiresGrad) {
        throw std::runtime_error("Tensor does not require gradients");
    }

    std::vector<std::shared_ptr<Tensor>> topology;
    std::unordered_set<Tensor*> visited;



    std::function<void(const std::shared_ptr<Tensor>&)> build =
        [&](const std::shared_ptr<Tensor>& node) {
            if (!visited.insert(node.get()).second) {
                return;
            }

            for (const auto& parent : node->parents) {
                build(parent);
            }

            topology.push_back(node);
        };

    build(shared_from_this());

    for (const auto& node : topology) {
        std::fill(node->grad.begin(), node->grad.end(), 0.0);
    }

    grad[0] = 1.0;

    for (auto it = topology.rbegin(); it != topology.rend(); ++it) {
        if ((*it)->backwardFn) {
            (*it)->backwardFn();
        }
    }

}

std::shared_ptr<Tensor> Tensor::addBias(
    const std::shared_ptr<Tensor>& bias
) {
    if (shape.size() != 2 ||
        bias->shape.size() != 2 ||
        bias->shape[0] != 1 ||
        bias->shape[1] != shape[1]) {
        throw std::invalid_argument("Bias shape mismatch");
        }

    size_t rows = shape[0];
    size_t cols = shape[1];

    std::vector<double> result(data.size());

    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            result[i * cols + j] =
                data[i * cols + j] + bias->data[j];
        }
    }

    auto output = create(
        result,
        shape,
        requiresGrad || bias->requiresGrad
    );

    if (output->requiresGrad) {
        output->parents = {shared_from_this(), bias};

        std::weak_ptr<Tensor> weakOutput = output;

        output->backwardFn = [
            weakOutput,
            self = shared_from_this(),
            bias,
            rows,
            cols
        ]() {
            auto output = weakOutput.lock();

            if (!output) return;

            for (size_t i = 0; i < rows; i++) {
                for (size_t j = 0; j < cols; j++) {
                    size_t index = i * cols + j;

                    if (self->requiresGrad) {
                        self->grad[index] += output->grad[index];
                    }

                    if (bias->requiresGrad) {
                        bias->grad[j] += output->grad[index];
                    }
                }
            }
        };
    }

    return output;
}

std::shared_ptr<Tensor> Tensor::sigmoid() {
    std::vector<double> result(data.size());

    for (size_t i = 0; i < data.size(); i++) {
        if (data[i] >= 0) {
            result[i] = 1.0 / (1.0 + std::exp(-data[i]));
        } else {
            double expValue = std::exp(data[i]);
            result[i] = expValue / (1.0 + expValue);
        }
    }

    auto output = create(result, shape, requiresGrad);

    if (requiresGrad) {
        output->parents = {shared_from_this()};

        std::weak_ptr<Tensor> weakOutput = output;

        output->backwardFn = [
            weakOutput,
            self = shared_from_this()
        ]() {
            auto output = weakOutput.lock();

            if (!output) return;

            for (size_t i = 0; i < self->data.size(); i++) {
                if (self->requiresGrad) {
                    double value = output->data[i];

                    self->grad[i] +=
                        output->grad[i] * value * (1.0 - value);
                }
            }
        };
    }

    return output;
}


std::shared_ptr<Tensor> Tensor::relu() {
    std::vector<double> result(data.size());

    for (size_t i = 0; i < data.size(); i++) {
        result[i] = std::max(0.0, data[i]);
    }

    auto output = create(result, shape, requiresGrad);

    if (requiresGrad) {
        output->parents = {shared_from_this()};

        std::weak_ptr<Tensor> weakOutput = output;

        output->backwardFn = [
            weakOutput,
            self = shared_from_this()
        ]() {
            auto output = weakOutput.lock();

            if (!output) return;

            for (size_t i = 0; i < self->data.size(); i++) {
                if (self->requiresGrad) {
                    self->grad[i] +=
                        self->data[i] > 0
                            ? output->grad[i]
                            : 0.0;
                }
            }
        };
    }

    return output;
}

std::shared_ptr<Tensor> Tensor::mean() {
    if (data.empty()) {
        throw std::invalid_argument("Cannot calculate mean of empty tensor");
    }

    double total = 0.0;

    for (double value : data) {
        total += value;
    }

    double result = total / static_cast<double>(data.size());

    auto output = create({result}, {1}, requiresGrad);

    if (requiresGrad) {
        output->parents = {shared_from_this()};

        std::weak_ptr<Tensor> weakOutput = output;

        output->backwardFn = [
            weakOutput,
            self = shared_from_this()
        ]() {
            auto output = weakOutput.lock();

            if (!output) return;

            double gradient =
                output->grad[0] / static_cast<double>(self->data.size());

            for (size_t i = 0; i < self->data.size(); i++) {
                self->grad[i] += gradient;
            }
        };
    }

    return output;
}

std::shared_ptr<Tensor> Tensor::softmax() const {
    if (shape.size() != 2) {
        throw std::invalid_argument("Softmax requires a 2D tensor");
    }

    size_t rows = shape[0];
    size_t cols = shape[1];

    std::vector<double> resultData(data.size());

    for (size_t i = 0; i < rows; i++) {
        double maxValue = data[i * cols];

        for (size_t j = 1; j < cols; j++) {
            maxValue = std::max(maxValue, data[i * cols + j]);
        }

        double sum = 0.0;

        for (size_t j = 0; j < cols; j++) {
            double value = std::exp(data[i * cols + j] - maxValue);

            resultData[i * cols + j] = value;
            sum += value;
        }

        for (size_t j = 0; j < cols; j++) {
            resultData[i * cols + j] /= sum;
        }
    }

    auto result = Tensor::create(resultData, shape, requiresGrad);

    auto self = std::const_pointer_cast<Tensor>(shared_from_this());

    result->parents.push_back(self);

    result->backwardFn = [self, result, rows, cols]() {
        for (size_t i = 0; i < rows; i++) {
            double dot = 0.0;

            for (size_t j = 0; j < cols; j++) {
                size_t index = i * cols + j;

                dot += result->grad[index] * result->data[index];
            }

            for (size_t j = 0; j < cols; j++) {
                size_t index = i * cols + j;

                self->grad[index] += result->data[index] *
                    (result->grad[index] - dot);
            }
        }
    };

    return result;
}

std::shared_ptr<Tensor> Tensor::concatColumns(
    const std::vector<std::shared_ptr<Tensor>>& tensors
) {
    if (tensors.empty()) {
        throw std::invalid_argument("Cannot concatenate empty tensors");
    }

    size_t rows = tensors[0]->getShape()[0];
    size_t totalCols = 0;

    std::vector<size_t> widths;

    for (const auto& tensor : tensors) {
        if (tensor->getShape().size() != 2 ||
            tensor->getShape()[0] != rows) {
            throw std::invalid_argument("Tensor shape mismatch");
        }

        size_t cols = tensor->getShape()[1];
        widths.push_back(cols);
        totalCols += cols;
    }

    std::vector<double> resultData(rows * totalCols);
    bool requiresGrad = false;

    for (const auto& tensor : tensors) {
        if (tensor->requiresGrad) {
            requiresGrad = true;
        }
    }

    for (size_t row = 0; row < rows; row++) {
        size_t offset = 0;

        for (size_t i = 0; i < tensors.size(); i++) {
            const auto& data = tensors[i]->getData();
            size_t cols = widths[i];

            for (size_t col = 0; col < cols; col++) {
                resultData[row * totalCols + offset + col] =
                    data[row * cols + col];
            }

            offset += cols;
        }
    }

    auto result = Tensor::create(
        resultData,
        {rows, totalCols},
        requiresGrad
    );

    for (const auto& tensor : tensors) {
        result->parents.push_back(tensor);
    }

    result->backwardFn = [tensors, result, rows, totalCols, widths]() {
        for (size_t i = 0; i < tensors.size(); i++) {
            if (!tensors[i]->requiresGrad) continue;

            size_t offset = 0;

            for (size_t j = 0; j < i; j++) {
                offset += widths[j];
            }

            size_t cols = widths[i];

            for (size_t row = 0; row < rows; row++) {
                for (size_t col = 0; col < cols; col++) {
                    tensors[i]->grad[row * cols + col] +=
                        result->grad[row * totalCols + offset + col];
                }
            }
        }
    };

    return result;
}

std::shared_ptr<Tensor> Tensor::layerNorm(
    const std::shared_ptr<Tensor>& gamma,
    const std::shared_ptr<Tensor>& beta,
    double epsilon
) const {
    if (shape.size() != 2) {
        throw std::invalid_argument("LayerNorm requires 2D tensor");
    }

    size_t rows = shape[0];
    size_t cols = shape[1];

    if (gamma->getShape() != std::vector<size_t>{1, cols} ||
        beta->getShape() != std::vector<size_t>{1, cols}) {
        throw std::invalid_argument("LayerNorm parameter shape mismatch");
    }

    std::vector<double> normalized(data.size());
    std::vector<double> inverseStd(rows);
    std::vector<double> resultData(data.size());

    for (size_t row = 0; row < rows; row++) {
        double mean = 0.0;

        for (size_t col = 0; col < cols; col++) {
            mean += data[row * cols + col];
        }

        mean /= static_cast<double>(cols);

        double variance = 0.0;

        for (size_t col = 0; col < cols; col++) {
            double diff = data[row * cols + col] - mean;
            variance += diff * diff;
        }

        variance /= static_cast<double>(cols);

        inverseStd[row] = 1.0 / std::sqrt(variance + epsilon);

        for (size_t col = 0; col < cols; col++) {
            size_t index = row * cols + col;

            normalized[index] =
                (data[index] - mean) * inverseStd[row];

            resultData[index] =
                normalized[index] * gamma->getData()[col]
                + beta->getData()[col];
        }
    }

    auto self = std::const_pointer_cast<Tensor>(shared_from_this());

    auto result = Tensor::create(
        resultData,
        shape,
        requiresGrad || gamma->requiresGrad || beta->requiresGrad
    );

    result->parents.push_back(self);
    result->parents.push_back(gamma);
    result->parents.push_back(beta);

    result->backwardFn = [
        self, gamma, beta, result,
        normalized, inverseStd, rows, cols
    ]() {
        for (size_t col = 0; col < cols; col++) {
            double gammaGrad = 0.0;
            double betaGrad = 0.0;

            for (size_t row = 0; row < rows; row++) {
                size_t index = row * cols + col;

                double grad = result->grad[index];

                gammaGrad += grad * normalized[index];
                betaGrad += grad;
            }

            if (gamma->requiresGrad) {
                gamma->grad[col] += gammaGrad;
            }

            if (beta->requiresGrad) {
                beta->grad[col] += betaGrad;
            }
        }

        for (size_t row = 0; row < rows; row++) {
            double sumGrad = 0.0;
            double sumGradNormalized = 0.0;

            for (size_t col = 0; col < cols; col++) {
                size_t index = row * cols + col;

                double grad = result->grad[index] * gamma->data[col];

                sumGrad += grad;
                sumGradNormalized += grad * normalized[index];
            }

            for (size_t col = 0; col < cols; col++) {
                size_t index = row * cols + col;

                if (self->requiresGrad) {
                    double grad = result->grad[index] * gamma->data[col];

                    self->grad[index] +=
                        inverseStd[row] / static_cast<double>(cols) *
                        (
                            static_cast<double>(cols) * grad
                            - sumGrad
                            - normalized[index] * sumGradNormalized
                        );
                }
            }
        }
    };

    return result;
}