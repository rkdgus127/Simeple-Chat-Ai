#include "Tensor.h"

#include <stdexcept>
#include <numeric>

Tensor::Tensor(std::vector<double> data, std::vector<size_t> shape)
    : data(std::move(data)), shape(std::move(shape)) {

    size_t expected = 1;

    for (const auto dimension : this->shape) {
        expected *= dimension;
    }

    if (expected != this->data.size()) {
        throw std::invalid_argument("Tensor shape mismatch");
    }
}

const std::vector<double>& Tensor::getData() const {
    return data;
}

const std::vector<size_t>& Tensor::getShape() const {
    return shape;
}

size_t Tensor::size() const {
    return data.size();
}

Tensor Tensor::add(const Tensor& other) const {
    if (shape != other.shape) {
        throw std::invalid_argument("Tensor shape mismatch");
    }

    std::vector<double> result(data.size());

    for (size_t i = 0; i < data.size(); i++) {
        result[i] = data[i] + other.data[i];
    }

    return Tensor(result, shape);
}

Tensor Tensor::subtract(const Tensor& other) const {
    if (shape != other.shape) {
        throw std::invalid_argument("Tensor shape mismatch");
    }

    std::vector<double> result(data.size());

    for (size_t i = 0; i < data.size(); i++) {
        result[i] = data[i] - other.data[i];
    }

    return Tensor(result, shape);
}

Tensor Tensor::multiply(const Tensor& other) const {
    if (shape != other.shape) {
        throw std::invalid_argument("Tensor shape mismatch");
    }

    std::vector<double> result(data.size());

    for (size_t i = 0; i < data.size(); i++) {
        result[i] = data[i] * other.data[i];
    }

    return Tensor(result, shape);
}

Tensor Tensor::matmul(const Tensor& other) const {
    if (shape.size() != 2 || other.shape.size() != 2) {
        throw std::invalid_argument("Matmul requires 2D tensors");
    }

    const size_t rows = shape[0];
    const size_t cols = shape[1];

    const size_t otherRows = other.shape[0];
    const size_t otherCols = other.shape[1];

    if (cols != otherRows) {
        throw std::invalid_argument("Matmul shape mismatch");
    }

    std::vector<double> result(rows * otherCols, 0.0);

    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < otherCols; j++) {
            for (size_t k = 0; k < cols; k++) {
                result[i * otherCols + j] +=
                    data[i * cols + k] *
                    other.data[k * otherCols + j];
            }
        }
    }

    return Tensor(result, {rows, otherCols});
}

Tensor Tensor::transpose() const {
    if (shape.size() != 2) {
        throw std::invalid_argument("Transpose requires 2D tensor");
    }

    const size_t rows = shape[0];
    const size_t cols = shape[1];

    std::vector<double> result(data.size());

    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            result[j * rows + i] = data[i * cols + j];
        }
    }

    return Tensor(result, {cols, rows});
}