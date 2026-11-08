#pragma once

#include <vector>
#include <cstddef>

class Tensor {
private:
    std::vector<double> data;
    std::vector<size_t> shape;

public:
    Tensor(std::vector<double> data, std::vector<size_t> shape);

    const std::vector<double>& getData() const;
    const std::vector<size_t>& getShape() const;

    size_t size() const;

    Tensor add(const Tensor& other) const;
    Tensor subtract(const Tensor& other) const;
    Tensor multiply(const Tensor& other) const;

    Tensor matmul(const Tensor& other) const;
    Tensor transpose() const;
};