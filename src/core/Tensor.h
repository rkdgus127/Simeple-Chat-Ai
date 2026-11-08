#pragma once

#include <vector>
#include <memory>
#include <functional>
#include <cmath>
#include <cstddef>

class Tensor : public std::enable_shared_from_this<Tensor> {
private:
    std::vector<double> data;
    std::vector<double> grad;
    std::vector<size_t> shape;

    bool requiresGrad;

    std::vector<std::shared_ptr<Tensor>> parents;
    std::function<void()> backwardFn;

    Tensor(
        std::vector<double> data,
        std::vector<size_t> shape,
        bool requiresGrad
    );

public:
    static std::shared_ptr<Tensor> create(
        std::vector<double> data,
        std::vector<size_t> shape,
        bool requiresGrad = false
    );

    void setData(const std::vector<double>& newData);

    std::shared_ptr<Tensor> relu();
    std::shared_ptr<Tensor> sigmoid();
    std::shared_ptr<Tensor> mean();

    const std::vector<double>& getData() const;
    const std::vector<double>& getGrad() const;
    const std::vector<size_t>& getShape() const;

    bool isRequiresGrad() const;

    size_t size() const;

    std::shared_ptr<Tensor> add(
        const std::shared_ptr<Tensor>& other
    );

    std::shared_ptr<Tensor> subtract(
        const std::shared_ptr<Tensor>& other
    );

    std::shared_ptr<Tensor> multiply(
        const std::shared_ptr<Tensor>& other
    );

    std::shared_ptr<Tensor> matmul(
        const std::shared_ptr<Tensor>& other
    );

    std::shared_ptr<Tensor> transpose();

    std::shared_ptr<Tensor> sum();

    std::shared_ptr<Tensor> addBias(
    const std::shared_ptr<Tensor>& bias
);

    void backward();
};