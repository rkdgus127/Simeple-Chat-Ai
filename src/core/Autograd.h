#pragma once

#include <vector>
#include <memory>
#include <functional>

class Autograd : public std::enable_shared_from_this<Autograd> {
public:
    double value;
    double grad;

    bool requiresGrad;

    std::vector<std::shared_ptr<Autograd>> parents;
    std::function<void()> backwardFn;

    explicit Autograd(double value, bool requiresGrad = false);

    static std::shared_ptr<Autograd> add(
        const std::shared_ptr<Autograd>& a,
        const std::shared_ptr<Autograd>& b
    );

    static std::shared_ptr<Autograd> multiply(
        const std::shared_ptr<Autograd>& a,
        const std::shared_ptr<Autograd>& b
    );

    static std::shared_ptr<Autograd> subtract(
        const std::shared_ptr<Autograd>& a,
        const std::shared_ptr<Autograd>& b
    );

    void backward();
};