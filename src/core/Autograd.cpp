#include "Autograd.h"

#include <stdexcept>
#include <unordered_set>

Autograd::Autograd(double value, bool requiresGrad)
    : value(value),
      grad(0.0),
      requiresGrad(requiresGrad) {
}

std::shared_ptr<Autograd> Autograd::add(
    const std::shared_ptr<Autograd>& a,
    const std::shared_ptr<Autograd>& b
) {
    auto result = std::make_shared<Autograd>(
        a->value + b->value,
        a->requiresGrad || b->requiresGrad
    );

    result->parents = {a, b};

    result->backwardFn = [result, a, b]() {
        if (a->requiresGrad) {
            a->grad += result->grad;
        }

        if (b->requiresGrad) {
            b->grad += result->grad;
        }
    };

    return result;
}

std::shared_ptr<Autograd> Autograd::multiply(
    const std::shared_ptr<Autograd>& a,
    const std::shared_ptr<Autograd>& b
) {
    auto result = std::make_shared<Autograd>(
        a->value * b->value,
        a->requiresGrad || b->requiresGrad
    );

    result->parents = {a, b};

    result->backwardFn = [result, a, b]() {
        if (a->requiresGrad) {
            a->grad += b->value * result->grad;
        }

        if (b->requiresGrad) {
            b->grad += a->value * result->grad;
        }
    };

    return result;
}

std::shared_ptr<Autograd> Autograd::subtract(
    const std::shared_ptr<Autograd>& a,
    const std::shared_ptr<Autograd>& b
) {
    auto result = std::make_shared<Autograd>(
        a->value - b->value,
        a->requiresGrad || b->requiresGrad
    );

    result->parents = {a, b};

    result->backwardFn = [result, a, b]() {
        if (a->requiresGrad) {
            a->grad += result->grad;
        }

        if (b->requiresGrad) {
            b->grad -= result->grad;
        }
    };

    return result;
}

void Autograd::backward() {
    if (!requiresGrad) {
        throw std::runtime_error("Cannot backward without gradients");
    }

    std::vector<std::shared_ptr<Autograd>> topology;
    std::unordered_set<Autograd*> visited;

    std::function<void(const std::shared_ptr<Autograd>&)> build =
        [&](const std::shared_ptr<Autograd>& node) {
            if (!visited.insert(node.get()).second) {
                return;
            }

            for (const auto& parent : node->parents) {
                build(parent);
            }

            topology.push_back(node);
        };

    build(shared_from_this());

    grad = 1.0;

    for (auto it = topology.rbegin(); it != topology.rend(); ++it) {
        if ((*it)->backwardFn) {
            (*it)->backwardFn();
        }
    }
}