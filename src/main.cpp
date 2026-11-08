#include <iostream>
#include <memory>
#include "core/Autograd.h"

int main() {
    auto x = std::make_shared<Autograd>(2.0, true);
    auto three = std::make_shared<Autograd>(3.0);

    auto square = Autograd::multiply(x, x);
    auto triple = Autograd::multiply(three, x);

    auto result = Autograd::add(square, triple);

    result->backward();

    std::cout << "Result: " << result->value << std::endl;
    std::cout << "Gradient: " << x->grad << std::endl;

    return 0;
}