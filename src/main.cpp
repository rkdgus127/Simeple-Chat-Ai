#include <iostream>

#include "neural/Sequential.h"
#include "neural/Loss.h"
#include "optimizer/Adam.h"

int main() {
    Sequential model;

    model.addLinear(2, 8);
    model.addReLU();
    model.addLinear(8, 1);

    Adam optimizer(
        model.getParameters(),
        0.01
    );

    auto input = Tensor::create(
        {
            1, 0,
            0, 1,
            1, 1,
            2, 1
        },
        {4, 2}
    );

    auto target = Tensor::create(
        {
            1,
            1,
            2,
            3
        },
        {4, 1}
    );

    for (int epoch = 0; epoch < 2000; epoch++) {
        auto prediction = model.forward(input);

        auto loss = Loss::mse(prediction, target);

        loss->backward();

        optimizer.step();

        if (epoch % 200 == 0) {
            std::cout
                << "Epoch: " << epoch
                << " | Loss: " << loss->getData()[0]
                << std::endl;
        }
    }

    auto prediction = model.forward(input);

    std::cout << "\nFinal Prediction:" << std::endl;

    for (double value : prediction->getData()) {
        std::cout << value << std::endl;
    }

    return 0;
}