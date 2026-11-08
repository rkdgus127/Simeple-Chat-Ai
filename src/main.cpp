#include <algorithm>
#include <iostream>
#include <vector>
#include <string>
#include <windows.h>

#include "tokenizer/BPETokenizer.h"
#include "neural/model/LanguageModel.h"
#include "neural/model/ModelSaver.h"
#include "training/Trainer.h"
#include "generation/TextGenerator.h"

int main() {
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);

    BPETokenizer tokenizer;

    std::string trainingData =
        "hello hello hello help help "
        "hello hello help "
        "안녕하세요 안녕하세요 "
        "AI를 만들고 있습니다 AI를 만들고 있습니다 ";

    tokenizer.train(trainingData, 30);

    std::string text = "hello 안녕하세요 AI를 만들고 있습니다";

    std::vector<int> tokens = tokenizer.encode(text);

    std::cout << "Vocabulary Size: "
              << tokenizer.vocabSize() << "\n";

    std::cout << "Token Count: "
              << tokens.size() << "\n\n";

    LanguageModel model(
        tokenizer.vocabSize(),
        8,
        4,
        2
    );

    auto parameters = model.getParameters();

    size_t totalParameters = 0;

    for (const auto& parameter : parameters) {
        totalParameters += parameter->size();
    }

    std::cout << "Total Parameters: "
              << totalParameters << "\n";

    std::cout << "Parameter Tensor Count: "
              << parameters.size() << "\n";

    Trainer trainer(model, 0.001);

    std::vector<int> trainingTokens = tokens;

    std::cout << "\nTraining Start\n\n";

    for (int epoch = 0; epoch <= 500; epoch++) {
        double loss = trainer.trainStep(trainingTokens);

        if (epoch % 50 == 0) {
            std::cout << "Epoch: "
                      << epoch
                      << " | Loss: "
                      << loss
                      << "\n";
        }
    }

    std::cout << "\nTraining Complete\n";

    ModelSaver::save(model, "model.bin");

    std::cout << "Model Saved!\n";

    LanguageModel loadedModel(
        tokenizer.vocabSize(),
        8,
        4,
        2
    );

    ModelSaver::load(loadedModel, "model.bin");

    std::cout << "Model Loaded!\n";

    auto loadedOutput = loadedModel.forward(tokens);

    std::cout << "Loaded Model Output:\n";

    for (size_t i = 0; i < 10; i++) {
        std::cout << loadedOutput->getData()[i] << " ";
    }

    std::cout << "\n\n";

    std::vector<int> fullTokens = tokenizer.encode(
        "hello 안녕하세요 AI를 만들고 있습니다"
    );

    std::vector<int> promptTokens = tokenizer.encode(
        "hello 안녕하세요 "
    );

    std::cout << "Full Tokens:\n";

    for (int token : fullTokens) {
        std::cout << token << " ";
    }

    std::cout << "\nPrompt Tokens:\n";

    for (int token : promptTokens) {
        std::cout << token << " ";
    }

    std::cout << "\n\n";

    std::vector<int> testInput(
        fullTokens.begin(),
        fullTokens.end() - 1
    );

    auto testOutput = loadedModel.forward(testInput, true);

    size_t vocabSize = testOutput->getShape()[1];

    int correct = 0;
    int total = static_cast<int>(fullTokens.size()) - 1;

    for (int i = 0; i < total; i++) {
        size_t offset = static_cast<size_t>(i) * vocabSize;

        auto begin = testOutput->getData().begin() + offset;
        auto end = begin + vocabSize;

        int predicted = static_cast<int>(
            std::distance(
                begin,
                std::max_element(begin, end)
            )
        );

        int expected = fullTokens[i + 1];

        if (predicted == expected) {
            correct++;
        }

        std::cout << "Position: " << i
                  << " | Predicted: " << predicted
                  << " | Expected: " << expected
                  << " | "
                  << (predicted == expected ? "CORRECT" : "WRONG")
                  << "\n";
    }

    std::cout << "\nCorrect: "
              << correct << " / " << total << "\n";

    std::cout << "Accuracy: "
              << static_cast<double>(correct) / total * 100.0
              << "%\n";

    std::cout << "\nText Generation Start\n\n";

    std::string generatedText = TextGenerator::generate(
        loadedModel,
        tokenizer,
        "hello 안녕하세요 ",
        10,
        0.0
    );

    std::cout << "Generated Text:\n";
    std::cout << generatedText << "\n";

    return 0;
}