#include <algorithm>
#include <filesystem>
#include <iostream>
#include <vector>
#include <string>
#include <windows.h>

#include "dataset/Dataset.h"
#include "tokenizer/BPETokenizer.h"
#include "neural/model/LanguageModel.h"
#include "neural/model/ModelSaver.h"
#include "training/Trainer.h"
#include "generation/TextGenerator.h"
#include "tokenizer/TokenizerSaver.h"
#include <limits>
#include <cmath>

int main() {
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);

    std::filesystem::path datasetPath =
    std::filesystem::path(PROJECT_DIR) / "data" / "training.txt";

    std::cout << "Dataset Path: " << datasetPath.string() << "\n";
    std::cout << "File Exists: "
              << std::filesystem::exists(datasetPath) << "\n";

    std::string trainingData = Dataset::readText(datasetPath);

    BPETokenizer tokenizer;
    tokenizer.train(trainingData, 100);

    Dataset dataset;
    dataset.load(datasetPath, tokenizer, 0.2);

    std::cout << "Vocabulary Size: "
              << tokenizer.vocabSize() << "\n";

    std::cout << "Train Dataset Size: "
              << dataset.trainSize() << "\n";

    std::cout << "Validation Dataset Size: "
              << dataset.validationSize() << "\n";

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

    Trainer trainer(model, 0.001);

    std::vector<std::vector<double>> bestParameters;

    double bestValidationLoss = std::numeric_limits<double>::infinity();

    int bestEpoch = 0;
    int patience = 30;
    int patienceCounter = 0;

    std::cout << "\nTraining Start\n\n";

    for (int epoch = 0; epoch < 500; epoch++) {
        double trainLoss = trainer.trainEpoch(
            dataset.getTrainSequences()
        );

        double validationLoss = trainer.evaluate(
            dataset.getValidationSequences()
        );

        if (!std::isfinite(validationLoss)) {
            throw std::runtime_error("Validation Loss became invalid");
        }

        if (validationLoss < bestValidationLoss - 0.001) {
            bestValidationLoss = validationLoss;
            bestEpoch = epoch;
            patienceCounter = 0;

            bestParameters.clear();

            for (const auto& parameter : parameters) {
                bestParameters.push_back(parameter->getData());
            }
        } else {
            patienceCounter++;
        }

        if (epoch % 10 == 0) {
            std::cout << "Epoch: " << epoch
                      << " | Train Loss: " << trainLoss
                      << " | Validation Loss: " << validationLoss
                      << " | Best Loss: " << bestValidationLoss
                      << "\n";
        }

        if (patienceCounter >= patience) {
            std::cout << "\nEarly Stopping!\n";
            break;
        }
    }

    for (size_t i = 0; i < parameters.size(); i++) {
        parameters[i]->setData(bestParameters[i]);
    }

    std::cout << "\nTraining Complete\n";
    std::cout << "Best Epoch: " << bestEpoch << "\n";
    std::cout << "Best Validation Loss: "
              << bestValidationLoss << "\n";

    ModelSaver::save(model, "model.bin");

    std::cout << "Best Model Saved!\n";

    LanguageModel loadedModel(
        tokenizer.vocabSize(),
        8,
        4,
        2
    );

    ModelSaver::load(loadedModel, "model.bin");

    std::cout << "Model Loaded!\n";

    std::string prompt = "안녕하세요";

    std::string generatedText = TextGenerator::generate(
        loadedModel,
        tokenizer,
        prompt,
        20,
        0.8
    );

    std::cout << "\nText Generation Start\n\n";
    std::cout << "Prompt: " << prompt << "\n";
    std::cout << "Generated Text:\n";
    std::cout << generatedText << "\n";

    auto input = Tensor::create(
    {1, 2, 3, 4, 5, 6, 7, 8},
    {2, 4},
    true
);

    auto output = input->dropout(0.5, true);

    std::cout << "Input:\n";

    for (double value : input->getData()) {
        std::cout << value << " ";
    }

    std::cout << "\n\nDropout Output:\n";

    for (double value : output->getData()) {
        std::cout << value << " ";
    }

    std::cout << "\n";

    auto loss = output->mean();

    loss->backward();

    std::cout << "\nInput Gradient:\n";

    for (double value : input->getGrad()) {
        std::cout << value << " ";
    }

    std::cout << "\n";
    TokenizerSaver::save(tokenizer, "tokenizer.bin");

    std::cout << "Tokenizer Saved!\n";

    BPETokenizer loadedTokenizer;

    TokenizerSaver::load(loadedTokenizer, "tokenizer.bin");

    std::cout << "Tokenizer Loaded!\n";

    std::cout << "Loaded Vocabulary Size: "
              << loadedTokenizer.vocabSize() << "\n";



    return 0;
}
