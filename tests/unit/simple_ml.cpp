#include <gtest/gtest.h>
#include <stdlib.h>

#include <tensor.hpp>

using namespace micro;

TEST(SimpleML, SimpleMLNotGate) {
    /**
     * [0 1]
     * [1 0]
     */

    float lr = 0.01;

    Tensor data({2}), weights({1}), bias({1}), out({2});
    data = {0.f, 1.f};
    out = {1.f, 0.f};

    weights = {rand() / (float)RAND_MAX};
    bias = {rand() / (float)RAND_MAX};

    weights.requires_grad(true);
    bias.requires_grad(true);

    for (int i = 0; i < 100; i++) {
        auto pred = data * weights + bias;
        auto loss = pred - out;
        loss = loss * loss;
        loss.backward();

        auto weights_grad = weights.grad();
        auto bias_grad = bias.grad();

        weights = weights - weights_grad * lr;
        bias = bias - bias_grad * lr;
    }

    auto pred = data * weights + bias;

    // LOG(INFO) << pred;

    EXPECT_GE((float)pred[{0}], 0.5f);
    EXPECT_LE((float)pred[{1}], 0.5f);
}

TEST(SimpleML, SimpleMLAndGate) {
    /**
     * [0 0 0]
     * [0 1 0]
     * [1 0 0]
     * [1 1 1]
     */

    float lr = 0.1;

    Tensor data({4, 2}), weights({2, 1}), bias({1}), out({4, 1});
    data = {0.f, 0.f, 0.f, 1.f, 1.f, 0.f, 1.f, 1.f};
    out = {0.f, 0.f, 0.f, 1.f};

    weights = {rand() / (float)RAND_MAX, rand() / (float)RAND_MAX};
    bias = {rand() / (float)RAND_MAX};

    weights.requires_grad(true);
    bias.requires_grad(true);

    for (int i = 0; i < 50; i++) {
        auto pred = data.mm(weights) + bias;
        auto loss = pred - out;
        loss = loss * loss;
        loss = loss.sum(0);

        // LOG(INFO) << "Loss: " << (float)loss[{0}];

        weights.reset_grad();
        bias.reset_grad();
        loss.backward();

        auto weights_grad = weights.grad();
        auto bias_grad = bias.grad();

        weights = weights - weights_grad * lr;
        bias = bias - bias_grad * lr;
    }

    auto pred = data.mm(weights) + bias;
    // LOG(INFO) << pred;

    EXPECT_LE((float)(pred[{0, 0}]), 0.5f);
    EXPECT_LE((float)(pred[{1, 0}]), 0.5f);
    EXPECT_LE((float)(pred[{2, 0}]), 0.5f);
    EXPECT_GE((float)(pred[{3, 0}]), 0.5f);
}

TEST(SimpleML, SimpleMLORGate) {
    /**
     * [0 0 0]
     * [0 1 1]
     * [1 0 1]
     * [1 1 1]
     */

    Tensor data({4, 2}), weights({2, 1}), bias({1}), out({4, 1});
    data = {0.f, 0.f, 0.f, 1.f, 1.f, 0.f, 1.f, 1.f};
    out = {0.f, 1.f, 1.f, 1.f};

    weights = {rand() / (float)RAND_MAX, rand() / (float)RAND_MAX};
    bias = {rand() / (float)RAND_MAX};

    weights.requires_grad(true);
    bias.requires_grad(true);

    float lr = 0.1;
    for (int i = 0; i < 50; i++) {
        auto pred = data.mm(weights) + bias;
        auto loss = pred - out;
        loss = loss * loss;
        loss = loss.sum(0);

        // LOG(INFO) << "Loss: " << (float)loss[{0}];

        weights.reset_grad();
        bias.reset_grad();
        loss.backward();

        auto weights_grad = weights.grad();
        auto bias_grad = bias.grad();

        weights = weights - weights_grad * lr;
        bias = bias - bias_grad * lr;
    }

    auto pred = data.mm(weights) + bias;
    // LOG(INFO) << pred;

    EXPECT_LE((float)(pred[{0, 0}]), 0.5f);
    EXPECT_GE((float)(pred[{1, 0}]), 0.5f);
    EXPECT_GE((float)(pred[{2, 0}]), 0.5f);
    EXPECT_GE((float)(pred[{3, 0}]), 0.5f);
}
