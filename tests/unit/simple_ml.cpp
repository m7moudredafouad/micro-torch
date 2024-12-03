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
    data[{0}] = 0;
    data[{1}] = 1.f;

    out[{0}] = 1.f;
    out[{1}] = 0;

    weights[{0}] = rand() / (float)RAND_MAX;
    bias[{0}] = rand() / (float)RAND_MAX;

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

    LOG(INFO) << pred;

    EXPECT_GE((float)pred[{0}], 0.5f);
    EXPECT_LE((float)pred[{1}], 0.5f);
}
