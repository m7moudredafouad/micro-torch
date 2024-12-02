#include <gtest/gtest.h>

#include <tensor.hpp>

using namespace micro;

TEST(AutoGrad, BasicGradient) {
    uint32_t tensor_size = 3;

    Tensor t1({tensor_size}), t2({tensor_size}), t3({tensor_size});
    t1.requires_grad(true);

    t1 = 2;
    t2 = t1 * t1;
    t3 = t2 + t1;
    t3.backward();

    auto t1_grad = t1.grad();
    auto t2_grad = t2.grad();
    auto t3_grad = t3.grad();

    std::vector<uint32_t> index(1, 0);
    for (uint32_t i = 0; i < tensor_size; i++) {
        index[0] = i;
        EXPECT_EQ((float)t1_grad[index], 5.f);
        EXPECT_EQ((float)t2_grad[index], 1.f);
        EXPECT_EQ((float)t3_grad[index], 1.f);
    }
}

TEST(AutoGrad, BasicGradient2) {
    uint32_t tensor_size = 3;

    Tensor t1({tensor_size}), t2({tensor_size}), t3({tensor_size});
    t1.requires_grad(true);

    t1 = 2;
    t2 = t1 * t1 * t1;
    t3 = t2 + t1;
    t3.backward();

    auto t1_grad = t1.grad();
    auto t2_grad = t2.grad();
    auto t3_grad = t3.grad();

    std::vector<uint32_t> index(1, 0);
    for (uint32_t i = 0; i < tensor_size; i++) {
        index[0] = i;
        EXPECT_EQ((float)t1_grad[index], 13.f);
        EXPECT_EQ((float)t2_grad[index], 1.f);
        EXPECT_EQ((float)t3_grad[index], 1.f);
    }
}