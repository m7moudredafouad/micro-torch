#include <gtest/gtest.h>

#include <tensor.hpp>

TEST(BasicTensorOperations, Assignment) {
    uint32_t tensor_size = 3;
    Tensor t1({tensor_size});
    t1 = 100.f;

    std::vector<uint32_t> index(1, 0);

    for (uint32_t i = 0; i < tensor_size; i++) {
        index[0] = i;
        EXPECT_EQ((float)t1[index], 100.f);
    }
}

TEST(BasicTensorOperations, Add) {
    uint32_t tensor_size = 3;
    Tensor t1({tensor_size}, Tensor::Type::UINT32);
    t1 = 0;
    t1 += 16;

    std::vector<uint32_t> index(1, 0);

    for (uint32_t i = 0; i < tensor_size; i++) {
        index[0] = i;
        EXPECT_EQ((uint32_t)t1[index], 16);
    }
}
