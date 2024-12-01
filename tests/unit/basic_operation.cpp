#include <gtest/gtest.h>

#include <tensor.hpp>

using namespace micro;

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

TEST(BasicTensorOperations, AddConstant) {
    uint32_t tensor_size = 3;
    Tensor t1({tensor_size}, Type::UINT32);
    t1 = 0;
    t1 += 16;

    std::vector<uint32_t> index(1, 0);

    for (uint32_t i = 0; i < tensor_size; i++) {
        index[0] = i;
        EXPECT_EQ((uint32_t)t1[index], 16);
    }
}

TEST(BasicTensorOperations, SubConstant) {
    uint32_t tensor_size = 3;
    Tensor t1({tensor_size}, Type::UINT32);
    t1 = 16;
    t1 -= 1;

    std::vector<uint32_t> index(1, 0);

    for (uint32_t i = 0; i < tensor_size; i++) {
        index[0] = i;
        EXPECT_EQ((uint32_t)t1[index], 15);
    }
}

TEST(BasicTensorOperations, MulConstant) {
    uint32_t tensor_size = 3;
    Tensor t1({tensor_size}, Type::UINT32);
    t1 = 1;
    t1 *= 16;

    std::vector<uint32_t> index(1, 0);

    for (uint32_t i = 0; i < tensor_size; i++) {
        index[0] = i;
        EXPECT_EQ((uint32_t)t1[index], 16);
    }
}

TEST(BasicTensorOperations, DivConstant) {
    uint32_t tensor_size = 3;
    Tensor t1({tensor_size}, Type::UINT32);
    t1 = 16;
    t1 /= 4;

    std::vector<uint32_t> index(1, 0);

    for (uint32_t i = 0; i < tensor_size; i++) {
        index[0] = i;
        EXPECT_EQ((uint32_t)t1[index], 4);
    }
}

TEST(BasicTensorOperations, AddTensor) {
    uint32_t tensor_size = 3;
    Tensor t1({tensor_size}, Type::UINT32);
    Tensor t2({tensor_size}, Type::UINT32);
    t1 = 1;
    t2 = 2;

    auto t3 = t1 + t2;
    std::vector<uint32_t> index(1, 0);

    for (uint32_t i = 0; i < tensor_size; i++) {
        index[0] = i;
        EXPECT_EQ((uint32_t)t3[index], 3);
    }
}

TEST(BasicTensorOperations, SubTensor) {
    uint32_t tensor_size = 3;
    Tensor t1({tensor_size}, Type::INT32);
    Tensor t2({tensor_size}, Type::INT32);
    t1 = 1;
    t2 = 2;

    auto t3 = t1 - t2;
    std::vector<uint32_t> index(1, 0);

    for (uint32_t i = 0; i < tensor_size; i++) {
        index[0] = i;
        EXPECT_EQ((int32_t)t3[index], -1);
    }
}

TEST(BasicTensorOperations, MulTensor) {
    uint32_t tensor_size = 3;
    Tensor t1({tensor_size}, Type::INT32);
    Tensor t2({tensor_size}, Type::INT32);
    t1 = -1;
    t2 = 2;

    auto t3 = t1 * t2;
    std::vector<uint32_t> index(1, 0);

    for (uint32_t i = 0; i < tensor_size; i++) {
        index[0] = i;
        EXPECT_EQ((int32_t)t3[index], -2);
    }
}

TEST(BasicTensorOperations, DivTensor) {
    uint32_t tensor_size = 3;
    Tensor t1({tensor_size}, Type::INT32);
    Tensor t2({tensor_size}, Type::INT32);
    t1 = -16;
    t2 = -4;

    auto t3 = t1 / t2;
    std::vector<uint32_t> index(1, 0);

    for (uint32_t i = 0; i < tensor_size; i++) {
        index[0] = i;
        EXPECT_EQ((int32_t)t3[index], 4);
    }
}