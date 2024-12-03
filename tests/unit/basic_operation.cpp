#include <gtest/gtest.h>

#include <tensor.hpp>

using namespace micro;

TEST(BasicTensorOperations, Assignment) {
    uint32_t tensor_size = 3;
    Tensor t1({tensor_size});
    t1 = 100.f;

    for (uint32_t i = 0; i < tensor_size; i++) {
        EXPECT_EQ((float)t1[{i}], 100.f);
    }
}

TEST(BasicTensorOperations, AddConstant) {
    uint32_t tensor_size = 3;
    Tensor t1({tensor_size}, Type::UINT32);
    t1 = 0;
    t1 += 16;

    for (uint32_t i = 0; i < tensor_size; i++) {
        EXPECT_EQ((uint32_t)t1[{i}], 16);
    }
}

TEST(BasicTensorOperations, SubConstant) {
    uint32_t tensor_size = 3;
    Tensor t1({tensor_size}, Type::UINT32);
    t1 = 16;
    t1 -= 1;

    for (uint32_t i = 0; i < tensor_size; i++) {
        EXPECT_EQ((uint32_t)t1[{i}], 15);
    }
}

TEST(BasicTensorOperations, MulConstant) {
    uint32_t tensor_size = 3;
    Tensor t1({tensor_size}, Type::UINT32);
    t1 = 1;
    t1 *= 16;

    for (uint32_t i = 0; i < tensor_size; i++) {
        EXPECT_EQ((uint32_t)t1[{i}], 16);
    }
}

TEST(BasicTensorOperations, DivConstant) {
    uint32_t tensor_size = 3;
    Tensor t1({tensor_size}, Type::UINT32);
    t1 = 16;
    t1 /= 4;

    for (uint32_t i = 0; i < tensor_size; i++) {
        EXPECT_EQ((uint32_t)t1[{i}], 4);
    }
}

TEST(BasicTensorOperations, AddTensor) {
    uint32_t tensor_size = 3;
    Tensor t1({tensor_size}, Type::UINT32);
    Tensor t2({tensor_size}, Type::UINT32);
    t1 = 1;
    t2 = 2;

    auto t3 = t1 + t2;

    for (uint32_t i = 0; i < tensor_size; i++) {
        EXPECT_EQ((uint32_t)t3[{i}], 3);
    }
}

TEST(BasicTensorOperations, SubTensor) {
    uint32_t tensor_size = 3;
    Tensor t1({tensor_size}, Type::INT32);
    Tensor t2({tensor_size}, Type::INT32);
    t1 = 1;
    t2 = 2;

    auto t3 = t1 - t2;
    for (uint32_t i = 0; i < tensor_size; i++) {
        EXPECT_EQ((int32_t)t3[{i}], -1);
    }
}

TEST(BasicTensorOperations, MulTensor) {
    uint32_t tensor_size = 3;
    Tensor t1({tensor_size}, Type::INT32);
    Tensor t2({tensor_size}, Type::INT32);
    t1 = -1;
    t2 = 2;

    auto t3 = t1 * t2;
    for (uint32_t i = 0; i < tensor_size; i++) {
        EXPECT_EQ((int32_t)t3[{i}], -2);
    }
}

TEST(BasicTensorOperations, DivTensor) {
    uint32_t tensor_size = 3;
    Tensor t1({tensor_size}, Type::INT32);
    Tensor t2({tensor_size}, Type::INT32);
    t1 = -16;
    t2 = -4;

    auto t3 = t1 / t2;
    for (uint32_t i = 0; i < tensor_size; i++) {
        EXPECT_EQ((int32_t)t3[{i}], 4);
    }
}