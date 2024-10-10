#pragma once
#include "includes.hpp"

class Tensor;

Tensor get_element_wise_empty_output(const Tensor& in1, const Tensor& in2);

Tensor add(const Tensor& in1, const Tensor& in2);

Tensor mul(const Tensor& in1, const Tensor& in2);
