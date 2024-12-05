#include "tensor.hpp"

namespace micro {

#define PRINT_ELEMENT(dtype, element, os)                                   \
    {                                                                       \
        switch (dtype) {                                                    \
            case Type::UINT32:                                              \
                os << uint32_t(element);                                    \
                break;                                                      \
            case Type::INT32:                                               \
                os << int32_t(element);                                     \
                break;                                                      \
            case Type::FLOAT32:                                             \
                os << float(element);                                       \
                break;                                                      \
            default:                                                        \
                LOG(FATAL) << "Can't print value with unknown tensor type"; \
        }                                                                   \
    }

std::ostream& operator<<(std::ostream& os, const Tensor& t) {
    int32_t ndims = t.m_shape.size();

    os << "Tensor(";
    os << "[";

    for (size_t i = 0; i < t.size(); i++) {
        PRINT_ELEMENT(t.m_dtype, t[i], os);
        if (i != t.size() - 1) os << ", ";
    }

    os << "], ";

    os << "shape=[";
    for (uint8_t i = 0; i < ndims; i++) {
        os << t.m_shape[i];
        if (i != ndims - 1) os << ", ";
    }

    os << "], dtype=";
    os << t.m_dtype;
    os << ")";
    return os;
}
};  // namespace micro