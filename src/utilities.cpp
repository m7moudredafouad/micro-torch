
#include "utilities.hpp"

#include "tensor.hpp"

std::ostream& operator<<(std::ostream& os, Tensor& t) {
    os << "Tensor(";

    int32_t ndims = t.m_shape.size();
    os << "ndims=" << ndims << ", offset=" << t.m_offset;

    os << ", shape=[";
    for (uint8_t i = 0; i < ndims - 1; i++) {
        os << t.m_shape[i] << ", ";
    }
    os << t.m_shape[ndims - 1] << "]";

    os << ", stride=[";
    for (uint8_t i = 0; i < ndims - 1; i++) {
        os << t.m_stride[i] << ", ";
    }
    os << t.m_stride[ndims - 1] << "]";

    os << ", values=[";
    for (uint32_t i = 0; i < t.Size() - 1; i++) {
        os << t[i] << ", ";
    }
    os << t[t.Size() - 1] << "]";

    os << ")";
    return os;
}
