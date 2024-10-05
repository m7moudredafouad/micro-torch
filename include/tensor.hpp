#pragma once
#include "includes.hpp"
#include "storage.hpp"

template <typename T = uint32_t>
class Tensor {
   public:
    Tensor() = default;

    Tensor(uint32_t ndims, uint32_t* const shape) : m_ndims(ndims) {
        m_shape = new uint32_t[m_ndims];
        m_stride = new uint32_t[m_ndims];
        std::copy(&shape[0], &shape[m_ndims], m_shape);
        SetDefaultStrides();

        m_storage = Storage(NumberOfBytes());
    }

    Tensor(const std::vector<uint32_t>& shape) : m_ndims(shape.size()) {
        m_shape = new uint32_t[m_ndims];
        m_stride = new uint32_t[m_ndims];
        std::copy(&shape[0], &shape[m_ndims], m_shape);
        SetDefaultStrides();

        m_storage = Storage(NumberOfBytes());
    }

    Tensor(const std::vector<uint32_t>& shape, const std::vector<uint32_t>& stride) : m_ndims(shape.size()) {
        LOG_IF(FATAL, shape.size() != stride.size());

        m_shape = new uint32_t[m_ndims];
        m_stride = new uint32_t[m_ndims];

        std::copy(&shape[0], &shape[m_ndims], m_shape);
        std::copy(&stride[0], &stride[m_ndims], m_shape);
        m_storage = Storage(NumberOfBytes());
    }

    ~Tensor() {
        delete[] m_shape;
        delete[] m_stride;
    }

    uint32_t Size() {
        LOG_IF(FATAL, !m_shape);
        uint32_t size = 1;
        for (uint8_t i = 0; i < m_ndims; i++) {
            size *= m_shape[i];
        }

        return size;
    }

    uint32_t NumberOfBytes() { return Size() * sizeof(T); }

    void SetDefaultStrides() {
        LOG_IF(FATAL, !m_stride);
        LOG_IF(FATAL, !m_shape);

        m_stride[m_ndims - 1] = 1;

        for (int8_t i = (int8_t)m_ndims - 2; i >= 0; i--) {
            m_stride[i] = m_stride[i + 1] * m_shape[i + 1];
        }
    }

    // TODO: Remove this later when introducing iterators
    T& operator[](uint32_t offset) { return m_storage.at<T>(offset); }

    T& operator[](std::initializer_list<uint32_t> indices) {
        LOG_IF(FATAL, (uint8_t)indices.size() != m_ndims)
            << "Indices size=" << indices.size() << " don't match the full_shape=" << int(m_ndims);
        uint32_t offset = 0, i = 0;

        for (auto idx : indices) {
            LOG_IF(FATAL, idx >= m_shape[i])
                << "index is out of range, full_shape= " << m_shape[i] << " and index=" << idx;
            offset += idx * m_stride[i];
            i++;
        }

        return m_storage.at<T>(offset);
    }

    T& at(std::initializer_list<uint32_t> indices) { return this->operator[](indices); }

    Tensor<T> operator+(const Tensor<T>& other) { return Tensor<T>(m_ndims, m_shape); }

    void operator=(T value) {
        for (uint32_t i = 0; i < Size(); i++) {
            m_storage.at<T>(i) = value;
        }
    }

    Tensor<T> operator*(T value) {
        Tensor<T> tensor(m_ndims, m_shape);

        for (uint32_t i = 0; i < Size(); i++) {
            tensor.m_storage.at<T>(i) = m_storage.at<T>(i) * value;
        }

        return tensor;
    }

    Tensor<T> operator+(T value) {
        Tensor<T> tensor(m_ndims, m_shape);

        for (uint32_t i = 0; i < Size(); i++) {
            tensor.m_storage.at<T>(i) = m_storage.at<T>(i) + value;
        }

        return tensor;
    }

    template <typename TT>
    friend std::ostream& operator<<(std::ostream& os, Tensor<TT>& t);

   private:
    int64_t m_offset = 0;
    uint8_t m_ndims = 0;

    uint32_t* m_shape = nullptr;
    uint32_t* m_stride = nullptr;
    Storage m_storage;
};

template <typename T>
std::ostream& operator<<(std::ostream& os, Tensor<T>& t) {
    os << "Tensor(";

    os << "ndims=" << int(t.m_ndims) << ", offset=" << t.m_offset;

    os << ", shape=[";
    for (uint8_t i = 0; i < t.m_ndims - 1; i++) {
        os << t.m_shape[i] << ", ";
    }
    os << t.m_shape[t.m_ndims - 1] << "]";

    os << ", stride=[";
    for (uint8_t i = 0; i < t.m_ndims - 1; i++) {
        os << t.m_stride[i] << ", ";
    }
    os << t.m_stride[t.m_ndims - 1] << "]";

    os << ",\n\r";
    os << "values=[";
    for (uint32_t i = 0; i < t.Size() - 1; i++) {
        os << t[i] << ", ";
    }
    os << t[t.Size() - 1] << "]";

    os << ")";
    return os;
}
