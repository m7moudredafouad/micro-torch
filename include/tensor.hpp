#pragma once
#include "includes.hpp"
#include "kernels.hpp"
#include "storage.hpp"
#include "utilities.hpp"

template <typename T = uint32_t>
class Tensor {
   public:
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

    Tensor(const Tensor<T>& other) {
        m_offset = other.m_offset;
        m_ndims = other.m_ndims;
        m_shape = new uint32_t[m_ndims];
        m_stride = new uint32_t[m_ndims];
        std::copy(&other.m_shape[0], &other.m_shape[m_ndims], m_shape);
        std::copy(&other.m_stride[0], &other.m_stride[m_ndims], m_stride);
        m_storage = other.m_storage;
    }

    Tensor<T>& operator=(const Tensor<T>& other) {
        if (this == &other) return *this;

        m_offset = other.m_offset;
        m_ndims = other.m_ndims;
        m_shape = new uint32_t[m_ndims];
        m_stride = new uint32_t[m_ndims];
        std::copy(&other.m_shape[0], &other.m_shape[m_ndims], m_shape);
        std::copy(&other.m_stride[0], &other.m_stride[m_ndims], m_stride);
        m_storage = other.m_storage;

        return *this;
    }

    ~Tensor() {
        if (m_shape) {
            delete[] m_shape;
        }

        if (m_stride) {
            delete[] m_stride;
        }
    }

    uint32_t Size() const {
        LOG_IF(FATAL, !m_shape);
        uint32_t size = 1;
        for (int8_t i = 0; i < m_ndims; i++) {
            size *= m_shape[i];
        }

        return size;
    }

    uint32_t NumberOfBytes() const { return Size() * sizeof(T); }

    void SetDefaultStrides() {
        LOG_IF(FATAL, !m_stride);
        LOG_IF(FATAL, !m_shape);

        m_stride[m_ndims - 1] = 1;

        for (int8_t i = (int8_t)m_ndims - 2; i >= 0; i--) {
            m_stride[i] = m_stride[i + 1] * m_shape[i + 1];
        }
    }

    T operator[](std::initializer_list<uint32_t> indices) const {
        LOG_IF(FATAL, (int8_t)indices.size() != m_ndims)
            << "Indices size=" << indices.size() << " don't match the full_shape=" << int(m_ndims);
        uint32_t offset = 0, i = 0;

        for (auto idx : indices) {
            LOG_IF(FATAL, idx >= m_shape[i])
                << "index is out of range, full_shape= " << m_shape[i] << " and index=" << idx;
            offset += idx * m_stride[i];
            i++;
        }

        return this->operator[](offset);
    }

    T& operator[](std::initializer_list<uint32_t> indices) {
        LOG_IF(FATAL, (int8_t)indices.size() != m_ndims)
            << "Indices size=" << indices.size() << " don't match the full_shape=" << int(m_ndims);
        uint32_t offset = 0, i = 0;

        for (auto idx : indices) {
            LOG_IF(FATAL, idx >= m_shape[i])
                << "index is out of range, full_shape= " << m_shape[i] << " and index=" << idx;
            offset += idx * m_stride[i];
            i++;
        }

        return this->operator[](offset);
    }

    T& at(std::initializer_list<uint32_t> indices) { return this->operator[](indices); }

    Tensor<T> operator+(T value) {
        Tensor<T> tensor(1, (uint32_t [1]){1});
        tensor[0] = value;
        return add(*this, tensor);
    }

    Tensor<T> operator+(const Tensor<T>& other) { return add(*this, other); }

    void operator=(T value) {
        for (uint32_t i = 0; i < Size(); i++) {
            this->operator[](i) = value;
        }
    }

    Tensor<T> operator*(T value) {
        Tensor<T> tensor(m_ndims, m_shape);

        for (uint32_t i = 0; i < Size(); i++) {
            tensor[i] = this->operator[](i) * value;
        }

        return tensor;
    }

    template <typename TT>
    friend std::ostream& operator<<(std::ostream& os, Tensor<TT>& t);
    template <typename TT>
    friend Tensor<TT> add(const Tensor<TT>& in1, const Tensor<TT>& in2);

   private:
    T operator[](uint32_t flatten_index) const {
        LOG_IF(FATAL, flatten_index >= Size()) << "index out of range";
        return *reinterpret_cast<T*>(m_storage.at(flatten_index * sizeof(T)));
    }

    T& operator[](uint32_t flatten_index) {
        LOG_IF(FATAL, flatten_index >= Size()) << "index out of range";
        return *reinterpret_cast<T*>(m_storage.at(flatten_index * sizeof(T)));
    }

    T broadcasted_read(std::initializer_list<uint32_t> indices) const {
        int8_t ndims = indices.size();
        uint32_t offset = 0, i = 0, j = 0;

        for (auto idx : indices) {
            if ((ndims - i) > m_ndims) {
                i++;
                continue;
            }

            if (idx < m_shape[j]) {
                offset += idx * m_stride[j];
            } else {
                LOG_IF(FATAL, m_shape[j] != 1) << "Broadcasting failed";
            }

            j++;
        }

        return this->operator[](offset);
    }

   private:
    int64_t m_offset = 0;
    int8_t m_ndims = 0;

    uint32_t* m_shape = nullptr;
    uint32_t* m_stride = nullptr;
    Storage m_storage;
};