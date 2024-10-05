#pragma once
#include "includes.hpp"

class Storage {
   public:
    Storage() = default;

    Storage(uint32_t size) : m_count_owners(new int32_t), m_size(size), m_ptr((void*)new char[size]) {
        *m_count_owners = 1;
    }

    Storage(const Storage& other) {
        m_count_owners = other.m_count_owners;
        m_ptr = other.m_ptr;
        m_size = other.m_size;
        if (m_count_owners) {
            (*m_count_owners)++;
        }
    }

    void operator=(const Storage& other) {
        m_count_owners = other.m_count_owners;
        m_ptr = other.m_ptr;
        m_size = other.m_size;
        if (m_count_owners) {
            (*m_count_owners)++;
        }
    }

    ~Storage() {
        if (m_count_owners == nullptr) return;

        (*m_count_owners)--;
        if (*m_count_owners > 0) return;
        delete m_count_owners;

        if (m_ptr == nullptr) return;
        delete[](char*) m_ptr;
    }

    template <typename T = int8_t>
    T at(uint32_t offset) const {
        LOG_IF(FATAL, !m_ptr);
        return *(reinterpret_cast<T*>(m_ptr) + offset);
    }

    template <typename T = int8_t>
    T& at(uint32_t offset) {
        LOG_IF(FATAL, !m_ptr);
        return *(reinterpret_cast<T*>(m_ptr) + offset);
    }

   private:
    int32_t* m_count_owners{nullptr};
    uint32_t m_size{0};
    void* m_ptr{nullptr};
};