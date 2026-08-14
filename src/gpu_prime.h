#ifndef FF_GPU_PRIME_H
#define FF_GPU_PRIME_H

#include <cstdint>

class GpuPrime {
public:
    typedef int Boolean;
    enum {False, True};
    
    GpuPrime(const uint8_t* primeMap, uint64_t maxMapValue);
    ~GpuPrime();
    
    Boolean IsPrime(uint64_t n) const;
    inline Boolean operator[](uint64_t n) const { return IsPrime(n); }
    inline uint64_t MaxPrimeMapValue() const { return maxPrimeMapValue_; }
    inline Boolean AskMillerRabin(uint64_t n) const;
    
private:
    static uint64_t ModularMulL(uint64_t a, uint64_t b, uint64_t modulus);
    static uint64_t ModularPowerL(uint64_t base, uint64_t exp, uint64_t modulus);
    static uint64_t ModularPower(uint64_t base, uint64_t exp, uint64_t modulus);
    
    const uint8_t* primeMap_;
    uint64_t maxPrimeMapValue_;
};

#endif