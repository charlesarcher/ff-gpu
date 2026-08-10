/* Two mathemeticians, Peter Product and Sammy Sum are in a room with Mel Moderator.   Mel says "I am thinking of an integer A >=2 and an integer B >= A.  I have written the sum of A and B on this paper and am giving it to Sammy." He then does what he says.  "I have written the product of A and B on this other paper and am giving it to Peter" Again, he does this.  At this point, Sammy says "Peter, you do not know the value of A and you do not know the value of B."  Peter then says, "Now I do know A and I do know B."  Sammy then says, "Aha, now I know them also." */

/* The following calculates integer pairs such that all of Sammy's and Peter's above statements are true. */




#ifndef PRIMEMEM
  #define PRIMEMEM 85*uint64_t(1024*1024*1024)
#endif
#ifndef THREADS
  #define THREADS 31
#endif




#include<stdlib.h>
#include<atomic>
#include<thread>
#include<cstring>
#include<sstream>
#include<iostream>
#include<fstream>
#include<iomanip>
#include<filesystem>
#include<chrono>
#include<math.h>
#include<sys/stat.h>
#include<unistd.h>
using namespace std;
using namespace std::chrono;




inline uint64_t isqrt64(uint64_t arg)
 {uint64_t result=0,
           bitsToShift=8*sizeof(uint64_t),
           testValue=0;
  do
   {(testValue<<=2)|=arg>>(bitsToShift-=2)&3;
    result<<=1;
    uint64_t divisor=result<<1;
    if (divisor<testValue)
     {result|=1;
      testValue-=divisor|1;
     }
   }
  while (bitsToShift);
  return result;
 }





// Stein's algorithm for GCD calculation (shamelessly copied from the internet and modified slightly).
// While it's true that the Freudenthal terms must be relatively prime or the GCD of the terms
//   must be prime, selecting terms based on GCD is not a performance winner. This algorithm, however,
//   is way cool! Generic version follows in comments.
/*uint32_t int gcd(uint32_t int u, uint32_t int v) {
  int shift;
  if (u == 0) return v;
  if (v == 0) return u;
  shift = __builtin_ctz(u | v);
  u >>= __builtin_ctz(u);
  do {
    v >>= __builtin_ctz(v);
    v -= u;
    uint32_t m = (int)v >> 31;
    u += v & m;
    v = (v + m) ^ m;
  } while (v != 0);
  return u << shift;
}

inline uint32_t oddgcd(uint32_t u, uint32_t v) // Currenty unused.
 {do
   {v>>=__builtin_ctz(v);
    v-=u;
    uint32_t m=(int32_t)v>>31;
    u+=v&m;
    v=(v+m)^m;
   }
  while (v);
  return u;
 }*/





static const uint32_t threads=THREADS;  

class Prime
 {static const uint32_t pthreads=THREADS-1; //Reserve one thread to shepherd all the rest
  static const uint64_t maxPrimeAlloc=PRIMEMEM;

  typedef int Boolean;
  enum {False, True};

  public:
    Prime(const uint64_t limit, const uint32_t numberOfPrimes);
    ~Prime();
    inline uint32_t operator[](const uint32_t index) const;
    inline Boolean IsPrime(const uint64_t n) const;
    inline uint64_t MaxPrimeMapValue() const;
    inline unsigned char* PrimeMap() const; // Test-only accessor: raw primeMap buffer
    inline uint64_t PrimeMapSize() const;   // Test-only accessor: primeMap byte count

  private:
   struct Stage2ThreadData {
      thread s2Thread;
      uint64_t pStart,
               loopIncrement,
               maxPrimeMapValue;
      unsigned char* primeMap;
    };

    static void PrimeWheelStage1(const uint64_t p, unsigned char* const primeMap, const uint64_t maxPrimeMapValue, atomic_uint64_t* const blockValPtr, atomic_bool* const threadActive, Stage2ThreadData* const tData);
    static void PrimeWheelStage2(const Stage2ThreadData* const tData);
    static void SegmentFill(const uint64_t segLo, const uint64_t segHi, const uint32_t* const primeList, const uint32_t numList, unsigned char* const primeMap);
    inline static uint64_t ModularMulL(const uint64_t a, const uint64_t b, const uint64_t modulus);
    inline static uint64_t ModularPowerL(const uint64_t base, const uint64_t exponent, const uint64_t modulus);
    inline static uint64_t ModularPower(const uint64_t base, const uint64_t exponent, const uint64_t modulus);
    inline static Prime::Boolean AskMillerRabin(const uint64_t n);

    uint32_t* primes;
    uint32_t numPrimes;
    unsigned char* primeMap;
    uint64_t maxPrimeMapValue;
    
 };

Prime::Prime(const uint64_t limit, const uint32_t nPrimes): numPrimes(0)
 {uint64_t numPrimesRequested=nPrimes>1 ? nPrimes : 1;// numPrimesRequested is number of array addressable primes.
  uint64_t primeMapSize=(limit>1 ? limit : 2)+15>>4;
  maxPrimeMapValue=(primeMapSize<<4)-1;
  uint64_t estimatedMaxPrime=numPrimesRequested*(log(double(numPrimesRequested))
                                                 +log(log(double(numPrimesRequested)))); // Best maximum prime estimate
  if (maxPrimeMapValue<estimatedMaxPrime)
    maxPrimeMapValue=estimatedMaxPrime;
  primeMapSize=maxPrimeMapValue+15>>4;
  uint64_t requestedPrimesSize=sizeof(uint32_t)*numPrimesRequested;
  uint64_t totalBytes=requestedPrimesSize+primeMapSize;
  cout<<"maxPrimeMapValue "<<maxPrimeMapValue<<" numPrimesRequested "<<numPrimesRequested<<" totalBytes "<<totalBytes<<endl;
  if (totalBytes>maxPrimeAlloc)
   {// Divide maxPrimeAllocc into 8 times as many prime map values as the number of primes. Is this optimal? Doubtful as
    //   it's a subjective choice. Seems to work fairly well though runs which approach maximuim prime memory utilization 
    //   are merely functional, not necessarily time-optimal.
    if (16*maxPrimeAlloc>3*estimatedMaxPrime) // If there's enough room for thrice the number of prime values for the prime map
     {uint64_t newNumPrimesRequested=2*numPrimesRequested/5, // Only generate 40% of the primes requested
               newRequestedPrimesSize=sizeof(uint32_t)*newNumPrimesRequested;
      estimatedMaxPrime=newNumPrimesRequested*(log(double(newNumPrimesRequested))
                                             +log(log(double(newNumPrimesRequested)))); // Best maximum prime estimate
      if (newRequestedPrimesSize+(estimatedMaxPrime+1)/2>maxPrimeAlloc)// Is a prime map 8 times the number of primes too big?
       {// Scale the original prime requirements to fit in maxPrimeAlloc using a linear ratio. (This is a horrible performer.)
        numPrimesRequested=((requestedPrimesSize*maxPrimeAlloc+totalBytes-1)/totalBytes+sizeof(uint32_t)-1)&-sizeof(uint32_t);
        requestedPrimesSize=sizeof(uint32_t)*numPrimesRequested;
        cout<<"You probably should consider how badly this run is going to perform, just sayin'."<<endl;
      }
      else
       {numPrimesRequested=newNumPrimesRequested;
        requestedPrimesSize=newRequestedPrimesSize;
       }
      primeMapSize=maxPrimeAlloc-requestedPrimesSize;
      totalBytes=requestedPrimesSize+primeMapSize;
      maxPrimeMapValue=(primeMapSize<<4)-1;
      cout<<"numPrimesRequested "<<numPrimesRequested<<" maxPrimeMapValue "<<maxPrimeMapValue<<" totalBytes "<<totalBytes<<endl;
    }
    else
     {cout<<"\nThe input parameters would indicate that more memory is required. Please adjust input parameters or Prime::maxPrimeAlloc.\n"<<endl;
      exit(-1);
     }
   }
  (primes=new uint32_t[numPrimesRequested])[numPrimes++]=2;
  primeMap=new unsigned char[primeMapSize];
  memset(primeMap,0xff,primeMapSize);
  primeMap[0]^=0x80; // 1 is not considered a prime despite being only divisible by 1 and itself probably because it divides literally everything.
  uint64_t sqrtLimit=isqrt64(maxPrimeMapValue);
  cout<<"maxPrimeMapValue "<<maxPrimeMapValue<<" numPrimesRequested "<<numPrimesRequested<<" sqrtLimit "<<sqrtLimit<<endl;

  // Sieve all primes up to sqrtLimit, single-threaded on a tiny map.
  uint64_t smallMapSize=sqrtLimit+15>>4;
  unsigned char* smallMap=new unsigned char[smallMapSize];
  memset(smallMap,0xff,smallMapSize);
  smallMap[0]^=0x80; // 1 is not considered a prime
  for (uint64_t p=3; p<=isqrt64(sqrtLimit); p+=2)
    if (smallMap[p>>4]&0x80>>(p>>1&7))
      for (uint64_t i=p*p; i<=sqrtLimit; i+=p<<1)
        smallMap[i>>4]&=~(0x80>>(i>>1&7));
  uint64_t smallPrimeCount=0;
  for (uint64_t p=3; p<=sqrtLimit; p+=2)
    if (smallMap[p>>4]&0x80>>(p>>1&7)) ++smallPrimeCount;
  uint32_t* smallPrimes=new uint32_t[smallPrimeCount?smallPrimeCount:1];
  smallPrimeCount=0;
  for (uint64_t p=3; p<=sqrtLimit; p+=2)
    if (smallMap[p>>4]&0x80>>(p>>1&7))
     {if (numPrimes<numPrimesRequested) primes[numPrimes++]=p;
      smallPrimes[smallPrimeCount++]=p;
     }
  delete [] smallMap;

  // Parallel segmented fill of the big prime map. Each thread owns a contiguous,
  // byte-aligned chunk exclusively, so every byte is touched by exactly one thread
  // and clearing needs no atomics and causes no cache-line ping-pong.
  const uint64_t chunkSize=((maxPrimeMapValue+1+threads-1)/threads+15)&-uint64_t(16);
  thread segThreadsList[threads];
  for (uint32_t t=0; t<threads; ++t)
   {uint64_t segLo=uint64_t(t)*chunkSize,
            segHi=segLo+chunkSize;
    if (segHi>maxPrimeMapValue+1) segHi=maxPrimeMapValue+1;
    if (segLo>=segHi) continue;
    segThreadsList[t]=thread(&SegmentFill, segLo, segHi, smallPrimes, smallPrimeCount, primeMap);
   }
  for (uint32_t t=0; t<threads; ++t)
    if (segThreadsList[t].joinable()) segThreadsList[t].join();
  delete [] smallPrimes;

  for (uint64_t p=sqrtLimit+1|1; numPrimes<numPrimesRequested; p+=2) // Fill in any remaining prime values
    if (primeMap[p>>4]&0x80>>(p>>1&7)) primes[numPrimes++]=p;
 }

Prime::~Prime()
 {delete [] primeMap;
  delete [] primes;
 }

void Prime::PrimeWheelStage1(const uint64_t p, unsigned char* const primeMap, const uint64_t maxPrimeMapValue, atomic_uint64_t* const blockValPtr, atomic_bool* const threadActive, Stage2ThreadData* const tData)
 {const uint64_t loopIncrement=p<<1;
  const uint64_t blkVal=(p+2)*(p+2);
  uint64_t i=p*p;
  for (; i<=blkVal&&i<=maxPrimeMapValue; i+=loopIncrement)
   {atomic_uchar* ap=(atomic_uchar*)&primeMap[i>>4];
    *ap&=~(0x80>>(i>>1&7));
   }
  *blockValPtr=blkVal;
  if (tData->s2Thread.joinable()) tData->s2Thread.join(); // Wait for the stage 2 thread slot to open up
  tData->pStart=i;
  tData->loopIncrement=loopIncrement;
  tData->maxPrimeMapValue=maxPrimeMapValue;
  tData->primeMap=primeMap;
  tData->s2Thread=thread(&PrimeWheelStage2, tData);
  *threadActive=False;
 }

void Prime::PrimeWheelStage2(const Stage2ThreadData* const tData)
 {for (uint64_t i=tData->pStart; i<=tData->maxPrimeMapValue; i+=tData->loopIncrement)
   {atomic_uchar* ap=(atomic_uchar*)&tData->primeMap[i>>4];
    *ap&=~(0x80>>(i>>1&7));
   }
  }

void Prime::SegmentFill(const uint64_t segLo, const uint64_t segHi, const uint32_t* const primeList, const uint32_t numList, unsigned char* const primeMap)
 {const uint64_t subBlockSize=1<<19; // 512K values, small enough to stay cache-resident
  for (uint64_t bLo=segLo; bLo<segHi; bLo+=subBlockSize)
   {uint64_t bHi=bLo+subBlockSize;
    if (bHi>segHi) bHi=segHi;
    for (uint32_t k=0; k<numList; ++k)
     {uint64_t p=primeList[k];
      if (p*p>=bHi) break; // Larger primes no longer influence this sub-block
      uint64_t first=((bLo+p-1)/p)*p; // First multiple of p within this sub-block
      if (first<p*p) first=p*p;       // ...but no smaller than p squared
      if (!(first&1)) first+=p;       // Only the odd multiples are marked
      for (uint64_t i=first; i<bHi; i+=p<<1)
        primeMap[i>>4]&=~(0x80>>(i>>1&7)); // Plain store: each chunk is owned exclusively
     }
   }
  }

inline uint32_t Prime::operator[](const uint32_t index) const
 {if (index>=numPrimes)
   {uint32_t n=numPrimes-1;
    uint64_t p=primes[n]+2;
    for (; !IsPrime(p)||++n<index; p+=2);
    return p;
   }
  else
    return primes[index];
 }

inline Prime::Boolean Prime::IsPrime(const uint64_t n) const
 {if (n==2) return True;
  if (!(n&1)) return False;
  if (n>maxPrimeMapValue)
   {/*uint64_t testValue=isqrt64(n);
    for (uint32_t i=1; i<numPrimes&&primes[i]<=testValue; ++i)
      if (!(n%primes[i])) return False;
    for (uint64_t p=primes[numPrimes-1]+2; p<=maxPrimeMapValue&&p<=testValue; p+=2)
      if (primeMap[p>>4]&0x80>>(p>>1&7) && !(n%p)) return False;*/
    return AskMillerRabin(n);
   }
  else
    return (primeMap[n>>4]&0x80>>(n>>1&7))!=0;
 }

inline uint64_t Prime::MaxPrimeMapValue() const
 {return maxPrimeMapValue;
 }

inline unsigned char* Prime::PrimeMap() const
 {return primeMap;
 }

inline uint64_t Prime::PrimeMapSize() const
 {return (maxPrimeMapValue+1+15)>>4;
 }

inline uint64_t Prime::ModularMulL(uint64_t a, uint64_t b, uint64_t modulus)
 {uint64_t ah=a>>32,
           al=a&(uint64_t(1)<<32)-1,
           bh=b>>32,
           bl=b&(uint64_t(1)<<32)-1,
           subTotal = 0;
  if (ah&&bh)
   {uint64_t p=ah*bh;
    if (p<1<<16)
      subTotal+=((p<<48)%modulus<<16)%modulus;
    else
      subTotal+=(((ah*bh<<32)%modulus<<16)%modulus<<16)%modulus;
   }
  if (ah&&bl)
   {uint64_t p=ah*bl;
    if (p<uint64_t(1)<<32)
      subTotal+=(p<<32)%modulus;
    else
      subTotal+=((p<<16)%modulus<<16)%modulus;
   }
  if (al&&bh)
   {uint64_t p = al*bh;
    if (p<uint64_t(1)<<32)
      subTotal+=(p<<32)%modulus;
    else
      subTotal+=((p<<16)%modulus<<16)%modulus;
   }
   subTotal+=(al*bl)%modulus;
   if (subTotal>=modulus) subTotal%=modulus;
   return subTotal;
  }

inline uint64_t Prime::ModularPowerL(uint64_t base, uint64_t exponent, uint64_t modulus)
 {uint64_t result=1;
  while (exponent)
   {if (exponent&1) result=ModularMulL(result, base, modulus);
    exponent>>=1;
    base=ModularMulL(base, base, modulus);
   }
  return result;
 }

inline uint64_t Prime::ModularPower(uint64_t base, uint64_t exponent, uint64_t modulus)
 {if (modulus>=uint64_t(1)<<32) return ModularPowerL(base, exponent, modulus);
  uint64_t result=1;
  while (exponent)
   {if (exponent&1) // if odd
      result=(result*base)%modulus;
    exponent>>=1;
    base=(base*base)%modulus;
   }
  return result;
 }

inline Prime::Boolean Prime::AskMillerRabin(const uint64_t n)
 {// Miller-Rabin primality test (75% accurate each loop)
  static const uint64_t primeTestList[]={2,3,5,7,11,13,17};
  uint32_t power2=__builtin_ctzll(n>>1)+1;
  uint64_t nDiv2Odd=n>>power2;
  // For 'small' numbers, only the first few numbers in the list need to be checked.
  // Note that this Miller-Rabin test only works if maxTests is tailored to the range in which n lies.
  //   In the case of this code, primes less than 25,326,001 are assumed to always be pre-generated.
  uint32_t maxTests=/*n<2047 ? 1 : n<1373653 ? 2 : n<25326001 ? 3 : */n<3215031751 ? 4 : n<2152302898747 ? 5 : n<3474749660383 ? 6 : 7;
  // if (n >= 341,550,071,728,321) Miller-Rabin is no longer valid without additional bases (i.e. primeTestList extension)
  for (uint32_t i=0; i<maxTests; ++i)
   {uint64_t p = primeTestList[i],
             x = ModularPower(p, nDiv2Odd, n);
    if (x!=1&&x!=n-1)
     {for (uint32_t r=1; r<power2; ++r)
       {x=ModularMulL(x, x, n);
        if (x==1) return False;
        if (x==n-1) break;
       }
      if (x!=n-1) return False;
     }
   }
  return True;
}





class FreudenthalTools
 {typedef int Boolean;
  enum {False, True};

  public:
    FreudenthalTools(const uint64_t prodLimit, const Prime& p);
    ~FreudenthalTools();
    inline Boolean ProductOfTermPairsHasSingleFactorPair(const uint64_t i) const;
    inline Boolean AllButOneProductOfTermPairsOfSumOfFactorPairsHasSingleFactorPair(const uint32_t power2, const uint64_t oddProduct) const;

  private:
    const Prime& primes;
    uint64_t* factors;
 };

FreudenthalTools::FreudenthalTools(const uint64_t prodLimit, const Prime& p): primes(p)
 {const uint64_t productLimit=prodLimit>3 ? prodLimit+1 : 4,
                 greatestNumberPrimeFactors=log(double(productLimit))/log(double(2))+.5,
                 greatestNumberFactors=(greatestNumberPrimeFactors*(greatestNumberPrimeFactors-1))<<1;
  factors=new uint64_t[greatestNumberFactors];
 }

FreudenthalTools::~FreudenthalTools()
 {delete [] factors;
 }

inline FreudenthalTools::Boolean FreudenthalTools::ProductOfTermPairsHasSingleFactorPair(const uint64_t i) const
 {if (!(i&1)) return True;
  return primes.IsPrime(i-2); // Goldbach's Conjecture is your friend with benefits! There was a lot more code here once.
 }

inline FreudenthalTools::Boolean
FreudenthalTools::AllButOneProductOfTermPairsOfSumOfFactorPairsHasSingleFactorPair(const uint32_t power2, const uint64_t oddProduct) const
 {uint32_t primeIndex=1,
           primeFactor=3,
           numFactors=0,
           numPendingFactors=0,
           numMultipleFactorPairs=0;
  uint64_t product=oddProduct<<power2,
           testProduct=oddProduct,
           temp=testProduct/3,
           testFactor=3,
           factorLimit=isqrt64(product);
  factors[0]=1<<power2;
  if (factors[0]<=factorLimit)
    numMultipleFactorPairs+=!ProductOfTermPairsHasSingleFactorPair(factors[numFactors++]+oddProduct);
  for(;;)
   {if (temp*primeFactor==testProduct)
     {if (testFactor<=factorLimit)
       {if ((numMultipleFactorPairs+=!ProductOfTermPairsHasSingleFactorPair(testFactor+product/testFactor))>1) return False;
        factors[numFactors+numPendingFactors++]=testFactor;
        for (uint32_t f=0; f<numFactors; ++f)
         {uint64_t multipleSave=factors[f]*testFactor;
          if (multipleSave<=factorLimit)
           {if ((numMultipleFactorPairs+=!ProductOfTermPairsHasSingleFactorPair(multipleSave+product/multipleSave))>1) return False;
            factors[numFactors+numPendingFactors++]=multipleSave;
           }
          else
            for (uint64_t factor=factors[f]; f+1<numFactors&&factor<=factors[f+1]; ++f); // factors[] is not sorted but subsets are, skip entries until the next sorted subset.
         }
        if (numMultipleFactorPairs>1) return False;
       }
      testProduct=temp;
      temp/=primeFactor;
      testFactor*=primeFactor;
     }
    else
     {if (testProduct==1) break;
      if (primes.IsPrime(testProduct))
       {if (testProduct>factorLimit) break;
        primeFactor=testProduct;
        temp=1;
       }
      else
       {primeFactor=primes[++primeIndex];
        if (primeFactor>factorLimit) break;
        temp=testProduct/primeFactor;
       }
      numFactors+=numPendingFactors;
      numPendingFactors=0;
      testFactor=primeFactor;
     }
   }
  return numMultipleFactorPairs==1;
 }





class Sammy2Loopy
 {typedef int Boolean;
  enum {False, True};

  public:
    Sammy2Loopy(const uint32_t sumLimit,
                const FreudenthalTools& freudenthalTools,
                const Prime& primes);
    ~Sammy2Loopy();
    inline Boolean DoesSammyKnow(const uint32_t sum);
    inline uint32_t GetTermA() const;
    inline uint32_t GetTermB() const;

  private:
    inline uint32_t Power2Prime(const uint32_t sum);
    inline uint32_t Power2Composite(const uint32_t sum);
    inline uint32_t Power2Odd(const uint32_t sum);
    inline uint32_t Power2Even(const uint32_t sum);
    inline Boolean DoesPeterKnow(const uint32_t power2, const uint32_t oddPartOfEven,
                                 const uint32_t odd, const uint64_t oddProduct);


    const FreudenthalTools& freudenthalTools;
    const Prime& primes;
    uint32_t numProductsWithOnlyOneMultipleFactorPair,
             termA,
             termB,
     * const compositePower2;
    int32_t numComposite;
    Boolean termsFound;
 };

Sammy2Loopy::Sammy2Loopy(const uint32_t sumLimit,
                         const FreudenthalTools& ft,
                         const Prime& p): freudenthalTools(ft),
                                          primes(p),
                                          termA(0),
                                          termB(0),
                                          compositePower2(new uint32_t[uint32_t(log(sumLimit-2)/log(2)+.5)])
 {
 }

Sammy2Loopy::~Sammy2Loopy()
 {delete [] compositePower2;
 }

inline Sammy2Loopy::Boolean Sammy2Loopy::DoesSammyKnow(const uint32_t sum)
 {numProductsWithOnlyOneMultipleFactorPair=0;
  termsFound=False;
  if (Power2Prime(sum)<=1)
    if (Power2Odd(sum)<=1)
      if (Power2Composite(sum)<=1)
        Power2Even(sum);
  return numProductsWithOnlyOneMultipleFactorPair==1;
 }

inline uint32_t Sammy2Loopy::GetTermA() const
 {return termA;
 }

inline uint32_t Sammy2Loopy::GetTermB() const
 {return termB;
 }

inline uint32_t Sammy2Loopy::Power2Prime(const uint32_t sum)
 {// First, check the potential sums for power of two and a prime, save any non-primes
  numComposite=0;
  for (uint32_t evenTerm=4, power2=2; evenTerm<sum-2; evenTerm<<=1, ++power2)
   {uint32_t oddTerm=sum-evenTerm;
    if (primes.IsPrime(oddTerm))
     {if (++numProductsWithOnlyOneMultipleFactorPair>1) break;
      if (!termsFound&&numProductsWithOnlyOneMultipleFactorPair==1)
       {termA=evenTerm;
        termB=oddTerm;
        termsFound=True;
       }
     }
    else
      compositePower2[numComposite++]=power2;
   }
  return numProductsWithOnlyOneMultipleFactorPair;
 }

inline uint32_t Sammy2Loopy::Power2Odd(const uint32_t sum)
 {//Second, check the potential sums with odd powers of 2.
  // Run the power of 2 loop from highest to lowest since large values cause earlier exit.
  uint32_t power2;
  for (power2=2; 5<<power2<sum-2; ++power2);
  power2=power2-2|1; // An odd turn for power2-1.
  for (; power2>=2; power2-=2)
//for (uint32_t power2=3; 5<<power2<sum-2; power2+=2)
   {uint32_t evenTerm;
    for (uint32_t oddPartOfEven=5; (evenTerm=oddPartOfEven<<power2)<sum-2; oddPartOfEven+=2)
     {uint32_t oddTerm=sum-evenTerm;
      uint64_t oddProd=uint64_t(oddPartOfEven)*oddTerm;
      if (oddProd%3) // Can't have odd power of two and be a multiple of 3.
       {if ((numProductsWithOnlyOneMultipleFactorPair+=DoesPeterKnow(power2, oddPartOfEven, oddTerm, oddProd))>1) break;
        if (!termsFound&&numProductsWithOnlyOneMultipleFactorPair==1)
         {termA=evenTerm;
          termB=oddTerm;
          termsFound=True;
         }
       }
     }
   }
  return numProductsWithOnlyOneMultipleFactorPair;
 }

 inline uint32_t Sammy2Loopy::Power2Composite(const uint32_t sum)
  {//Third, check the portential sums for power of two and a composite odd term
   for (--numComposite; numComposite>=0; --numComposite)
    {uint32_t evenTerm=1<<compositePower2[numComposite],
              oddTerm=sum-evenTerm;
     if ((numProductsWithOnlyOneMultipleFactorPair+=freudenthalTools
            .AllButOneProductOfTermPairsOfSumOfFactorPairsHasSingleFactorPair(compositePower2[numComposite], oddTerm))>1) break;
     if (!termsFound&&numProductsWithOnlyOneMultipleFactorPair==1)
      {termA=evenTerm;
       termB=oddTerm;
       termsFound=True;
      }
    }
   return numProductsWithOnlyOneMultipleFactorPair;
  }

inline uint32_t Sammy2Loopy::Power2Even(const uint32_t sum)
 {//Fourth!, check the potential sums with even powers of 2.
  // Run the power of 2 loop from highest to lowest since large values cause earlier exit.
  uint32_t power2;
  for (power2=2; 3<<power2<sum-2; ++power2);
  power2=power2-1&-2; // Even things up for power2-1.
  for (; power2>=2; power2-=2)
//for (uint32_t power2=2; 3<<power2<sum-2; power2+=2)
   {uint32_t evenTerm;
    for (uint32_t oddPartOfEven=3; (evenTerm=oddPartOfEven<<power2)<sum-2; oddPartOfEven+=2)
     {uint32_t oddTerm=sum-evenTerm;
      uint64_t oddProd=uint64_t(oddPartOfEven)*oddTerm;
      if (!(oddProd%3)||primes.IsPrime(oddTerm)) // Must be multiple of 3 or a prime odd term if even power of 2.
       {if ((numProductsWithOnlyOneMultipleFactorPair+=DoesPeterKnow(power2, oddPartOfEven, oddTerm, oddProd))>1) break;
        if (!termsFound&&numProductsWithOnlyOneMultipleFactorPair==1)
         {termA=evenTerm;
          termB=oddTerm;
          termsFound=True;
         }
       }
     }
   }
  return numProductsWithOnlyOneMultipleFactorPair;
 }

inline Sammy2Loopy::Boolean Sammy2Loopy::DoesPeterKnow(const uint32_t power2, const uint32_t oddPartOfEven,
                                                       const uint32_t odd, const uint64_t oddProduct)
 {// Note that invoking this routine is invalid when either term is an exact power of 2. In other words, this routine should
  //   never be invoked if oddPartOfEven is 1.
  if (!freudenthalTools.ProductOfTermPairsHasSingleFactorPair((uint64_t(1)<<power2)+oddProduct)) return False;
  if (oddPartOfEven!=odd&&!freudenthalTools.ProductOfTermPairsHasSingleFactorPair(oddPartOfEven+(uint64_t(odd)<<power2))) return False;
  return freudenthalTools.AllButOneProductOfTermPairsOfSumOfFactorPairsHasSingleFactorPair(power2, oddProduct);
 }





class OutputTags
 {friend ostream& operator<<(ostream&, const OutputTags&);
  public:
    typedef int LineEnd;
    enum {nullString, blanks};
    OutputTags(const Prime&, const int64_t, const LineEnd=nullString);
  private:
    const Prime& prime;
    const int64_t term;
    const LineEnd lineEnd;
 };

OutputTags::OutputTags(const Prime& p, const int64_t t, const LineEnd le): prime(p),
                                                                           term(t),
                                                                           lineEnd(le)
 {
 }

ostream& operator<<(ostream& os, const OutputTags& tags)
 {if (tags.prime.IsPrime(tags.term))
   {os<<" (prime)";
    if (tags.lineEnd==OutputTags::blanks) os<<", ";
   }
  else
    if ((-tags.term&tags.term)==tags.term)
     {double power2=floor(log(double(tags.term))/log(double(2))+.5);
      os<<" (2^"<<fixed<<setprecision(0)<<power2<<')';
      if (tags.lineEnd==OutputTags::blanks) os<<','<<setw(3-floor(log10(power2)))<<" ";
     }
    else
      if (tags.lineEnd==OutputTags::blanks) os<<",         ";
  return os;
 }





class FreudenthalThreads
 {typedef int Boolean;
  enum {False, True};

  public:
    FreudenthalThreads();
    void GenerateThreads(const uint32_t sumStart, const uint32_t sumLimit, const uint64_t productLimit, const Prime& prime);

  private:
    struct ThreadData {uint32_t sumStart;
                       uint32_t sumLimit;
                       uint64_t productLimit;
                       const Prime* prime;
                       char fileName[10];
                      };
    static void FreudenthalTwins(const ThreadData thread1, const ThreadData thread2);
    inline static void RunIt(const ThreadData& thread);
    uint32_t sumCount;
  };

FreudenthalThreads::FreudenthalThreads(): sumCount(0){
 }

void FreudenthalThreads::GenerateThreads(const uint32_t sumStart, const uint32_t sumLimit, const uint64_t productLimit, const Prime& prime)
 {// Kick off threads
  uint32_t numSum = sumLimit-sumStart+1;
  double delta = double(numSum)/(2*threads);
  ThreadData thread1,
             thread2;
  thread1.sumStart=0;
  thread2.sumStart=numSum;
  thread1.productLimit=thread2.productLimit=productLimit;
  thread1.prime=thread2.prime=&prime;
  uint32_t sS1Prev=thread1.sumStart,
           sS2Prev=thread2.sumStart;
  Boolean sS1Found=True,
          sS2Found=True;
  thread threadList[threads];
  char fileList[2*threads][10];
  uint32_t numList=0;
  for (uint32_t t=0; t<threads; ++t)
   {thread1.sumStart=t*delta+.5;
    if (thread1.sumStart>sS1Prev)
     {sS1Prev=thread1.sumStart;
      sS1Found = True;
     }
    uint32_t t2Limit=2*threads-t-1;
    thread2.sumStart=t2Limit*delta+.5;
    if (thread2.sumStart<sS2Prev)
     {sS2Prev=thread2.sumStart;
      sS2Found = True;
     }
    if (sS1Found&&sS2Found)
     {sS1Found=False;
      sS2Found=False;
      thread1.sumLimit=(t+1)*delta-.5;
      if (thread1.sumLimit<thread1.sumStart) thread1.sumLimit=thread1.sumStart;// Make sure at least one item is calculated
      stringstream fn;
      fn<<"ffpart"<<setw(3)<<setfill('0')<<t;
      fn.getline(thread1.fileName,10);
      memcpy(fileList[2*numList],thread1.fileName,10);
      if (thread1.sumStart<thread2.sumStart)
       {thread2.sumLimit=(t2Limit+1)*delta-.5;
        if (thread2.sumLimit<thread2.sumStart) thread2.sumLimit=thread2.sumStart;// Make sure at least one item is calculated
        fn.clear();
        fn<<"ffpart"<<setw(3)<<setfill('0')<<t2Limit;
        fn.getline(thread2.fileName,10);
       }
      else
        memset(thread2.fileName,0x00,10); //Disable thread 2
      memcpy(fileList[2*numList+1],thread2.fileName,10);
      thread1.sumStart+=sumStart;// input sumStart is just an offset by now
      thread1.sumLimit+=sumStart;
      thread2.sumStart+=sumStart;
      thread2.sumLimit+=sumStart;
//cout<<thread1.sumStart<<" "<<thread1.sumLimit<<" "<<thread2.sumStart<<" "<<thread2.sumLimit<<" "<<delta<<" "<<t<<endl;
      threadList[numList++]=thread(&FreudenthalTwins, thread1, thread2);
     }
   }

  // Tie it all back together
  for (uint32_t i=0; i<numList; ++i)
   {threadList[i].join(); //Wait for thread
    ifstream fin(fileList[2*i]);
    while (!fin.eof())
     {char line[120];
      fin.getline(line,120);
      if (!fin.eof())
       {memcpy(line,&line[7],113);
        cout<<setw(7)<<++sumCount<<line<<endl;
       }
     }
    fin.close();
    filesystem::remove(fileList[2*i]);
   }
  for (int32_t i=numList-1; i>=0; --i)
    if (fileList[2*i+1][0]!=0x00) // If the numbers are right, sometimes a second thread doesn't exist
     {ifstream fin(fileList[2*i+1]);
      while (!fin.eof())
       {char line[120];
        fin.getline(line,120);
        if (!fin.eof())
         {memcpy(line,&line[7],113);
          cout<<setw(7)<<++sumCount<<line<<endl;
         }
       }
      fin.close();
      filesystem::remove(fileList[2*i+1]);
     }
 }

void FreudenthalThreads::FreudenthalTwins(const ThreadData thread1, const ThreadData thread2)
 {RunIt(thread1);
  if (thread2.fileName[0]!=0x00) // If the numbers are right, sometimes a second thread doesn't exist
    RunIt(thread2);
 }

inline void FreudenthalThreads::RunIt(const ThreadData& thread)
 {FreudenthalTools freudenthalTools(thread.productLimit, *thread.prime);
  Sammy2Loopy sammyQuestion2(thread.sumLimit, freudenthalTools, *thread.prime);
  ofstream fout(thread.fileName);
  uint32_t count=0;
  for (uint32_t sum=thread.sumStart|1; sum<=thread.sumLimit; sum+=2)
   {uint32_t sumDiv3=sum/3;
    if (!freudenthalTools.ProductOfTermPairsHasSingleFactorPair(sum)&&(sum!=3*sumDiv3||thread.prime->IsPrime(sumDiv3))&& //Sammy's 1st statement.
        sammyQuestion2.DoesSammyKnow(sum)) //Sammy's 2nd statement which encompasses Peter's response.
     {uint32_t lowTerm=sammyQuestion2.GetTermA(),
               highTerm=sammyQuestion2.GetTermB();
      if (lowTerm>highTerm)
       {lowTerm^=highTerm;
        highTerm^=lowTerm;
        lowTerm^=highTerm;
       }
      fout<<setw(7)<<++count<<") sum ="<<setw(9)<<sum<<", product ="<<setw(16)<<uint64_t(lowTerm)*highTerm
          <<",  low term ="<<setw(9)<<lowTerm<<OutputTags(*thread.prime,lowTerm,OutputTags::blanks)
          <<"high term ="<<setw(9)<<highTerm<<OutputTags(*thread.prime,highTerm)
          <<endl;
     }
   }
  fout.close(); 
 }





int main (int argc, char* argv[])
 {char* dumpFile=nullptr;
  // Sieve-only test option: --dump-map <file> [<sumStart>] [<sumLimit>]. The
  // flag must be FIRST; strip it (and its file argument) from argv so the
  // reference's positional argument parsing below is left unchanged.
  if (argc>1&&strcmp(argv[1],"--dump-map")==0)
   {if (argc<3)
     {cout<<"--dump-map requires a file argument."<<endl;
      exit(-6);
     }
    dumpFile=argv[2];
    for (int i=3; i<argc; ++i)
      argv[i-2]=argv[i];
    argc-=2;
   }
  uint32_t sumStart,
           sumLimit;
  if (argc>2)
   {sumStart=atoi(argv[1])|1;
    sumLimit=atoi(argv[2]);
   }
  else
   {sumStart=5;
    sumLimit=argc>1 ? atoi(argv[1]) : 2627;
   }
  if (sumStart<5) sumStart=5;
  if (sumStart>sumLimit) sumLimit=sumStart;

  // Put temporary work files in their own directory
  if (chdir("ffPlayground"))
   {if (mkdir("ffPlayground", S_IRWXU|S_IRWXG|S_IROTH|S_IXOTH))
     {cout<<"Directory 'ffPlayground' cannot be created. Fix that and try again."<<endl;
      exit(-3);
     }
    else
      if (chdir("ffPlayground"))
       {cout<<"Something has gone horribly haywire with directory 'ffPlayground' creation. The only winning move is not to play."<<endl;
        exit(-5);
       }
   }
  else
   {cout<<"Directory 'ffPlayground' already exists. Delete it before trying again."<<endl;
    exit(-2);
   }

  auto start = high_resolution_clock::now();
  uint32_t maxGeneratedPrime=sumLimit+1>>1;
  uint64_t productLimit=uint64_t(maxGeneratedPrime)*maxGeneratedPrime,
           primeLimit=(productLimit>>2)+2; // Peter Product always gets a multiple of 4
  double xlogx=double(maxGeneratedPrime)/log(double(maxGeneratedPrime));
  Prime prime(primeLimit,xlogx+1.2762*xlogx/log(double(maxGeneratedPrime))); //Estimate number of primes <= maxGeneratedPrime.
  auto stop = high_resolution_clock::now();
  auto pTime = duration_cast<microseconds>(stop-start);

  if (dumpFile)
   {ofstream dumpStream(dumpFile, ios::binary|ios::trunc);
    if (!dumpStream)
     {cout<<"Cannot open dump file '"<<dumpFile<<"' for writing."<<endl;
      if (chdir("..")||rmdir("ffPlayground")) {}
      exit(-6);
     }
    dumpStream.write((const char*)prime.PrimeMap(), prime.PrimeMapSize());
    if (!dumpStream)
     {cout<<"Failed to write dump file '"<<dumpFile<<"'."<<endl;
      if (chdir("..")||rmdir("ffPlayground")) {}
      exit(-7);
     }
    dumpStream.close();
    if (chdir("..")||rmdir("ffPlayground"))
     {cout<<"Something has gone horribly wrong with the file system. Quitting seems prudent."<<endl;
      exit(-4);
     }
    return 0;
   }

  start = high_resolution_clock::now();
  uint64_t pmapSumLimit=isqrt64((prime.MaxPrimeMapValue()-2)<<2)<<1;
  FreudenthalThreads freudenthalThreads;
  if (sumLimit<=pmapSumLimit)
    freudenthalThreads.GenerateThreads(sumStart, sumLimit, productLimit, prime); // Send Freudenthal values to stdout
  else
   {freudenthalThreads.GenerateThreads(sumStart, pmapSumLimit, productLimit, prime); // Send some Freudenthal values to stdout
    freudenthalThreads.GenerateThreads(pmapSumLimit+1|1, sumLimit, productLimit, prime); // Rernormalize and send remaining Freudenthal values to stdout
   }
  stop = high_resolution_clock::now();
  auto fTime = duration_cast<microseconds>(stop-start);

  // Remove temporary working directory.
  if (chdir("..")||rmdir("ffPlayground"))
   {cout<<"Something has gone horribly wrong with the file system. Quitting seems prudent."<<endl;
    exit(-4);
   }

  cout<<"Prime time: "<<pTime.count()<<" μs"<<endl<<"Freudenthal time: "<<fTime.count()<<" μs"<<endl;

  return 0;
 }
