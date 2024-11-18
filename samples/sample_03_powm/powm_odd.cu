#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <cuda.h>
#include <gmp.h>
#include "cgbn/cgbn.h"
#include "../utility/support.h"

template<uint32_t tpi, uint32_t bits, uint32_t window_bits>
class powm_params_t {
  public:
  // parameters used by the CGBN context
  static const uint32_t TPB=0;                     // threads per block (zero means to use the blockDim.x)
  static const uint32_t MAX_ROTATION=4;            // must be small power of 2, imperically, 4 works well
  static const uint32_t SHM_LIMIT=0;               // number of bytes of dynamic shared memory available to the kernel
  static const bool     CONSTANT_TIME=false;       // constant time implementations aren't available yet

  // parameters used locally in the application
  static const uint32_t TPI=tpi;                   // threads per instance
  static const uint32_t BITS=bits;                 // number of bits per instance
  static const uint32_t WINDOW_BITS=window_bits;   // number of bits to use for the windowed exponentiation
};

template<class params>
class powm_odd_t {
  public:
  static const uint32_t window_bits=params::WINDOW_BITS;  // used a lot, give it an instance variable

  typedef struct {
    cgbn_mem_t<params::BITS> x;
    cgbn_mem_t<params::BITS> power;
    cgbn_mem_t<params::BITS> modulus;
    cgbn_mem_t<params::BITS> result;
  } instance_t;

  typedef cgbn_context_t<params::TPI, params>   context_t;
  typedef cgbn_env_t<context_t, params::BITS>   env_t;
  typedef typename env_t::cgbn_t                bn_t;
  typedef typename env_t::cgbn_local_t          bn_local_t;

  context_t _context;
  env_t     _env;
  int32_t   _instance;
  
  __device__ __forceinline__ powm_odd_t(cgbn_monitor_t monitor, cgbn_error_report_t *report, int32_t instance) :
    _context(monitor, report, (uint32_t)instance), _env(_context), _instance(instance) {}

  __device__ __forceinline__ void fixed_window_powm_odd(bn_t &result, const bn_t &x, const bn_t &power, const bn_t &modulus) {
    bn_t       t;
    bn_local_t window[1<<window_bits];
    int32_t    index, position, offset;
    uint32_t   np0;

    // conmpute x^power mod modulus, using the fixed window algorithm, requires:  x<modulus,  modulus is odd

    // compute x^0 (in Montgomery space, this is just 2^BITS - modulus)
    cgbn_negate(_env, t, modulus);
    cgbn_store(_env, window+0, t);

    // convert x into Montgomery space, store into window table
    np0=cgbn_bn2mont(_env, result, x, modulus);
    cgbn_store(_env, window+1, result);
    cgbn_set(_env, t, result);

    // compute x^2, x^3, ... x^(2^window_bits-1), store into window table
    #pragma nounroll
    for(index=2;index<(1<<window_bits);index++) {
      cgbn_mont_mul(_env, result, result, t, modulus, np0);
      cgbn_store(_env, window+index, result);
    }

    position=params::BITS - cgbn_clz(_env, power); // find leading high bit

    // break the exponent into chunks, each window_bits in length
    // load the most significant non-zero exponent chunk
    offset=position % window_bits;
    if(offset==0)
      position=position-window_bits;
    else
      position=position-offset;
    index=cgbn_extract_bits_ui32(_env, power, position, window_bits);
    cgbn_load(_env, result, window+index);

    while(position>0) { // process the remaining exponent chunks
      #pragma nounroll
      for(int sqr_count=0;sqr_count<window_bits;sqr_count++)
        cgbn_mont_sqr(_env, result, result, modulus, np0);

      position=position-window_bits;
      index=cgbn_extract_bits_ui32(_env, power, position, window_bits);
      cgbn_load(_env, t, window+index);
      cgbn_mont_mul(_env, result, result, t, modulus, np0);
    }

    cgbn_mont2bn(_env, result, result, modulus, np0); // we've processed the exponent now, convert back to normal space
  }
  
  __device__ __forceinline__ void sliding_window_powm_odd(bn_t &result, const bn_t &x, const bn_t &power, const bn_t &modulus) {
    bn_t         t, starts;
    int32_t      index, position, leading;
    uint32_t     mont_inv;
    bn_local_t   odd_powers[1<<window_bits-1];

    // conmpute x^power mod modulus, using Constant Length Non-Zero windows (CLNZ), requires:  x<modulus,  modulus is odd
        
    // find the leading one in the power
    leading=params::BITS-1-cgbn_clz(_env, power);
    if(leading>=0) {
      mont_inv=cgbn_bn2mont(_env, result, x, modulus); // convert x into Montgomery space, store in the odd powers table

      cgbn_mont_sqr(_env, t, result, modulus, mont_inv); // compute t=x^2 mod modulus
      
      // compute odd powers window table: x^1, x^3, x^5, ...
      cgbn_store(_env, odd_powers, result);
      #pragma nounroll
      for(index=1;index<(1<<window_bits-1);index++) {
        cgbn_mont_mul(_env, result, result, t, modulus, mont_inv);
        cgbn_store(_env, odd_powers+index, result);
      }

      cgbn_set_ui32(_env, starts, 0); // starts contains an array of bits indicating the start of a window
  
      // organize p as a sequence of odd window indexes
      position=0;
      while(true) {
        if(cgbn_extract_bits_ui32(_env, power, position, 1)==0)
          position++;
        else {
          cgbn_insert_bits_ui32(_env, starts, starts, position, 1, 1);
          if(position+window_bits>leading)
            break;
          position=position+window_bits;
        }
      }
  
      // load first window.  Note, since the window index must be odd, we have to
      // divide it by two before indexing the window table.  Instead, we just don't
      // load the index LSB from power
      index=cgbn_extract_bits_ui32(_env, power, position+1, window_bits-1);
      cgbn_load(_env, result, odd_powers+index);
      position--;

      while(position>=0) { // Process remaining windows
        cgbn_mont_sqr(_env, result, result, modulus, mont_inv);
        if(cgbn_extract_bits_ui32(_env, starts, position, 1)==1) { // found a window, load the index
          index=cgbn_extract_bits_ui32(_env, power, position+1, window_bits-1);
          cgbn_load(_env, t, odd_powers+index);
          cgbn_mont_mul(_env, result, result, t, modulus, mont_inv);
        }
        position--;
      }

      cgbn_mont2bn(_env, result, result, modulus, mont_inv); // convert result from Montgomery space
    }
    else cgbn_set_ui32(_env, result, 1); // p=0, thus x^p mod modulus=1
  }
  
  __host__ static instance_t *generate_instances(uint32_t count) {
      static cgbn_mem_t<params::BITS> tmp;

      mpz_t p, gx, gy;

      set_words(tmp._limbs, "fffffffffffffffffffffffffffffffffffffffffffffffffffffffefffffc2f", params::BITS/32);

      mpz_init(p);
      mpz_init(gx);
      mpz_init(gy);

      to_mpz(p, tmp._limbs, params::BITS/32);

    instance_t *instances=(instance_t *)malloc(sizeof(instance_t)*count);
    int         index;
  
    for(index=0;index<count;index++) {
      random_words(instances[index].x._limbs, params::BITS/32);
      random_words(instances[index].power._limbs, params::BITS/32);
      random_words(instances[index].modulus._limbs, params::BITS/32);

      instances[index].modulus._limbs[0] |= 1; // ensure modulus is odd

      // ensure modulus is greater than 
      if(compare_words(instances[index].x._limbs, instances[index].modulus._limbs, params::BITS/32)>0) {
        swap_words(instances[index].x._limbs, instances[index].modulus._limbs, params::BITS/32);

        instances[index].modulus._limbs[0] |= 1; // modulus might now be even, ensure it's odd
      }
      else if(compare_words(instances[index].x._limbs, instances[index].modulus._limbs, params::BITS/32)==0) {
        instances[index].x._limbs[0] -= 1; // since modulus is odd and modulus = x, we can just subtract 1 from x
      }
    }
    return instances;
  }
  
  __host__ static void verify_results(instance_t *instances, uint32_t count) {
    mpz_t x, p, m, computed, correct;
    
    mpz_init(x);
    mpz_init(p);
    mpz_init(m);
    mpz_init(computed);
    mpz_init(correct);
    
    for(int index=0;index<count;index++) {
      to_mpz(x, instances[index].x._limbs, params::BITS/32);
      to_mpz(p, instances[index].power._limbs, params::BITS/32);
      to_mpz(m, instances[index].modulus._limbs, params::BITS/32);
      to_mpz(computed, instances[index].result._limbs, params::BITS/32);
      
      mpz_powm(correct, x, p, m);
      if(mpz_cmp(correct, computed)!=0) {
        printf("gpu inverse kernel failed on instance %d\n", index);
        return;
      }
    }
  
    mpz_clear(x);
    mpz_clear(p);
    mpz_clear(m);
    mpz_clear(computed);
    mpz_clear(correct);
    
    printf("All results match\n");
  }
};

template<class params>
__global__ void kernel_powm_odd(cgbn_error_report_t *report, typename powm_odd_t<params>::instance_t *instances, uint32_t count) {
  int32_t instance;

  instance=(blockIdx.x*blockDim.x + threadIdx.x)/params::TPI;
  if(instance>=count)
    return;

  powm_odd_t<params>                 po(cgbn_report_monitor, report, instance);
  typename powm_odd_t<params>::bn_t  r, x, p, m;

  cgbn_load(po._env, x, &(instances[instance].x));
  cgbn_load(po._env, p, &(instances[instance].power));
  cgbn_load(po._env, m, &(instances[instance].modulus));

  po.fixed_window_powm_odd(r, x, p, m); // po.sliding_window_powm_odd(r, x, p, m);

  cgbn_store(po._env, &(instances[instance].result), r);
}

template<class params>
void run_test(uint32_t nelts, uint32_t addElms) {
  typedef typename powm_odd_t<params>::instance_t instance_t;

  instance_t          *instances, *gpuInstances;
  cgbn_error_report_t *report;
  int32_t              TPB=(params::TPB==0) ? 128 : params::TPB;    // default threads per block to 128
  int32_t              TPI=params::TPI, IPB=TPB/TPI;                // IPB is instances per block

  printf("Genereating instances ...\n");
  instances=powm_odd_t<params>::generate_instances(instance_count);

  printf("Copying instances to the GPU ...\n");
  CUDA_CHECK(cudaSetDevice(0));
  CUDA_CHECK(cudaMalloc((void **)&gpuInstances, sizeof(instance_t)*instance_count));
  CUDA_CHECK(cudaMemcpy(gpuInstances, instances, sizeof(instance_t)*instance_count, cudaMemcpyHostToDevice));

  CUDA_CHECK(cgbn_error_report_alloc(&report)); 

  printf("Running GPU kernel ...\n");

  kernel_powm_odd<params><<<(instance_count+IPB-1)/IPB, TPB>>>(report, gpuInstances, instance_count);

  CUDA_CHECK(cudaDeviceSynchronize());
  CGBN_CHECK(report);

  printf("Copying results back to CPU ...\n");
  CUDA_CHECK(cudaMemcpy(instances, gpuInstances, sizeof(instance_t)*instance_count, cudaMemcpyDeviceToHost));

  printf("Verifying the results ...\n");
  powm_odd_t<params>::verify_results(instances, instance_count);

  free(instances);
  CUDA_CHECK(cudaFree(gpuInstances));
  CUDA_CHECK(cgbn_error_report_free(report));
}

int main() {
  typedef powm_params_t<4, 256, 4> params; // thr/inst = 4, bits = 256, window_bits = 4 (as per fixnum-add BYTES_TO_K_ARY_WINDOW_SIZE_TABLE[])

  run_test<params>(20000, 10000);
}
