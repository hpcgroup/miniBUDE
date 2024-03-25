#pragma once

#include "../bude.h"
#include <string>

#include "RAJA/RAJA.hpp"
#include "umpire/Allocator.hpp"
#include "umpire/ResourceManager.hpp"
#include "camp/resource.hpp"

#ifdef IMPL_CLS
  #error IMPL_CLS was already defined
#endif
#define IMPL_CLS RajaBude

// XXX RAJA appears to support one device backend and/or one host backend at compile time
// so the device/host enable macros (RAJA_ENABLE_*) are mutually exclusive

template <size_t PPWI> class IMPL_CLS final : public Bude<PPWI> {

  using launch_policy = RAJA::LaunchPolicy< //
#if defined(RAJA_ENABLE_OPENMP)
      RAJA::omp_launch_t
#else
      RAJA::seq_launch_t
#endif
#if defined(RAJA_ENABLE_CUDA)
      ,
      RAJA::cuda_launch_t<false>
#endif
#if defined(RAJA_ENABLE_HIP)
      ,
      RAJA::hip_launch_t<false>
#endif
#if defined(RAJA_ENABLE_SYCL)
      ,
      RAJA::sycl_launch_t<false>
#endif
      >;

  using teams_x = RAJA::LoopPolicy< //
#if defined(RAJA_ENABLE_OPENMP)
      RAJA::omp_parallel_for_exec
#else
      RAJA::seq_exec
#endif
#if defined(RAJA_ENABLE_CUDA)
      ,
      RAJA::cuda_block_x_direct
#endif
#if defined(RAJA_ENABLE_HIP)
      ,
      RAJA::hip_block_x_direct
#endif
      >;

  using threads_x = RAJA::LoopPolicy< //
      RAJA::seq_exec
#if defined(RAJA_ENABLE_CUDA)
      ,
      RAJA::cuda_thread_x_loop
#endif
#if defined(RAJA_ENABLE_HIP)
      ,
      RAJA::hip_thread_x_loop
#endif
      >;

  // XXX enable USE_LOCAL_ARRAY once RAJA supports dynamically sized LocalArray
  //#define USE_LOCAL_ARRAY

public:
  static void fasten_main(size_t device, size_t wgsize,                               //
                          size_t ntypes, size_t nposes, size_t natlig, size_t natpro, //
                          const Atom *RAJA_RESTRICT proteins, const Atom *RAJA_RESTRICT ligands,
                          const FFParams *RAJA_RESTRICT forcefields, //
                          const float *RAJA_RESTRICT transforms_0, const float *RAJA_RESTRICT transforms_1,
                          const float *RAJA_RESTRICT transforms_2, const float *RAJA_RESTRICT transforms_3,
                          const float *RAJA_RESTRICT transforms_4, const float *RAJA_RESTRICT transforms_5,
                          float *RAJA_RESTRICT etotals) {

    size_t global = std::ceil(double(nposes) / PPWI);
    global = int(std::ceil(double(global) / double(wgsize)));
    size_t local = int(wgsize);

    size_t dynamic_shared_mem_size = ntypes * sizeof(FFParams);

    RAJA::launch<launch_policy>(                                           //
        static_cast<RAJA::ExecPlace>(device),                              //
        RAJA::LaunchParams(RAJA::Teams(global), RAJA::Threads(local), dynamic_shared_mem_size), //
        [=] RAJA_HOST_DEVICE(RAJA::LaunchContext ctx) {                    //
          RAJA::loop<teams_x>(ctx, RAJA::RangeSegment(0, global), [&](int gid) {
            FFParams *local_forcefield = ctx.getSharedMemory<FFParams>(ntypes);

            float etot[PPWI];
            float transform[3][4][PPWI];

            RAJA::loop<threads_x>(ctx, RAJA::RangeSegment(0, local), [&](int lid) {
              size_t ix = gid * local * PPWI + lid;
              ix = ix < nposes ? ix : nposes - PPWI;

              if (ix < ntypes) {
                local_forcefield[ix] = forcefields[ix];
              }

              // Compute transformation matrix to private memory
              const size_t lsz = local;
              for (size_t i = 0; i < PPWI; i++) {
                size_t index = ix + i * lsz;

                const float sx = std::sin(transforms_0[index]);
                const float cx = std::cos(transforms_0[index]);
                const float sy = std::sin(transforms_1[index]);
                const float cy = std::cos(transforms_1[index]);
                const float sz = std::sin(transforms_2[index]);
                const float cz = std::cos(transforms_2[index]);

                transform[0][0][i] = cy * cz;
                transform[0][1][i] = sx * sy * cz - cx * sz;
                transform[0][2][i] = cx * sy * cz + sx * sz;
                transform[0][3][i] = transforms_3[index];
                transform[1][0][i] = cy * sz;
                transform[1][1][i] = sx * sy * sz + cx * cz;
                transform[1][2][i] = cx * sy * sz - sx * cz;
                transform[1][3][i] = transforms_4[index];
                transform[2][0][i] = -sy;
                transform[2][1][i] = sx * cy;
                transform[2][2][i] = cx * cy;
                transform[2][3][i] = transforms_5[index];

                etot[i] = ZERO;
              }
            });

            ctx.teamSync();

            RAJA::loop<threads_x>(ctx, RAJA::RangeSegment(0, local), [&](int lid) {
              // Loop over ligand atoms
              size_t il = 0;
              do {
                // Load ligand atom data
                const Atom l_atom = ligands[il];
                const FFParams l_params = local_forcefield[l_atom.type];
                const bool lhphb_ltz = l_params.hphb < ZERO;
                const bool lhphb_gtz = l_params.hphb > ZERO;

                float lpos_x[PPWI], lpos_y[PPWI], lpos_z[PPWI];
                for (size_t i = 0; i < PPWI; i++) {
                  // Transform ligand atom
                  lpos_x[i] = transform[0][3][i] + l_atom.x * transform[0][0][i] + l_atom.y * transform[0][1][i] +
                              l_atom.z * transform[0][2][i];
                  lpos_y[i] = transform[1][3][i] + l_atom.x * transform[1][0][i] + l_atom.y * transform[1][1][i] +
                              l_atom.z * transform[1][2][i];
                  lpos_z[i] = transform[2][3][i] + l_atom.x * transform[2][0][i] + l_atom.y * transform[2][1][i] +
                              l_atom.z * transform[2][2][i];
                }

                // Loop over protein atoms
                size_t ip = 0;
                do {
                  // Load protein atom data
                  const Atom p_atom = proteins[ip];
                  const FFParams p_params = local_forcefield[p_atom.type];

                  const float radij = p_params.radius + l_params.radius;
                  const float r_radij = 1.f / (radij);

                  const float elcdst = (p_params.hbtype == HBTYPE_F && l_params.hbtype == HBTYPE_F) ? FOUR : TWO;
                  const float elcdst1 = (p_params.hbtype == HBTYPE_F && l_params.hbtype == HBTYPE_F) ? QUARTER : HALF;
                  const bool type_E = ((p_params.hbtype == HBTYPE_E || l_params.hbtype == HBTYPE_E));

                  const bool phphb_ltz = p_params.hphb < ZERO;
                  const bool phphb_gtz = p_params.hphb > ZERO;
                  const bool phphb_nz = p_params.hphb != ZERO;
                  const float p_hphb = p_params.hphb * (phphb_ltz && lhphb_gtz ? -ONE : ONE);
                  const float l_hphb = l_params.hphb * (phphb_gtz && lhphb_ltz ? -ONE : ONE);
                  const float distdslv =
                      (phphb_ltz ? (lhphb_ltz ? NPNPDIST : NPPDIST) : (lhphb_ltz ? NPPDIST : -FloatMax));
                  const float r_distdslv = 1.f / (distdslv);

                  const float chrg_init = l_params.elsc * p_params.elsc;
                  const float dslv_init = p_hphb + l_hphb;

                  for (size_t i = 0; i < PPWI; i++) {
                    // Calculate distance between atoms
                    const float x = lpos_x[i] - p_atom.x;
                    const float y = lpos_y[i] - p_atom.y;
                    const float z = lpos_z[i] - p_atom.z;

                    const float distij = std::sqrt(x * x + y * y + z * z);

                    // Calculate the sum of the sphere radii
                    const float distbb = distij - radij;
                    const bool zone1 = (distbb < ZERO);

                    // Calculate steric energy
                    etot[i] += (ONE - (distij * r_radij)) * (zone1 ? 2 * HARDNESS : ZERO);

                    // Calculate formal and dipole charge interactions
                    float chrg_e = chrg_init * ((zone1 ? 1 : (ONE - distbb * elcdst1)) * (distbb < elcdst ? 1 : ZERO));
                    const float neg_chrg_e = -std::fabs(chrg_e);
                    chrg_e = type_E ? neg_chrg_e : chrg_e;
                    etot[i] += chrg_e * CNSTNT;

                    // Calculate the two cases for Nonpolar-Polar repulsive interactions
                    const float coeff = (ONE - (distbb * r_distdslv));
                    float dslv_e = dslv_init * ((distbb < distdslv && phphb_nz) ? 1 : ZERO);
                    dslv_e *= (zone1 ? 1 : coeff);
                    etot[i] += dslv_e;
                  }
                } while (++ip < natpro); // loop over protein atoms
              } while (++il < natlig);   // loop over ligand atoms

              // Write results
              const size_t td_base = gid * local * PPWI + lid;

              if (td_base < nposes) {
                for (size_t i = 0; i < PPWI; i++) {
                  etotals[td_base + i * local] = etot[i] * HALF;
                }
              }
            });
          });
        });
  }

  template <typename T> static void *registerAllocation(const std::vector<T> &xs) {
    auto &rm = umpire::ResourceManager::getInstance();
    auto host_alloc = rm.getAllocator("HOST");
    auto strategy = host_alloc.getAllocationStrategy();
    // Not allowed to create a record without casting away the const
    auto host_data = const_cast<void*>(static_cast<const void*>(std::data(xs)));
    umpire::util::AllocationRecord record{host_data, sizeof(T) * std::size(xs), strategy};
    rm.registerAllocation(host_data, record);
    return host_data;
  }

  template <typename T> static T *allocate(const std::vector<T> &xs) {
    auto &rm = umpire::ResourceManager::getInstance();
    auto host_data = registerAllocation(xs);
    auto data = allocate<T>(std::size(xs));
    rm.copy(data, host_data);
    return data;
  }
    
  template <typename T> static T *allocate(const size_t size) {
    auto &rm = umpire::ResourceManager::getInstance();
#ifndef RAJA_TARGET_GPU
    auto alloc = rm.getAllocator("HOST");
#else
  #ifdef BUDE_MANAGED_ALLOC
    auto alloc = rm.getAllocator("UM");
  #else
    auto alloc = rm.getAllocator("DEVICE");
  #endif
#endif
    return static_cast<T *>(alloc.allocate(sizeof(T) * size));
  }

  template <typename T> static void deallocate(T *ptr) {
    auto &rm = umpire::ResourceManager::getInstance();
    rm.getAllocator(ptr).deallocate(ptr);
  }

  static void synchronise() {
    // nothing to do for host devices
#if defined(RAJA_ENABLE_CUDA)
    RAJA::synchronize<RAJA::cuda_synchronize>();
#endif
#if defined(RAJA_ENABLE_HIP)
    RAJA::synchronize<RAJA::hip_synchronize>();
#endif
#if defined(RAJA_ENABLE_SYCL)
    RAJA::synchronize<RAJA::sycl_synchronize>();
#endif
  }

public:
  IMPL_CLS() = default;

  [[nodiscard]] std::string name() { return "raja"; };

  [[nodiscard]] std::vector<Device> enumerateDevices() override {
    std::vector<Device> devices{{(size_t) RAJA::ExecPlace::HOST, "RAJA Host device"}};
#if defined(RAJA_TARGET_GPU)
  #if defined(RAJA_ENABLE_CUDA)
    const auto deviceName = "RAJA CUDA device";
  #endif
  #if defined(RAJA_ENABLE_HIP)
    const auto deviceName = "RAJA HIP device";
  #endif
  #if defined(RAJA_ENABLE_SYCL)
    const auto deviceName = "RAJA SYCL device";
  #endif
    devices.template emplace_back((size_t) RAJA::ExecPlace::DEVICE, deviceName);
#endif
    return devices;
  };

  [[nodiscard]] Sample fasten(const Params &p, size_t wgsize, size_t device) const override {

    Sample sample(PPWI, wgsize, p.nposes());
    auto &rm = umpire::ResourceManager::getInstance();

    auto hostToDeviceStart = now();

    auto protein = allocate(p.protein);
    auto ligand = allocate(p.ligand);
    auto transforms_0 = allocate(p.poses[0]);
    auto transforms_1 = allocate(p.poses[1]);
    auto transforms_2 = allocate(p.poses[2]);
    auto transforms_3 = allocate(p.poses[3]);
    auto transforms_4 = allocate(p.poses[4]);
    auto transforms_5 = allocate(p.poses[5]);
    auto forcefield = allocate(p.forcefield);

    synchronise();

    auto hostToDeviceEnd = now();
    sample.hostToDevice = {hostToDeviceStart, hostToDeviceEnd};
    
    auto host_energies = registerAllocation(sample.energies);
    auto results = allocate<float>(sample.energies.size());
    
    synchronise();

    for (size_t i = 0; i < p.totalIterations(); ++i) {
      auto kernelStart = now();
      fasten_main(device, wgsize, p.ntypes(), p.nposes(), p.natlig(), p.natpro(), //
                  protein, ligand, forcefield,                                    //
                  transforms_0, transforms_1, transforms_2, transforms_3, transforms_4, transforms_5, results);
      synchronise();
      auto kernelEnd = now();
      sample.kernelTimes.emplace_back(kernelStart, kernelEnd);
    }

    auto deviceToHostStart = now();
    rm.copy(host_energies, results);

    auto deviceToHostEnd = now();
    sample.deviceToHost = {deviceToHostStart, deviceToHostEnd};

    deallocate(protein);
    deallocate(ligand);
    deallocate(transforms_0);
    deallocate(transforms_1);
    deallocate(transforms_2);
    deallocate(transforms_3);
    deallocate(transforms_4);
    deallocate(transforms_5);
    deallocate(forcefield);
    deallocate(results);

    return sample;
  };
};
