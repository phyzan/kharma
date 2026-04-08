# Foivos's workstation
# AMD Ryzen 7 7700X (Zen 4) + AMD RX 7900 XT (RDNA 3)

if [[ "$HOSTNAME" == *"foivos"* ]] || [[ "$HOST" == *"foivos"* ]] || [[ "$ARGS" == *"foivos"* ]]; then

  echo ">>> Loaded foivos machine config (NPROC=16) <<<"

  # ROCm/HIP environment setup
  export ROCM_PATH=/opt/rocm
  export HIP_PATH=$ROCM_PATH
  export PATH=$ROCM_PATH/bin:$PATH
  export LD_LIBRARY_PATH=$ROCM_PATH/lib:$LD_LIBRARY_PATH
  export LIBRARY_PATH=$ROCM_PATH/lib:$LIBRARY_PATH
  export HIP_VISIBLE_DEVICES=0  # Use Radeon RX 7900 XT (gfx1100)

  # Ryzen 7 7700X is Zen 4, use ZEN3 as fallback if ZEN4 not available
  HOST_ARCH=ZEN3

  # Use all 8 cores (16 threads)
  NPROC=16

  # Default compilers
  C_NATIVE=gcc
  CXX_NATIVE=g++

  if [[ "$ARGS" == *"hip"* ]]; then
    # HIP compile for AMD RX 7900 XT (RDNA 3, gfx1100)
    DEVICE_ARCH=NAVI1100

    # Use hipcc compiler
    CXX_NATIVE=hipcc
    C_NATIVE=hipcc
    EXTRA_FLAGS="-DCLANG_RT_LIBRARY=/opt/rocm/llvm/lib/clang/22/lib/linux/libclang_rt.builtins-x86_64.a $EXTRA_FLAGS"

  else
    # CPU-only runtime settings
    :
  fi

fi
