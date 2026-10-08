// exp5_p2p.cu —— 不开 NCCL：一条 store 指令穿过 NVLink
#include <cstdio>
#include <cuda_runtime.h>

#define CHECK(x) do { cudaError_t e_ = (x); \
    if (e_ != cudaSuccess) { \
        printf("CUDA 错误 @%d: %s\n", __LINE__, cudaGetErrorString(e_)); \
        return 1; } } while (0)

__global__ void write_peer(int* peer, int v) {
    peer[0] = v;        // ← 全篇的主角：一条 store，写的是"另一块卡"的显存地址
}

int main() {
    int can = 0;
    CHECK(cudaDeviceCanAccessPeer(&can, 0, 1));      // 查路：卡 0 能不能直接访问卡 1？
    printf("cudaDeviceCanAccessPeer(0→1) = %d（1 = 有 P2P 通路）\n", can);
    if (!can) { printf("两卡间无 P2P 通路，本实验退出（见附录 A，exp6 照做）。\n"); return 0; }

    int* buf1 = nullptr;
    CHECK(cudaSetDevice(1));
    CHECK(cudaMalloc(&buf1, sizeof(int)));           // 备料：在卡 1 上申请 4 字节
    CHECK(cudaMemset(buf1, 0, sizeof(int)));

    CHECK(cudaSetDevice(0));
    CHECK(cudaDeviceEnablePeerAccess(1, 0));         // 开门：卡 0 的地址空间纳入卡 1 显存
    write_peer<<<1, 1>>>(buf1, 42);                  // 直写：kernel 跑在卡 0，写的是卡 1 的地址
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    int h = -1;
    CHECK(cudaSetDevice(1));
    CHECK(cudaMemcpy(&h, buf1, sizeof(int), cudaMemcpyDeviceToHost));  // 验证：拷回 host 看
    printf("卡 1 显存里的值 = %d（应为 42，且全程没有经过卡 0 的显存）\n", h);
    return 0;
}
