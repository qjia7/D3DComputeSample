//*********************************************************
//
// Copyright (c) Microsoft. All rights reserved.
// This code is licensed under the MIT License (MIT).
// THIS CODE IS PROVIDED *AS IS* WITHOUT WARRANTY OF
// ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING ANY
// IMPLIED WARRANTIES OF FITNESS FOR A PARTICULAR
// PURPOSE, MERCHANTABILITY, OR NON-INFRINGEMENT.
//
//*********************************************************

#pragma once
#include <stdexcept>
using namespace DirectX;

// Note that while ComPtr is used to manage the lifetime of resources on the CPU,
// it has no understanding of the lifetime of resources on the GPU. Apps must account
// for the GPU lifetime of resources to avoid destroying objects that may still be
// referenced by the GPU.
// An example of this can be found in the class method: OnDestroy().
using Microsoft::WRL::ComPtr;

inline std::string HrToString(HRESULT hr)
{
    char s_str[64] = {};
    sprintf_s(s_str, "HRESULT of 0x%08X", static_cast<UINT>(hr));
    return std::string(s_str);
}

class HrException : public std::runtime_error
{
public:
    HrException(HRESULT hr) : std::runtime_error(HrToString(hr)), m_hr(hr) {}
    HRESULT Error() const { return m_hr; }
private:
    const HRESULT m_hr;
};

class D3D12Sample
{
public:
    D3D12Sample();
    inline ID3D12Device* GetDevice() { return m_d3d12Device.Get(); }
    inline ID3D12CommandQueue* GetCommandQueue() { return m_commandQueue.Get(); }
	inline void ThrowIfFailed(HRESULT hr)
{
    if (FAILED(hr))
    {
        throw HrException(hr);
    }
}
    void Start(int argc, char *argv[]);

private:
    struct SceneConstantBuffer
    {
        int M;
        int K;
        int N;
        int TILE_K;
    };

    // Pipeline objects.
    ComPtr<ID3D12Device> m_d3d12Device;
    ComPtr<ID3D12CommandQueue> m_commandQueue;
    ComPtr<ID3D12CommandAllocator> m_computeAllocator;
    ComPtr<ID3D12RootSignature> m_computeRootSignature;
    ComPtr<ID3D12DescriptorHeap> m_cbSrvHeap;
    ComPtr<ID3D12QueryHeap> m_queryHeap;
    ComPtr<ID3D12PipelineState> m_computePSO;
    ComPtr<ID3D12GraphicsCommandList> m_commandList;
    UINT m_cbSrvDescriptorSize;

    // App resources.
    ComPtr<ID3D12Resource> m_intermediateBuffer;
    ComPtr<ID3D12Resource> m_constantBuffer;
    ComPtr<ID3D12Resource> m_intermediatebuffer1;
    ComPtr<ID3D12Resource> m_intermediatebuffer2;
    ComPtr<ID3D12Resource> m_buffer1;
    ComPtr<ID3D12Resource> m_buffer2;
    ComPtr<ID3D12Resource> m_bufferResult;
    ComPtr<ID3D12Resource> mTexture1;
    ComPtr<ID3D12Resource> mTexture2;
    ComPtr<ID3D12Resource> mTextureResult;
    ComPtr<ID3D12Resource> m_queryResult;

    SceneConstantBuffer m_constantBufferData;
    UINT8* m_pCbSrvDataBegin;

    // Synchronization objects.
    HANDLE m_computeFenceEvent;
    ComPtr<ID3D12Fence> m_computeFence;
    UINT64 m_computeFenceValue;
    UINT64 m_timestampFrequency;

    enum STORAGETYPE : short
    {
        STRUCTURED_BUFFER,
        BYTEADDRESS_BUFFER,
        TEXTURE,
        UNSUPPORTED
    };
    STORAGETYPE mStorageType;

    enum KERNELTYPE : short
    {
        SLM_8X8_4X16,
        SLM_4x4_16x16_v4,
        SLM_4x4_shared_A,
        SLM_4x4_16x16_float,
        SLM_4x4_16x16_4_FLOATS,
        MatMul_4x4_16x4_float,
        MatMul_vector_float,
        SLM_MatMul_vector_float,
        SLM_MatMul_vector_matrix_float,
        SLM_MatMul_vector_matrix_one,
    };
    KERNELTYPE mKernelType;

	UINT m_M;
	UINT m_N;
	UINT m_K;
	UINT m_tileK;
    UINT mWorkPerThreadX;
    UINT mWorkPerThreadY;
    UINT mDispatchX;
    UINT mDispatchY;
    UINT mLocalGroupSizeX;
    UINT mLocalGroupSizeY;
	UINT m_componentSize;
	UINT m_computeCount = 500;
	std::vector<float> buf1Data;
	std::vector<float> buf2Data;

	void GetHardwareAdapter(IDXGIFactory2* pFactory, IDXGIAdapter1** ppAdapter);
    void CreateDevice(const ComPtr<IDXGIFactory4>& factory);
    void LoadPipeline();
    void LoadAssets();
    void LoadBufferResources();
    void LoadTextureResources();
    void WaitForGpu();
    void RunCompute();
};
