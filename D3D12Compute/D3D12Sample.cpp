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

#include "pch.h"
#include "stdafx.h"
#include "D3D12Sample.h"
#include <chrono>

// #define USE_STRUCTURED_BUFFERS
// #define USE_SLM_8X8_4X16
#define PRINT_DATA

namespace
{
	//--------------------------------------------------------------------------------------
	// Inserts a resource transition operation in the command list
	//--------------------------------------------------------------------------------------
	void ResourceBarrier(_In_ ID3D12GraphicsCommandList* pCmdList, _In_ ID3D12Resource* pResource, D3D12_RESOURCE_STATES Before, D3D12_RESOURCE_STATES After, D3D12_RESOURCE_BARRIER_FLAGS Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE)
	{
		D3D12_RESOURCE_BARRIER barrierDesc = {};

		barrierDesc.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
		barrierDesc.Flags = Flags;
		barrierDesc.Transition.pResource = pResource;
		barrierDesc.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
		barrierDesc.Transition.StateBefore = Before;
		barrierDesc.Transition.StateAfter = After;

		pCmdList->ResourceBarrier(1, &barrierDesc);
	}
}

D3D12Sample::D3D12Sample() :
    m_pCbSrvDataBegin(nullptr),
    m_cbSrvDescriptorSize(0),
    m_constantBufferData{},
    m_M(1024),
    m_N(1024),
    m_K(1024),
    m_tileM(32),
    m_tileN(128),
    m_tileK(64),
	m_componentSize(4)
{
#ifdef USE_SLM_8X8_4X16
	m_tileM = 32; m_tileN = 128; m_tileK = 64;
#else
	m_tileM = 64; m_tileN = 64; m_tileK = 64;
#endif  // USE_SLM_8X8_4X16
}

void D3D12Sample::Start()
{
    LoadPipeline();
    LoadAssets();
    RunCompute();
}


// Helper function for acquiring the first available hardware adapter that supports Direct3D 12.
// If no such adapter can be found, *ppAdapter will be set to nullptr.
void D3D12Sample::GetHardwareAdapter(IDXGIFactory2* pFactory, IDXGIAdapter1** ppAdapter)
{
    ComPtr<IDXGIAdapter1> adapter;
    *ppAdapter = nullptr;

    for (UINT adapterIndex = 0; DXGI_ERROR_NOT_FOUND != pFactory->EnumAdapters1(adapterIndex, &adapter); ++adapterIndex)
    {
        DXGI_ADAPTER_DESC1 desc;
        ThrowIfFailed(adapter->GetDesc1(&desc));

        if (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE)
        {
            // Don't select the Basic Render Driver adapter.
            // If you want a software adapter, pass in "/warp" on the command line.
            continue;
        }

        // Check to see if the adapter supports Direct3D 12, but don't create the
        // actual device yet.
        if (SUCCEEDED(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_11_0, _uuidof(ID3D12Device), nullptr)))
        {
            break;
        }
    }

    *ppAdapter = adapter.Detach();
}

void D3D12Sample::CreateDevice(const ComPtr<IDXGIFactory4>& factory)
{
    ComPtr<IDXGIAdapter1> hardwareAdapter;
    GetHardwareAdapter(factory.Get(), &hardwareAdapter);

    ThrowIfFailed(D3D12CreateDevice(
        hardwareAdapter.Get(),
        D3D_FEATURE_LEVEL_11_0,
        IID_PPV_ARGS(&m_d3d12Device)
    ));
}


// Load the compute pipeline dependencies.
void D3D12Sample::LoadPipeline()
{
    UINT dxgiFactoryFlags = 0;

#if defined(_DEBUG)
    // Enable the debug layer (requires the Graphics Tools "optional feature").
    // NOTE: Enabling the debug layer after device creation will invalidate the active device.
    {
        ComPtr<ID3D12Debug> debugController;
        if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&debugController))))
        {
            debugController->EnableDebugLayer();

            // Enable additional debug layers.
            dxgiFactoryFlags |= DXGI_CREATE_FACTORY_DEBUG;
        }
    }
#endif

    // Create DXGIFactory.
    ComPtr<IDXGIFactory4> factory;
    ThrowIfFailed(CreateDXGIFactory2(dxgiFactoryFlags, IID_PPV_ARGS(&factory)));

    // Create device.
    CreateDevice(factory);   
    
    // Describe and create the command queue.
    D3D12_COMMAND_QUEUE_DESC queueDesc = { D3D12_COMMAND_LIST_TYPE_COMPUTE, 0, D3D12_COMMAND_QUEUE_FLAG_NONE };

    ThrowIfFailed(m_d3d12Device->CreateCommandQueue(&queueDesc, IID_PPV_ARGS(&m_commandQueue)));
    ThrowIfFailed(m_commandQueue->GetTimestampFrequency(&m_timestampFrequency));
    ThrowIfFailed(
        m_d3d12Device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COMPUTE, IID_PPV_ARGS(&m_computeAllocator)));


    // Create descriptor heaps.
    {
        // Describe and create a constant buffer view, shader resource views and unordered access view descriptor heap.
        // Flags indicate that this descriptor heap can be bound to the pipeline 
        // and that descriptors contained in it can be referenced by a root table.
        D3D12_DESCRIPTOR_HEAP_DESC heapDesc = {};
        heapDesc.NumDescriptors = 4;  // 1 constant buffer, 2 SRV, 1 UAV.
        heapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
        heapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
        ThrowIfFailed(m_d3d12Device->CreateDescriptorHeap(&heapDesc, IID_PPV_ARGS(&m_cbSrvHeap)));

        m_cbSrvDescriptorSize = m_d3d12Device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    }
}

// Load the sample assets.
void D3D12Sample::LoadAssets()
{
    // Create root signatures.
    {
        D3D12_FEATURE_DATA_ROOT_SIGNATURE featureData = {};

        // This is the highest version the sample supports. If CheckFeatureSupport succeeds, the HighestVersion returned will not be greater than this.
        featureData.HighestVersion = D3D_ROOT_SIGNATURE_VERSION_1_1;

        CD3DX12_DESCRIPTOR_RANGE1 ranges[3];
        CD3DX12_ROOT_PARAMETER1 rootParameters[3];

        if (FAILED(m_d3d12Device->CheckFeatureSupport(D3D12_FEATURE_ROOT_SIGNATURE, &featureData, sizeof(featureData))))
        {
            featureData.HighestVersion = D3D_ROOT_SIGNATURE_VERSION_1_0;
        }
        // Root signature for compute pass.
        ranges[0].Init(D3D12_DESCRIPTOR_RANGE_TYPE_CBV, 1, 0, 0, D3D12_DESCRIPTOR_RANGE_FLAG_NONE);
        ranges[1].Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 2, 0, 0, D3D12_DESCRIPTOR_RANGE_FLAG_NONE);
        ranges[2].Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 0, 0, D3D12_DESCRIPTOR_RANGE_FLAG_NONE);
        rootParameters[0].InitAsDescriptorTable(1, &ranges[0], D3D12_SHADER_VISIBILITY_ALL);
        rootParameters[1].InitAsDescriptorTable(1, &ranges[1], D3D12_SHADER_VISIBILITY_ALL);
        rootParameters[2].InitAsDescriptorTable(1, &ranges[2], D3D12_SHADER_VISIBILITY_ALL);

        CD3DX12_VERSIONED_ROOT_SIGNATURE_DESC rootSignatureDesc;
        rootSignatureDesc.Init_1_1(_countof(rootParameters), rootParameters, 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_NONE);

        ComPtr<ID3DBlob> signature;
        ComPtr<ID3DBlob> error;
        ThrowIfFailed(D3DX12SerializeVersionedRootSignature(&rootSignatureDesc, featureData.HighestVersion, &signature, &error));
        ThrowIfFailed(m_d3d12Device->CreateRootSignature(0, signature->GetBufferPointer(), signature->GetBufferSize(), IID_PPV_ARGS(&m_computeRootSignature)));
    }

    // Create the compute pipeline state, which includes compiling and loading shaders.
    D3D12_COMPUTE_PIPELINE_STATE_DESC descComputePSO = {};
    descComputePSO.pRootSignature = m_computeRootSignature.Get();
    ComPtr<ID3DBlob> computeShader;
    UINT compileFlags = 0;
#ifdef USE_SLM_8X8_4X16
    ThrowIfFailed(D3DCompileFromFile(L"SLM_8X8_4X16.hlsl", nullptr, nullptr, "CSMain", "cs_5_0", compileFlags, 0, &computeShader, nullptr));
#else
    const D3D_SHADER_MACRO defines[] =
    {
#ifdef USE_STRUCTURED_BUFFERS
        "USE_STRUCTURED_BUFFERS", "1",
#endif
        nullptr, nullptr
    };
    ThrowIfFailed(D3DCompileFromFile(L"SLM_4x4_16x16.hlsl", defines, nullptr, "main", "cs_5_0", compileFlags, 0, &computeShader, nullptr));
#endif
    descComputePSO.CS = CD3DX12_SHADER_BYTECODE(computeShader.Get());
    ThrowIfFailed(m_d3d12Device->CreateComputePipelineState(&descComputePSO, IID_PPV_ARGS(&m_computePSO)));
    m_computePSO->SetName(L"Compute PSO");

    // Create the command list.
    ThrowIfFailed(
        m_d3d12Device->CreateCommandList(
            0,
            D3D12_COMMAND_LIST_TYPE_COMPUTE,
            m_computeAllocator.Get(),
            m_computePSO.Get(),
            IID_PPV_ARGS(&m_commandList)));

    // Create a constant buffer.
    {
        ThrowIfFailed(m_d3d12Device->CreateCommittedResource(
            &CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD),
            D3D12_HEAP_FLAG_NONE,
            &CD3DX12_RESOURCE_DESC::Buffer(256),
            D3D12_RESOURCE_STATE_GENERIC_READ,
            nullptr,
            IID_PPV_ARGS(&m_intermediateBuffer)));

        ThrowIfFailed(m_d3d12Device->CreateCommittedResource(
            &CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT),
            D3D12_HEAP_FLAG_NONE,
            &CD3DX12_RESOURCE_DESC::Buffer(256),
            D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER,
            nullptr,
            IID_PPV_ARGS(&m_constantBuffer)));

        ResourceBarrier(m_commandList.Get(), m_constantBuffer.Get(), D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER, D3D12_RESOURCE_STATE_COPY_DEST);
        m_constantBufferData.M = m_M;
        m_constantBufferData.N = m_K;
        m_constantBufferData.K = m_N;
        m_constantBufferData.TILE_K = m_tileK;
		D3D12_SUBRESOURCE_DATA bufferData = {};
        bufferData.pData = &m_constantBufferData;
        bufferData.RowPitch = sizeof(m_constantBufferData);
        UpdateSubresources(m_commandList.Get(), m_constantBuffer.Get(), m_intermediateBuffer.Get(), 0, 0, 1, &bufferData);
        ResourceBarrier(m_commandList.Get(), m_constantBuffer.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER);

        // Describe and create a constant buffer view.
        D3D12_CONSTANT_BUFFER_VIEW_DESC cbvDesc = {};
        cbvDesc.BufferLocation = m_constantBuffer->GetGPUVirtualAddress();
        cbvDesc.SizeInBytes = (sizeof(SceneConstantBuffer) + 255) & ~255;    // CB size is required to be 256-byte aligned.
        CD3DX12_CPU_DESCRIPTOR_HANDLE cbHandle(m_cbSrvHeap->GetCPUDescriptorHandleForHeapStart());
        m_d3d12Device->CreateConstantBufferView(&cbvDesc, cbHandle);
    }

    LoadSizeDependentResources();

    // Close the command list and execute it to begin the buffer copy into
    // the default heap.
    ThrowIfFailed(m_commandList->Close());
    ID3D12CommandList* ppCommandLists[] = { m_commandList.Get() };
    m_commandQueue->ExecuteCommandLists(_countof(ppCommandLists), ppCommandLists);

    // Create synchronization objects and wait until assets have been uploaded to the GPU.
    {
        ThrowIfFailed(m_d3d12Device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&m_computeFence)));
        m_computeFenceValue = 1;

        // Create an event handle to use for frame synchronization.
        m_computeFenceEvent = CreateEvent(nullptr, FALSE, FALSE, nullptr);
        if (m_computeFenceEvent == nullptr)
        {
            ThrowIfFailed(HRESULT_FROM_WIN32(GetLastError()));
        }

        // Wait for the command list to execute; we are reusing the same command
        // list in our main loop but for now, we just want to wait for setup to
        // complete before continuing.
        WaitForGpu();
    }

}


void D3D12Sample::LoadSizeDependentResources()
{
    {
        // Create the buffer1.
        const UINT elementCount = m_M * m_K;
        for ( int i = 0; i < elementCount; ++i )
        {
            buf1Data.push_back((float) rand() / float(RAND_MAX));
        }
        const UINT bufferSize = buf1Data.size() * sizeof(float);

        ThrowIfFailed(m_d3d12Device->CreateCommittedResource(
            &CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD),
            D3D12_HEAP_FLAG_NONE,
            &CD3DX12_RESOURCE_DESC::Buffer(bufferSize),
            D3D12_RESOURCE_STATE_GENERIC_READ,
            nullptr,
            IID_PPV_ARGS(&m_intermediatebuffer1)));

        ThrowIfFailed(m_d3d12Device->CreateCommittedResource(
            &CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT),
            D3D12_HEAP_FLAG_NONE,
            &CD3DX12_RESOURCE_DESC::Buffer(bufferSize),
            D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
            nullptr,
            IID_PPV_ARGS(&m_buffer1)));

        ResourceBarrier(m_commandList.Get(), m_buffer1.Get(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COPY_DEST);
        D3D12_SUBRESOURCE_DATA bufferData = {};
        bufferData.pData = buf1Data.data();
        bufferData.RowPitch = bufferSize;
        UpdateSubresources(m_commandList.Get(), m_buffer1.Get(), m_intermediatebuffer1.Get(), 0, 0, 1, &bufferData);
        ResourceBarrier(m_commandList.Get(), m_buffer1.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);

        // Create SRV for the buffer1
        D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc = {};
        srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        srvDesc.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
        srvDesc.Buffer.FirstElement = 0;
#ifdef USE_STRUCTURED_BUFFERS
        srvDesc.Format = DXGI_FORMAT_UNKNOWN;
        srvDesc.Buffer.NumElements = elementCount / m_componentSize;
        srvDesc.Buffer.StructureByteStride = m_componentSize * sizeof(float);
        srvDesc.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_NONE;
#else
        srvDesc.Format = DXGI_FORMAT_R32_TYPELESS;
        srvDesc.Buffer.NumElements = elementCount;
        srvDesc.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_RAW;
#endif // USE_STRUCTURED_BUFFERS
        CD3DX12_CPU_DESCRIPTOR_HANDLE srvHandle(m_cbSrvHeap->GetCPUDescriptorHandleForHeapStart());
        srvHandle.Offset(1, m_cbSrvDescriptorSize); // First one is for constant buffer.
        m_d3d12Device->CreateShaderResourceView(m_buffer1.Get(), &srvDesc, srvHandle);
    }

	{
        // create the buffer2
        const UINT elementCount = m_K * m_N;
        for ( int i = 0; i < elementCount; ++i )
        {
            buf2Data.push_back((float) rand() / float(RAND_MAX));
        }
        const UINT bufferSize = buf2Data.size() * sizeof(float);

        ThrowIfFailed(m_d3d12Device->CreateCommittedResource(
            &CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD),
            D3D12_HEAP_FLAG_NONE,
            &CD3DX12_RESOURCE_DESC::Buffer(bufferSize),
            D3D12_RESOURCE_STATE_GENERIC_READ,
            nullptr,
            IID_PPV_ARGS(&m_intermediatebuffer2)));

        ThrowIfFailed(m_d3d12Device->CreateCommittedResource(
            &CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT),
            D3D12_HEAP_FLAG_NONE,
            &CD3DX12_RESOURCE_DESC::Buffer(bufferSize),
            D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
            nullptr,
            IID_PPV_ARGS(&m_buffer2)));

        ResourceBarrier(m_commandList.Get(), m_buffer2.Get(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COPY_DEST);
        D3D12_SUBRESOURCE_DATA bufferData = {};
        bufferData.pData = buf2Data.data();
        bufferData.RowPitch = bufferSize;
        UpdateSubresources(m_commandList.Get(), m_buffer2.Get(), m_intermediatebuffer2.Get(), 0, 0, 1, &bufferData);
        ResourceBarrier(m_commandList.Get(), m_buffer2.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);

        // Create SRV for buffer2
        D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc = {};
        srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        srvDesc.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
        srvDesc.Buffer.FirstElement = 0;
#ifdef USE_STRUCTURED_BUFFERS
        srvDesc.Format = DXGI_FORMAT_UNKNOWN;
        srvDesc.Buffer.NumElements = elementCount / m_componentSize;
        srvDesc.Buffer.StructureByteStride = m_componentSize * sizeof(float);
        srvDesc.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_NONE;
#else
        srvDesc.Format = DXGI_FORMAT_R32_TYPELESS;
        srvDesc.Buffer.NumElements = elementCount;
        srvDesc.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_RAW;
#endif
        CD3DX12_CPU_DESCRIPTOR_HANDLE srvHandle(m_cbSrvHeap->GetCPUDescriptorHandleForHeapStart());
        srvHandle.Offset(2, m_cbSrvDescriptorSize); // First one is for constant buffer. Senond one is for buffer1
        m_d3d12Device->CreateShaderResourceView(m_buffer2.Get(), &srvDesc, srvHandle);
	}
        // Create bufferResult and UAV for it.
        {
        const UINT elementCount = m_M * m_N;
        const UINT bufferSize = elementCount * sizeof(float);

            ThrowIfFailed(m_d3d12Device->CreateCommittedResource(
                &CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT),
                D3D12_HEAP_FLAG_NONE,
                &CD3DX12_RESOURCE_DESC::Buffer(bufferSize, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS),
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                nullptr,
                IID_PPV_ARGS(&m_bufferResult))
            );

            // Create UAV for bufferResult
            D3D12_UNORDERED_ACCESS_VIEW_DESC uavDesc = {};
            uavDesc.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
            uavDesc.Buffer.FirstElement = 0;
#ifdef USE_STRUCTURED_BUFFERS
            uavDesc.Format = DXGI_FORMAT_UNKNOWN;
            uavDesc.Buffer.NumElements = elementCount / m_componentSize;
            uavDesc.Buffer.StructureByteStride = m_componentSize * sizeof(float);
            uavDesc.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_NONE;
#else
            uavDesc.Format = DXGI_FORMAT_R32_TYPELESS;
            uavDesc.Buffer.NumElements = elementCount;
            uavDesc.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_RAW;
#endif
            CD3DX12_CPU_DESCRIPTOR_HANDLE uavHandle(m_cbSrvHeap->GetCPUDescriptorHandleForHeapStart());
            uavHandle.Offset(3, m_cbSrvDescriptorSize); // First one is for constant buffer. Senond one is for buffer1. Third one is for buffer2.
            m_d3d12Device->CreateUnorderedAccessView(m_bufferResult.Get(), nullptr, &uavDesc, uavHandle);
        }

    // Create the query result buffer.
    {
        // Two timestamps for each frame.
        const UINT resultCount = 2 * m_computeCount;
        const UINT resultBufferSize = resultCount * sizeof(UINT64);
        D3D12_QUERY_HEAP_DESC timestampHeapDesc = {};
        timestampHeapDesc.Type = D3D12_QUERY_HEAP_TYPE_TIMESTAMP;
        timestampHeapDesc.Count = resultCount;

        ThrowIfFailed(m_d3d12Device->CreateCommittedResource(
            &CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_READBACK),
            D3D12_HEAP_FLAG_NONE,
            &CD3DX12_RESOURCE_DESC::Buffer(resultBufferSize),
            D3D12_RESOURCE_STATE_COPY_DEST,
            nullptr,
            IID_PPV_ARGS(&m_queryResult)
            ));
        ThrowIfFailed(m_d3d12Device->CreateQueryHeap(&timestampHeapDesc, IID_PPV_ARGS(&m_queryHeap)));
    }
}

void D3D12Sample::RunCompute()
{
    double flops = 2 * m_M * m_N * m_K;
    double total = 0.0;
    for (int it = 0; it < m_computeCount; it++)
    {
        // This will restart the command list and start a new record.
        ThrowIfFailed(m_computeAllocator->Reset());
        ThrowIfFailed(m_commandList->Reset(m_computeAllocator.Get(), m_computePSO.Get()));

        // Record commands.

        ID3D12DescriptorHeap* pHeaps[] = { m_cbSrvHeap.Get() };
        m_commandList->SetDescriptorHeaps(_countof(pHeaps), pHeaps);

        m_commandList->SetComputeRootSignature(m_computeRootSignature.Get());
        CD3DX12_GPU_DESCRIPTOR_HANDLE gpuSrvDescriptorHandle(m_cbSrvHeap->GetGPUDescriptorHandleForHeapStart());
        m_commandList->SetComputeRootDescriptorTable(0, gpuSrvDescriptorHandle);
        gpuSrvDescriptorHandle.Offset(1, m_cbSrvDescriptorSize);
        m_commandList->SetComputeRootDescriptorTable(1, gpuSrvDescriptorHandle);
        gpuSrvDescriptorHandle.Offset(2, m_cbSrvDescriptorSize);
        m_commandList->SetComputeRootDescriptorTable(2, gpuSrvDescriptorHandle);

        m_commandList->SetPipelineState(m_computePSO.Get());

        // Get a timestamp at the before and after dispatch command.
        const UINT timestampHeapIndex = 2 * it;
        m_commandList->EndQuery(m_queryHeap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, timestampHeapIndex);
        m_commandList->Dispatch(m_N / m_tileN, m_M / m_tileM, 1);
        m_commandList->EndQuery(m_queryHeap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, timestampHeapIndex + 1);
        m_commandList->ResolveQueryData(m_queryHeap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, timestampHeapIndex, 2, m_queryResult.Get(), timestampHeapIndex * sizeof(UINT64));

        ThrowIfFailed(m_commandList->Close());
        auto start = std::chrono::steady_clock::now();
        // Execute the command list.
        ID3D12CommandList* ppCommandLists[] = { m_commandList.Get() };
        m_commandQueue->ExecuteCommandLists(_countof(ppCommandLists), ppCommandLists);
        WaitForGpu();
        auto end = std::chrono::steady_clock::now();
        auto diff = end - start;
        if (it > 0)
        {
            total += std::chrono::duration_cast<std::chrono::milliseconds>(diff).count();
        }
    }
    double avg_time = total / (m_computeCount - 1);
    double total_kernel = 0;
    double minTime = 1e100;

    // Get the timestamp values from the result buffers.
    D3D12_RANGE readRange = {};
    const D3D12_RANGE emptyRange = {};
    for (UINT i = 0; i < m_computeCount; i++)
    {
        readRange.Begin = (2 * i) * sizeof(UINT64);
        readRange.End = readRange.Begin + 2 * sizeof(UINT64);

        void* pData = nullptr;
        ThrowIfFailed(m_queryResult->Map(0, &readRange, &pData));

        const UINT64* pTimestamps = reinterpret_cast<UINT64*>(static_cast<UINT8*>(pData) + readRange.Begin);
        const UINT64 timeStampDelta = pTimestamps[1] - pTimestamps[0];

        // Unmap with an empty range (written range).
        m_queryResult->Unmap(0, &emptyRange);

        // Calculate the GPU execution time in milliseconds.
        const UINT64 gpuTimeMS =  (timeStampDelta * 1000) / m_timestampFrequency;
        // Don't consider the first dispatch time.
        if (i > 0)
        {
            if (gpuTimeMS < minTime)
                minTime = gpuTimeMS;
            total_kernel += gpuTimeMS;
        }
    }
    double avg_kernel = 0;
    avg_kernel = total_kernel / (m_computeCount - 1);
    printf("Avg Host GFlops = %f, Avg kernel GFlops = %f, Peak Kernel GFlops = %f\n"
		    "Avg_time = %f ms, Avg_kernel_time = %f ms, min_time = %f ms\n",
           flops / avg_time / 10000 / 100,
           flops / avg_kernel / 10000 / 100,
           flops / minTime / 10000 / 100, avg_time, avg_kernel, minTime);

    m_computeAllocator->Reset();
    m_commandList->Reset(m_computeAllocator.Get(), m_computePSO.Get());
#ifdef PRINT_DATA
    // Read data back to verify the result
    UINT64 outputBufferSize = m_M * m_N * sizeof(float);
    ComPtr<ID3D12Resource> readbackBuffer;
    ThrowIfFailed(m_d3d12Device->CreateCommittedResource(
        &CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_READBACK),
        D3D12_HEAP_FLAG_NONE,
        &CD3DX12_RESOURCE_DESC::Buffer(outputBufferSize),
        D3D12_RESOURCE_STATE_COPY_DEST,
        nullptr,
        IID_PPV_ARGS(&readbackBuffer)));
    readbackBuffer->SetName(L"Readback buffer Map");
    ResourceBarrier(m_commandList.Get(), m_bufferResult.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
    m_commandList->CopyResource(readbackBuffer.Get(), m_bufferResult.Get());

    m_commandList->Close();
    ID3D12CommandList* ppCommandLists[] = { m_commandList.Get() };
    m_commandQueue->ExecuteCommandLists(_countof(ppCommandLists), ppCommandLists);
    WaitForGpu();

    float result = 0.0;
    int m = rand() % m_M;
    int n = rand() % m_N;
    D3D12_RANGE readbackBufferRange{ 0, outputBufferSize };
    FLOAT * pReadbackBufferData{};
    ThrowIfFailed(readbackBuffer->Map(
        0,
        &readbackBufferRange,
        reinterpret_cast<void**>(&pReadbackBufferData)));

    result = pReadbackBufferData[m*m_N + n];

    readbackBuffer->Unmap(0, &emptyRange);

    float acc = 0.0;
    for (unsigned int k = 0; k < m_K; k++)
    {
        acc += buf1Data[m*m_K + k] * buf2Data[k*m_N + n];
    }
    printf("The result is GPU: %f, CPU: %f\n", result, acc);
#endif // PRINT_DATA
}

// Wait for pending GPU work to complete.
void D3D12Sample::WaitForGpu()
{
    // Schedule a Signal command in the queue.
    ThrowIfFailed(m_commandQueue->Signal(m_computeFence.Get(), m_computeFenceValue));

    // Wait until the fence has been processed.
    ThrowIfFailed(m_computeFence->SetEventOnCompletion(m_computeFenceValue, m_computeFenceEvent));
    WaitForSingleObjectEx(m_computeFenceEvent, INFINITE, FALSE);

    // Increment the fence value for the current frame.
    m_computeFenceValue++;
}
