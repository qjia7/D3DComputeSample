cbuffer SceneConstantBuffer : register( b0 )
{
    int M;
    int K;
    int N;
    int TILE_K;
}

static uint3 gl_WorkGroupID = uint3(0, 0, 0);
static uint3 gl_LocalInvocationID = uint3(0, 0, 0);
static uint3 gl_GlobalInvocationID = uint3(0, 0, 0);

struct CS_INPUT
{
    uint3 dx_WorkGroupID : SV_GroupID;
    uint3 dx_LocalInvocationID : SV_GroupThreadID;
    uint3 dx_GlobalInvocationID : SV_DispatchThreadID;
};

void initGLBuiltins(CS_INPUT input)
{
    gl_WorkGroupID = input.dx_WorkGroupID;
    gl_LocalInvocationID = input.dx_LocalInvocationID;
    gl_GlobalInvocationID = input.dx_GlobalInvocationID;
};

#ifdef USE_TEXTURE
Texture2D<float4> src0 : register(t0);
Texture2D<float4> src1 : register(t1);
RWTexture2D<float4> dst : register(u0);

float4 mm_readA(int row, int col) {
    if (row < M && col < K / 4)
    {
        return src0.Load(int3(col, row, 0));
    }
    else {
        return float4(0, 0, 0, 0);
    }
}

float4 mm_readB(int row, int col) {
    return src1.Load(int3(col, row, 0));
}

void mm_write(int row, int col, float4 value) {
    if (row < M && col < N / 4)
    {
        dst[uint2(col, row)] = value;
    }
}
#else
#ifdef USE_STRUCTURED_BUFFERS
StructuredBuffer<float> src0 : register(t0);
StructuredBuffer<float> src1 : register(t1);
RWStructuredBuffer<float> dst : register(u0);

float4 mm_readA(int row, int col) {
    if (row < M)
    {
        int index = row * K + col;
        float4 result = float4(src0[index],
            src0[index + 1],
            src0[index + 2],
            src0[index + 3]);
        return result;
    }
    else {
        return float4(0, 0, 0, 0);
    }
}

float4 mm_readB(int row, int col) {
    int index = row * N + col;
    float4 result = float4(src1[index],
        src1[index + 1],
        src1[index + 2],
        src1[index + 3]);
    return result;
}

void mm_write(int row, int col, float4 value) {
    if (row < M && col < N)
    {
        int index = row * N + col;
        if (col < (N - 3)) {
            dst[index] = value.x;
            dst[index + 1] = value.y;
            dst[index + 2] = value.z;
            dst[index + 3] = value.w;
        }
        else if (col < (N - 2))
        {
            dst[index] = value.x;
            dst[index + 1] = value.y;
            dst[index + 2] = value.z;
        }
        else if (col < (N - 1))
        {
            dst[index] = value.x;
            dst[index + 1] = value.y;
        }
        else if (col < N)
        {
            dst[index] = value.x;
        }
    }
}
#else
ByteAddressBuffer src0 : register(t0);
ByteAddressBuffer src1 : register(t1);
RWByteAddressBuffer dst : register(u0);

float4 mm_readA(int row, int col) {
    if (row < M)
    {
        int index = row * K + col;
        float4 result = float4(asfloat(src0.Load(4 * index)),
            asfloat(src0.Load(4 * (index + 1))),
            asfloat(src0.Load(4 * (index + 2))),
            asfloat(src0.Load(4 * (index + 3))));
        return result;
    }
    else {
        return float4(0, 0, 0, 0);
    }
}

float3 mm_readA(int row, int col, float3 value) {
    if (row < M)
    {
        int index = row * K + col;
        float3 value = float3(asfloat(src0.Load(4 * index)),
            asfloat(src0.Load(4 * (index + 1))),
            asfloat(src0.Load(4 * (index + 2))));
        return value;
    }
    else { return float3(0.0, 0.0, 0.0); }
}

float2 mm_readA(int row, int col, float2 value) {
    if (row < M)
    {
        int index = row * K + col;
        float2 value = float2(asfloat(src0.Load(4 * index)),
            asfloat(src0.Load(4 * (index + 1))));
        return value;
    }
    else { return float2(0.0, 0.0); }
}

float mm_readA(int row, int col, float value) {
    if (row < M)
    {
        int index = row * K + col;
        return asfloat(src0.Load(4 * index));
    }
    else { return 0; }
}

float4 mm_readB(int row, int col) {
    int index = row * N + col;
    float4 result = float4(asfloat(src1.Load(4 * index)),
        asfloat(src1.Load(4 * (index + 1))),
        asfloat(src1.Load(4 * (index + 2))),
        asfloat(src1.Load(4 * (index + 3))));
    return result;
}

void mm_write(int row, int col, float4 value) {
    if (row < M && col < N)
    {
        int index = row * N + col;
        if (col < (N - 3)) {
            dst.Store(4 * (index), asuint(value.x));
            dst.Store(4 * (index + 1), asuint(value.y));
            dst.Store(4 * (index + 2), asuint(value.z));
            dst.Store(4 * (index + 3), asuint(value.w));
        }
        else if (col < (N - 2))
        {
            dst.Store(4 * (index), asuint(value.x));
            dst.Store(4 * (index + 1), asuint(value.y));
            dst.Store(4 * (index + 2), asuint(value.z));
        }
        else if (col < (N - 1))
        {
            dst.Store(4 * (index), asuint(value.x));
            dst.Store(4 * (index + 1), asuint(value.y));
        }
        else if (col < N)
        {
            dst.Store(4 * (index), asuint(value.x));
        }
    }
}
#endif  // USE_STRUCTURED_BUFFERS
#endif  // USE_TEXTURE

[numthreads(LOCAL_GROUP_SIZE_X, LOCAL_GROUP_SIZE_Y, 1)]
void main(CS_INPUT input)
{
    initGLBuiltins(input);

    int globalRow = int(gl_GlobalInvocationID.y) * WORK_PER_THREAD_Y;
    int globalCol = int(gl_GlobalInvocationID.x) * WORK_PER_THREAD_X;
    int RowPerThread = WORK_PER_THREAD_Y;
    float4 acc[WORK_PER_THREAD_Y];
    float4 ACached;
    float4 BCached[4];

    // Without this initialization strange values show up in acc.
    for (int innerRow = 0; innerRow < RowPerThread; innerRow++) {
        acc[innerRow] = float4(0.0, 0.0, 0.0, 0.0);
    }

    int sharedDimNearestVec4 = floor(K / 4);
    int sharedDimVec4Remainder = K % 4;
    for (int k = 0; k < sharedDimNearestVec4; k++) {
        BCached[0] = mm_readB(k * 4, globalCol);
        BCached[1] = mm_readB(k * 4 + 1, globalCol);
        BCached[2] = mm_readB(k * 4 + 2, globalCol);
        BCached[3] = mm_readB(k * 4 + 3, globalCol);

        for (int i = 0; i < RowPerThread; i++) {
            ACached = mm_readA(globalRow + i, k * 4);
            acc[i] = BCached[0] * ACached.x + acc[i];
            acc[i] = BCached[1] * ACached.y + acc[i];
            acc[i] = BCached[2] * ACached.z + acc[i];
            acc[i] = BCached[3] * ACached.w + acc[i];
        }
    }

    if (sharedDimVec4Remainder == 1) {
        BCached[0] = mm_readB(K - 1, globalCol);
        for (int i = 0; i < RowPerThread; i++) {
            ACached.x = mm_readA(globalRow + i, K - 1);
            acc[i] = BCached[0] * ACached.x + acc[i];
        }
    }
    else if (sharedDimVec4Remainder == 2) {
        BCached[0] = mm_readB(K - 2, globalCol);
        BCached[1] = mm_readB(K - 1, globalCol);
        for (int i = 0; i < RowPerThread; i++) {
            ACached.xy = mm_readA(globalRow + i, K - 2);
            acc[i] = BCached[0] * ACached.x + acc[i];
            acc[i] = BCached[1] * ACached.y + acc[i];
        }
    }
    else if (sharedDimVec4Remainder == 3) {
        BCached[0] = mm_readB(K - 3, globalCol);
        BCached[1] = mm_readB(K - 2, globalCol);
        BCached[2] = mm_readB(K - 1, globalCol);
        for (int i = 0; i < RowPerThread; i++) {
            ACached.xyz = mm_readA(globalRow + i, K - 3);
            acc[i] = BCached[0] * ACached.x + acc[i];
            acc[i] = BCached[1] * ACached.y + acc[i];
            acc[i] = BCached[2] * ACached.z + acc[i];
        }
    }

    for (int innerRow = 0; innerRow < RowPerThread; innerRow++) {
        mm_write(globalRow + innerRow,
            globalCol,
            acc[innerRow]);
    }
}
