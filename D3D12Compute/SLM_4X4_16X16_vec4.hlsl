cbuffer SceneConstantBuffer : register( b0 )
{
    int M;
    int K;
    int N;
    int TILE_K;
}

static uint3 gl_LocalInvocationID = uint3(0, 0, 0);
static uint3 gl_GlobalInvocationID = uint3(0, 0, 0);

struct CS_INPUT
{
    uint3 dx_LocalInvocationID : SV_GroupThreadID;
    uint3 dx_GlobalInvocationID : SV_DispatchThreadID;
};

void initGLBuiltins(CS_INPUT input)
{
    gl_LocalInvocationID = input.dx_LocalInvocationID;
    gl_GlobalInvocationID = input.dx_GlobalInvocationID;
};

#ifdef USE_TEXTURE
Texture2D<float4> src0 : register(t0);
Texture2D<float4> src1 : register(t1);
RWTexture2D<float4> dst : register(u0);

float4 mm_readA(int row, int col) {
    return src0.Load(int3(col, row, 0));
}

float4 mm_readB(int row, int col) {
    return src1.Load(int3(col, row, 0));
}

void mm_write(int row, int col, float4 value) {
    dst[uint2(col, row)] = value;
}
#else
#ifdef USE_STRUCTURED_BUFFERS
StructuredBuffer<float4> src0 : register(t0);
StructuredBuffer<float4> src1 : register(t1);
RWStructuredBuffer<float4> dst : register(u0);

float4 mm_readA(int row, int col) {
    if (row < M && col < K / 4)
    {
        return src0[row * (K / 4) + col];
    }
    else {
        return float4(0, 0, 0, 0);
    }
}

float4 mm_readB(int row, int col) {
    return src1[row * (N / 4) + col];
}

void mm_write(int row, int col, float4 value) {
    if (row < M && col < N / 4)
    {
        dst[row * (N / 4) + col] = value;
    }
}
#else
ByteAddressBuffer src0 : register(t0);
ByteAddressBuffer src1 : register(t1);
RWByteAddressBuffer dst : register(u0);

float4 mm_readA(int row, int col) {
    if (row < M && col < K / 4)
    {
        float4 result = asfloat(src0.Load4(16 * (row * (K / 4) + col)));
        return result;
    }
    else {
        return float4(0, 0, 0, 0);
    }
}

float4 mm_readB(int row, int col) {
    float4 result = asfloat(src1.Load4(16 * (row * (N / 4) + col)));
    return result;
}

void mm_write(int row, int col, float4 value) {
    if (row < M && col < N / 4)
    {
        dst.Store4(16 * (row * (N / 4) + col), asuint(value));
    }
}
#endif  // USE_STRUCTURED_BUFFERS
#endif  // USE_TEXTURE

static int RowPerThread = 4;
static int ColPerThread = 4;
static int TileInner = LOCAL_GROUP_SIZE_X * 4;
static int VEC_SIZE = 4;

groupshared float4 mm_Asub[LOCAL_GROUP_SIZE_Y * 4][LOCAL_GROUP_SIZE_X];
groupshared float4 mm_Bsub[LOCAL_GROUP_SIZE_Y * 4][LOCAL_GROUP_SIZE_X]; // LOCAL_GROUP_SIZE_X and LOCAL_GROUP_SIZE_Y must be same.

[numthreads(LOCAL_GROUP_SIZE_X, LOCAL_GROUP_SIZE_Y, 1)]
void main(CS_INPUT input)
{
    initGLBuiltins(input);
    int dimAOuter = M;
    int dimInner = K;
    int dimBOuter = N;
    int tileRow = int(gl_LocalInvocationID.y) * RowPerThread;
    int tileCol = int(gl_LocalInvocationID.x);

    int globalRow = int(gl_GlobalInvocationID.y) * RowPerThread;
    int globalCol = int(gl_GlobalInvocationID.x);

    int numTiles = (dimInner - 1) / TileInner + 1;

    float4 acc[4];
    float4 ACached;
    float4 BCached[4];

    // Without this initialization strange values show up in acc.
    for (int innerRow = 0; innerRow < RowPerThread; innerRow++) {
        acc[innerRow] = (float4)(0.f);
    }

    int globalColA = tileCol;
    int tileRowB = int(gl_LocalInvocationID.y) * 4;

    // Loop over shared dimension.
    for (int t = 0; t < numTiles; t++) {
      // Load one tile of A into local memory.
      for (int innerRow = 0; innerRow < 4; innerRow++) {
          int inputRow = tileRow + innerRow;
          int inputCol = tileCol;

          mm_Asub[inputRow][inputCol] = mm_readA(
              globalRow + innerRow,
              globalColA);
      }
      globalColA += TileInner / VEC_SIZE;

      // Load one tile of B into local memory.
      for (int innerRow = 0; innerRow < 4; innerRow++) {
          int inputRow = tileRowB + innerRow;
          int inputCol = tileCol;

          mm_Bsub[inputRow][inputCol] = mm_readB(
            t * TileInner + inputRow,
            globalCol);
      }

      GroupMemoryBarrierWithGroupSync();

      // Compute acc values for a single thread.
      for (int k = 0; k < TileInner / VEC_SIZE; k++) {
        BCached[0] = mm_Bsub[k * VEC_SIZE][tileCol];
        BCached[1] = mm_Bsub[k * VEC_SIZE + 1][tileCol];
        BCached[2] = mm_Bsub[k * VEC_SIZE + 2][tileCol];
        BCached[3] = mm_Bsub[k * VEC_SIZE + 3][tileCol];

        ACached = mm_Asub[tileRow][k];
        acc[0] = BCached[0] * ACached.x + acc[0];
        acc[0] = BCached[1] * ACached.y + acc[0];
        acc[0] = BCached[2] * ACached.z + acc[0];
        acc[0] = BCached[3] * ACached.w + acc[0];

        ACached = mm_Asub[tileRow + 1][k];
        acc[1] = BCached[0] * ACached.x + acc[1];
        acc[1] = BCached[1] * ACached.y + acc[1];
        acc[1] = BCached[2] * ACached.z + acc[1];
        acc[1] = BCached[3] * ACached.w + acc[1];

        ACached = mm_Asub[tileRow + 2][k];
        acc[2] = BCached[0] * ACached.x + acc[2];
        acc[2] = BCached[1] * ACached.y + acc[2];
        acc[2] = BCached[2] * ACached.z + acc[2];
        acc[2] = BCached[3] * ACached.w + acc[2];

        ACached = mm_Asub[tileRow + 3][k];
        acc[3] = BCached[0] * ACached.x + acc[3];
        acc[3] = BCached[1] * ACached.y + acc[3];
        acc[3] = BCached[2] * ACached.z + acc[3];
        acc[3] = BCached[3] * ACached.w + acc[3];
      }

      GroupMemoryBarrierWithGroupSync();
    }

    for (int innerRow = 0; innerRow < RowPerThread; innerRow++) {
          mm_write(globalRow + innerRow,
                   globalCol,
                   acc[innerRow]);
    }
}
