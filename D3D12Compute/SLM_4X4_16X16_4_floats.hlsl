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

#ifdef USE_STRUCTURED_BUFFERS
StructuredBuffer<float> src0 : register(t0);
StructuredBuffer<float> src1 : register(t1);
RWStructuredBuffer<float> dst : register(u0);

float4 mm_readA(int row, int col) {
    if (row < M && col < K)
    {
        int index = row * K + col;
        float4 result = float4(src0[index],
            src0[index + 1], src0[index + 2], src0[index + 3]);
        return result;
    }
    else {
        return float4(0, 0, 0, 0);
    }
}

float4 mm_readB(int row, int col) {
    int index = row * N + col;
    float4 result = float4(src1[index],
        src1[index + 1], src1[index + 2], src1[index + 3]);
    return result;
}

void mm_write(int row, int col, float4 value) {
    if (row < M && col < N)
    {
        int index = row * N + col;
        dst[index] = value.x;
        dst[index + 1] = value.y;
        dst[index + 2] = value.z;
        dst[index + 3] = value.w;
    }
}
#else
ByteAddressBuffer src0 : register(t0);
ByteAddressBuffer src1 : register(t1);
RWByteAddressBuffer dst : register(u0);

float4 mm_readA(int row, int col) {
    if (row < M && col < K)
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
        dst.Store(4 * (index), asuint(value.x));
        dst.Store(4 * (index + 1), asuint(value.y));
        dst.Store(4 * (index + 2), asuint(value.z));
        dst.Store(4 * (index + 3), asuint(value.w));
    }
}
#endif  // USE_STRUCTURED_BUFFERS

static int RowPerThread = 4;
static int ColPerThread = 4;
static int TileInner = LOCAL_GROUP_SIZE_X * 4;
static int VEC_SIZE = 4;

groupshared float mm_Asub[LOCAL_GROUP_SIZE_Y * 4][LOCAL_GROUP_SIZE_X * 4];
groupshared float mm_Bsub[LOCAL_GROUP_SIZE_Y * 4][LOCAL_GROUP_SIZE_X * 4]; // LOCAL_GROUP_SIZE_X and LOCAL_GROUP_SIZE_Y must be same.

[numthreads(LOCAL_GROUP_SIZE_X, LOCAL_GROUP_SIZE_Y, 1)]
void main(CS_INPUT input)
{
    initGLBuiltins(input);
    int dimAOuter = M;
    int dimInner = K;
    int dimBOuter = N;
    int tileRow = int(gl_LocalInvocationID.y) * RowPerThread;
    int tileCol = int(gl_LocalInvocationID.x) * ColPerThread;

    int globalRow = int(gl_GlobalInvocationID.y) * RowPerThread;
    int globalCol = int(gl_GlobalInvocationID.x) * ColPerThread;

    int numTiles = (dimInner - 1) / TileInner + 1;

    float acc[4][4];
    float ACached;
    float BCached[4];

    // Without this initialization strange values show up in acc.
    for (int innerRow = 0; innerRow < RowPerThread; innerRow++) {
        for (int innerCol = 0; innerCol < ColPerThread; innerCol++) {
            acc[innerRow][innerCol] = 0.0;
        }
    }

    int tileRowB = int(gl_LocalInvocationID.y) * 4;

    // Loop over shared dimension.
    for (int t = 0; t < numTiles; t++) {
      // Load one tile of A into local memory.
      for (int innerRow = 0; innerRow < 4; innerRow++) {
          int inputRow = tileRow + innerRow;
          int inputCol = tileCol;

          float4 result = mm_readA(
              globalRow + innerRow,
              t * TileInner + inputCol);
          mm_Asub[inputRow][inputCol] = result.x;
          mm_Asub[inputRow][inputCol + 1] = result.y;
          mm_Asub[inputRow][inputCol + 2] = result.z;
          mm_Asub[inputRow][inputCol + 3] = result.w;
      }

      // Load one tile of B into local memory.
      for (int innerRow = 0; innerRow < 4; innerRow++) {
          int inputRow = tileRowB + innerRow;
          int inputCol = tileCol;

          float4 result = mm_readB(
              t * TileInner + inputRow,
              globalCol);
          mm_Bsub[inputRow][inputCol] = result.x;
          mm_Bsub[inputRow][inputCol + 1] = result.y;
          mm_Bsub[inputRow][inputCol + 2] = result.z;
          mm_Bsub[inputRow][inputCol + 3] = result.w;
      }

      GroupMemoryBarrierWithGroupSync();

      // Compute acc values for a single thread.
      for (int k = 0; k < TileInner; k++) {
          for (int inner = 0; inner < ColPerThread; inner++) {
              BCached[inner] = mm_Bsub[k][tileCol + inner];
          }

          for (int innerRow = 0; innerRow < RowPerThread; innerRow++) {
              ACached = mm_Asub[tileRow + innerRow][k];
              for (int innerCol = 0; innerCol < ColPerThread; innerCol++) {
                  acc[innerRow][innerCol] += ACached * BCached[innerCol];
              }
          }
      }

      GroupMemoryBarrierWithGroupSync();
    }

    for (int innerRow = 0; innerRow < RowPerThread; innerRow++) {
        float4 result = float4(acc[innerRow][0], acc[innerRow][1],
                               acc[innerRow][2], acc[innerRow][3]);
          mm_write(globalRow + innerRow,
                   globalCol,
                   result);
    }
}
