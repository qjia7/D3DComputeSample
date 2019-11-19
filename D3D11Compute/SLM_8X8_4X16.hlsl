cbuffer SceneConstantBuffer : register( b0 )
{
    int M;
    int K;
    int N;
    int TILE_K;
}

static uint3 gl_WorkGroupID = uint3(0, 0, 0);
static uint3 gl_LocalInvocationID = uint3(0, 0, 0);

struct CS_INPUT
{
    uint3 dx_WorkGroupID : SV_GroupID;
    uint3 dx_LocalInvocationID : SV_GroupThreadID;
};

void initGLBuiltins(CS_INPUT input)
{
    gl_WorkGroupID = input.dx_WorkGroupID;
    gl_LocalInvocationID = input.dx_LocalInvocationID;
};

StructuredBuffer<float4> src0 : register(t0);
StructuredBuffer<float4> src1 : register(t1);
RWStructuredBuffer<float4> dst : register(u0);

static int VEC_SIZE = 4;
static int TILE_M = 32;
static int TILE_N = 128;
static int ROWS_PER_WI = 8;
static int TILE_K0 = 64;

groupshared float4 atile[512];
[numthreads(16, 4, 1)]
void CSMain(CS_INPUT input)
{
    initGLBuiltins(input);
    int width0 = K / VEC_SIZE;
    int width1 = N / VEC_SIZE;

    int group_x = int(gl_WorkGroupID.x);
    int group_y = int(gl_WorkGroupID.y);
    int local_x = int(gl_LocalInvocationID.x);
    int local_y = int(gl_LocalInvocationID.y);

	// Result ctile is M rows x N columns
    // M = 32, we have 4 rows of work-items, so we need 32/4 8 results down
    // N = 128, we have 16 columns of work-items, so we need 128/16 = 8 results across = 2 float4s across

    float4 dot00 = {0, 0, 0, 0};
    float4 dot01 = {0, 0, 0, 0};
    float4 dot02 = {0, 0, 0, 0};
    float4 dot03 = {0, 0, 0, 0};
    float4 dot04 = {0, 0, 0, 0};
    float4 dot05 = {0, 0, 0, 0};
    float4 dot06 = {0, 0, 0, 0};
    float4 dot07 = {0, 0, 0, 0};
    float4 dot10 = {0, 0, 0, 0};
    float4 dot11 = {0, 0, 0, 0};
    float4 dot12 = {0, 0, 0, 0};
    float4 dot13 = {0, 0, 0, 0};
    float4 dot14 = {0, 0, 0, 0};
    float4 dot15 = {0, 0, 0, 0};
    float4 dot16 = {0, 0, 0, 0};
    float4 dot17 = {0, 0, 0, 0};

    int dst_write0 = local_x + ( group_x * ( TILE_N / VEC_SIZE ) ) + ( ( group_y * TILE_M ) + ROWS_PER_WI * local_y ) * width1;

    // Src0 is used to load atile.
    // It starts at the left side of src0 and walks across.
    // atile is M rows x K columns.
    int src0_read = local_x + ( ( group_y * TILE_M ) + ROWS_PER_WI * local_y ) * width0;

    // Src1 is directly used as btile.
    // It starts at the top of src1 and walks down.
    // btile is K rows x N columns.
    // K = 64, we'll process four rows at a time
    // N = 128, we have 16 columns of work-items, so we need 128/16 = 8 floats across = 2 float4s across
    int src1_read0 = local_x + ( group_x * ( TILE_N / VEC_SIZE ) );
    int src1_read1 = src1_read0 + ( TILE_N / 2 / VEC_SIZE );

    int slm = local_y * ( ROWS_PER_WI * TILE_K0 / VEC_SIZE );

    // Walk ACROSS src0 and DOWN src1:
    int w = 0;
    do{
      // We want to load atile, which is M rows x K columns
      // M = 32, and we have 4 rows of work-items, so each work-item must load 32/4 = 8 rows.
      // K = 64, and we have 16 columns of work-items, so each work-item must load 64/16 = 4 columns = 1 float4.
      atile[slm + local_x + 0 * TILE_K0 / VEC_SIZE] = src0[src0_read + 0 * width0];
      atile[slm + local_x + 1 * TILE_K0 / VEC_SIZE] = src0[src0_read + 1 * width0];
      atile[slm + local_x + 2 * TILE_K0 / VEC_SIZE] = src0[src0_read + 2 * width0];
      atile[slm + local_x + 3 * TILE_K0 / VEC_SIZE] = src0[src0_read + 3 * width0];
      atile[slm + local_x + 4 * TILE_K0 / VEC_SIZE] = src0[src0_read + 4 * width0];
      atile[slm + local_x + 5 * TILE_K0 / VEC_SIZE] = src0[src0_read + 5 * width0];
      atile[slm + local_x + 6 * TILE_K0 / VEC_SIZE] = src0[src0_read + 6 * width0];
      atile[slm + local_x + 7 * TILE_K0 / VEC_SIZE] = src0[src0_read + 7 * width0];

      src0_read += TILE_K0 / VEC_SIZE;

      GroupMemoryBarrierWithGroupSync();

      int i = 0;
      do{
          // We get better performance by loading btile first.
          float4 brow00 = src1[src1_read0];   src1_read0 += width1;
          float4 brow01 = src1[src1_read0];   src1_read0 += width1;
          float4 brow02 = src1[src1_read0];   src1_read0 += width1;
          float4 brow03 = src1[src1_read0];   src1_read0 += width1;
          float4 brow10 = src1[src1_read1];   src1_read1 += width1;
          float4 brow11 = src1[src1_read1];   src1_read1 += width1;
          float4 brow12 = src1[src1_read1];   src1_read1 += width1;
          float4 brow13 = src1[src1_read1];   src1_read1 += width1;

          float4 a0 = atile[slm + i + 0 * TILE_K0 / VEC_SIZE ];
          dot00 = brow00*a0.x + dot00;
          dot00 = brow01*a0.y + dot00;
          dot00 = brow02*a0.z + dot00;
          dot00 = brow03*a0.w + dot00;
          dot10 = brow10*a0.x + dot10;
          dot10 = brow11*a0.y + dot10;
          dot10 = brow12*a0.z + dot10;
          dot10 = brow13*a0.w + dot10;

          float4 a1 = atile[slm + i + 1 * TILE_K0 / VEC_SIZE ];
          dot01 = brow00*a1.x + dot01;
          dot01 = brow01*a1.y + dot01;
          dot01 = brow02*a1.z + dot01;
          dot01 = brow03*a1.w + dot01;
          dot11 = brow10*a1.x + dot11;
          dot11 = brow11*a1.y + dot11;
          dot11 = brow12*a1.z + dot11;
          dot11 = brow13*a1.w + dot11;

          float4 a2 = atile[slm + i + 2 * TILE_K0 / VEC_SIZE ];
          dot02 = brow00*a2.x + dot02;
          dot02 = brow01*a2.y + dot02;
          dot02 = brow02*a2.z + dot02;
          dot02 = brow03*a2.w + dot02;
          dot12 = brow10*a2.x + dot12;
          dot12 = brow11*a2.y + dot12;
          dot12 = brow12*a2.z + dot12;
          dot12 = brow13*a2.w + dot12;

          float4 a3 = atile[slm + i + 3 * TILE_K0 / VEC_SIZE ];
          dot03 = brow00*a3.x + dot03;
          dot03 = brow01*a3.y + dot03;
          dot03 = brow02*a3.z + dot03;
          dot03 = brow03*a3.w + dot03;
          dot13 = brow10*a3.x + dot13;
          dot13 = brow11*a3.y + dot13;
          dot13 = brow12*a3.z + dot13;
          dot13 = brow13*a3.w + dot13;

          float4 a4 = atile[slm + i + 4 * TILE_K0 / VEC_SIZE ];
          dot04 = brow00*a4.x + dot04;
          dot04 = brow01*a4.y + dot04;
          dot04 = brow02*a4.z + dot04;
          dot04 = brow03*a4.w + dot04;
          dot14 = brow10*a4.x + dot14;
          dot14 = brow11*a4.y + dot14;
          dot14 = brow12*a4.z + dot14;
          dot14 = brow13*a4.w + dot14;

          float4 a5 = atile[slm + i + 5 * TILE_K0 / VEC_SIZE ];
          dot05 = brow00*a5.x + dot05;
          dot05 = brow01*a5.y + dot05;
          dot05 = brow02*a5.z + dot05;
          dot05 = brow03*a5.w + dot05;
          dot15 = brow10*a5.x + dot15;
          dot15 = brow11*a5.y + dot15;
          dot15 = brow12*a5.z + dot15;
          dot15 = brow13*a5.w + dot15;

          float4 a6 = atile[slm + i + 6 * TILE_K0 / VEC_SIZE ];
          dot06 = brow00*a6.x + dot06;
          dot06 = brow01*a6.y + dot06;
          dot06 = brow02*a6.z + dot06;
          dot06 = brow03*a6.w + dot06;
          dot16 = brow10*a6.x + dot16;
          dot16 = brow11*a6.y + dot16;
          dot16 = brow12*a6.z + dot16;
          dot16 = brow13*a6.w + dot16;

          float4 a7 = atile[slm + i + 7 * TILE_K0 / VEC_SIZE ];
          dot07 = brow00*a7.x + dot07;
          dot07 = brow01*a7.y + dot07;
          dot07 = brow02*a7.z + dot07;
          dot07 = brow03*a7.w + dot07;
          dot17 = brow10*a7.x + dot17;
          dot17 = brow11*a7.y + dot17;
          dot17 = brow12*a7.z + dot17;
          dot17 = brow13*a7.w + dot17;

          i++;
      }
      while( i < TILE_K0 / VEC_SIZE );

      GroupMemoryBarrierWithGroupSync();

      w += TILE_K0 / VEC_SIZE;
    }
    while( w < width0 );

    int dst_write1 = dst_write0 + ( TILE_N / 2 / VEC_SIZE );

    dst[dst_write0] = dot00;  dst_write0 += width1;
    dst[dst_write0] = dot01;  dst_write0 += width1;
    dst[dst_write0] = dot02;  dst_write0 += width1;
    dst[dst_write0] = dot03;  dst_write0 += width1;
    dst[dst_write0] = dot04;  dst_write0 += width1;
    dst[dst_write0] = dot05;  dst_write0 += width1;
    dst[dst_write0] = dot06;  dst_write0 += width1;
    dst[dst_write0] = dot07;  dst_write0 += width1;

    dst[dst_write1] = dot10;  dst_write1 += width1;
    dst[dst_write1] = dot11;  dst_write1 += width1;
    dst[dst_write1] = dot12;  dst_write1 += width1;
    dst[dst_write1] = dot13;  dst_write1 += width1;
    dst[dst_write1] = dot14;  dst_write1 += width1;
    dst[dst_write1] = dot15;  dst_write1 += width1;
    dst[dst_write1] = dot16;  dst_write1 += width1;
    dst[dst_write1] = dot17;  dst_write1 += width1;
}
