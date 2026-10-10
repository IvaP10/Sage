#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <CoreFoundation/CoreFoundation.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define SAGE_Q4_BATCH_TILE_MAX 4U

@interface SageMetalContext : NSObject
@property(nonatomic, strong) id<MTLDevice> device;
@property(nonatomic, strong) id<MTLCommandQueue> queue;
@property(nonatomic, strong) id<MTLComputePipelineState> q4Pipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> q4Rows4Pipeline;
@property(nonatomic) NSUInteger maxThreadgroupMemoryLength;
@end

@implementation SageMetalContext
@end

@interface SageMetalBuffer : NSObject
@property(nonatomic, strong) id<MTLBuffer> buffer;
@end

@implementation SageMetalBuffer
@end

typedef struct {
    uint32_t rows;
    uint32_t columns;
    uint32_t groupSize;
    uint32_t batchSize;
    uint32_t batchTileSize;
    uint32_t threadsPerGroup;
} SageQ4Params;

static void sage_write_error(char *output, size_t capacity, NSString *message) {
    if (output == NULL || capacity == 0) {
        return;
    }
    const char *utf8 = message.UTF8String;
    if (utf8 == NULL) {
        utf8 = "Metal operation failed";
    }
    snprintf(output, capacity, "%s", utf8);
}

void *sage_metal_context_create(char *error, size_t capacity) {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (device == nil) {
            sage_write_error(error, capacity, @"No Metal device is available");
            return NULL;
        }

        NSString *source = @"#include <metal_stdlib>\n"
        "using namespace metal;\n"
        "struct SageQ4Params { uint rows; uint columns; uint groupSize; uint batchSize; uint batchTileSize; uint threadsPerGroup; };\n"
        "kernel void sage_q4_project(device const uchar *packed [[buffer(0)]],\n"
        " device const float *scales [[buffer(1)]],\n"
        " device const float *input [[buffer(2)]],\n"
        " device float *output [[buffer(3)]],\n"
        " constant SageQ4Params &p [[buffer(4)]],\n"
        " uint2 group [[threadgroup_position_in_grid]],\n"
        " uint lane [[thread_index_in_threadgroup]]) {\n"
        " uint row = group.x;\n"
        " uint firstBatch = group.y * p.batchTileSize;\n"
        " if (row >= p.rows || firstBatch >= p.batchSize) return;\n"
        " ulong base = ulong(row) * ulong(p.columns);\n"
        " float sum0 = 0.0f; float sum1 = 0.0f; float sum2 = 0.0f; float sum3 = 0.0f;\n"
        " ulong inputBase0 = ulong(firstBatch) * ulong(p.columns);\n"
        " ulong inputBase1 = ulong(firstBatch + 1) * ulong(p.columns);\n"
        " ulong inputBase2 = ulong(firstBatch + 2) * ulong(p.columns);\n"
        " ulong inputBase3 = ulong(firstBatch + 3) * ulong(p.columns);\n"
        " if (p.groupSize == 128 && (p.columns & 127) == 0) {\n"
        "  ulong rowScaleBase = ulong(row) * ulong(p.columns >> 7);\n"
        "  for (uint pairIndex = lane; pairIndex < p.columns / 2; pairIndex += p.threadsPerGroup) {\n"
        "   uint column = pairIndex * 2;\n"
        "   ulong index = base + ulong(column);\n"
        "   uchar pair = packed[index >> 1];\n"
        "   float scale = scales[rowScaleBase + ulong(column >> 7)];\n"
        "   float lowWeight = float(int(pair & 15) - 8) * scale;\n"
        "   float highWeight = float(int(pair >> 4) - 8) * scale;\n"
        "   sum0 = fma(lowWeight, input[inputBase0 + ulong(column)], sum0);\n"
        "   sum0 = fma(highWeight, input[inputBase0 + ulong(column + 1)], sum0);\n"
        "   if (p.batchTileSize > 1 && firstBatch + 1 < p.batchSize) { sum1 = fma(lowWeight, input[inputBase1 + ulong(column)], sum1); sum1 = fma(highWeight, input[inputBase1 + ulong(column + 1)], sum1); }\n"
        "   if (p.batchTileSize > 2 && firstBatch + 2 < p.batchSize) { sum2 = fma(lowWeight, input[inputBase2 + ulong(column)], sum2); sum2 = fma(highWeight, input[inputBase2 + ulong(column + 1)], sum2); }\n"
        "   if (p.batchTileSize > 3 && firstBatch + 3 < p.batchSize) { sum3 = fma(lowWeight, input[inputBase3 + ulong(column)], sum3); sum3 = fma(highWeight, input[inputBase3 + ulong(column + 1)], sum3); }\n"
        "  }\n"
        " } else if ((p.columns & 1) == 0) {\n"
        "  for (uint pairIndex = lane; pairIndex < p.columns / 2; pairIndex += p.threadsPerGroup) {\n"
        "   uint column = pairIndex * 2;\n"
        "   ulong index = base + ulong(column);\n"
        "   uchar pair = packed[index >> 1];\n"
        "   int low = int(pair & 15) - 8;\n"
        "   int high = int(pair >> 4) - 8;\n"
        "   float lowScale = scales[index / ulong(p.groupSize)];\n"
        "   float highScale = scales[(index + 1) / ulong(p.groupSize)];\n"
        "   float lowWeight = float(low) * lowScale;\n"
        "   float highWeight = float(high) * highScale;\n"
        "   sum0 = fma(lowWeight, input[inputBase0 + ulong(column)], sum0);\n"
        "   sum0 = fma(highWeight, input[inputBase0 + ulong(column + 1)], sum0);\n"
        "   if (p.batchTileSize > 1 && firstBatch + 1 < p.batchSize) { sum1 = fma(lowWeight, input[inputBase1 + ulong(column)], sum1); sum1 = fma(highWeight, input[inputBase1 + ulong(column + 1)], sum1); }\n"
        "   if (p.batchTileSize > 2 && firstBatch + 2 < p.batchSize) { sum2 = fma(lowWeight, input[inputBase2 + ulong(column)], sum2); sum2 = fma(highWeight, input[inputBase2 + ulong(column + 1)], sum2); }\n"
        "   if (p.batchTileSize > 3 && firstBatch + 3 < p.batchSize) { sum3 = fma(lowWeight, input[inputBase3 + ulong(column)], sum3); sum3 = fma(highWeight, input[inputBase3 + ulong(column + 1)], sum3); }\n"
        "  }\n"
        " } else {\n"
        "  for (uint column = lane; column < p.columns; column += p.threadsPerGroup) {\n"
        "   ulong index = base + ulong(column);\n"
        "   uchar pair = packed[index >> 1];\n"
        "   int q = int((index & 1) == 0 ? (pair & 15) : (pair >> 4)) - 8;\n"
        "   float weight = float(q) * scales[index / ulong(p.groupSize)];\n"
        "   sum0 = fma(weight, input[inputBase0 + ulong(column)], sum0);\n"
        "   if (p.batchTileSize > 1 && firstBatch + 1 < p.batchSize) sum1 = fma(weight, input[inputBase1 + ulong(column)], sum1);\n"
        "   if (p.batchTileSize > 2 && firstBatch + 2 < p.batchSize) sum2 = fma(weight, input[inputBase2 + ulong(column)], sum2);\n"
        "   if (p.batchTileSize > 3 && firstBatch + 3 < p.batchSize) sum3 = fma(weight, input[inputBase3 + ulong(column)], sum3);\n"
        "  }\n"
        " }\n"
        " float total0 = simd_sum(sum0); float total1 = simd_sum(sum1); float total2 = simd_sum(sum2); float total3 = simd_sum(sum3);\n"
        " if (lane == 0) {\n"
        "  output[ulong(firstBatch) * ulong(p.rows) + ulong(row)] = total0;\n"
        "  if (p.batchTileSize > 1 && firstBatch + 1 < p.batchSize) output[ulong(firstBatch + 1) * ulong(p.rows) + ulong(row)] = total1;\n"
        "  if (p.batchTileSize > 2 && firstBatch + 2 < p.batchSize) output[ulong(firstBatch + 2) * ulong(p.rows) + ulong(row)] = total2;\n"
        "  if (p.batchTileSize > 3 && firstBatch + 3 < p.batchSize) output[ulong(firstBatch + 3) * ulong(p.rows) + ulong(row)] = total3;\n"
        " }\n"
        "}\n"
        "kernel void sage_q4_project_rows4(device const uchar *packed [[buffer(0)]],\n"
        " device const float *scales [[buffer(1)]],\n"
        " device const float *input [[buffer(2)]],\n"
        " device float *output [[buffer(3)]],\n"
        " constant SageQ4Params &p [[buffer(4)]],\n"
        " threadgroup float *inputTile [[threadgroup(0)]],\n"
        " uint rowTile [[threadgroup_position_in_grid]],\n"
        " uint threadIndex [[thread_index_in_threadgroup]]) {\n"
        " uint threadCount = p.threadsPerGroup * 4;\n"
        " for (uint column = threadIndex; column < p.columns; column += threadCount) inputTile[column] = input[column];\n"
        " threadgroup_barrier(mem_flags::mem_threadgroup);\n"
        " uint rowInTile = threadIndex / p.threadsPerGroup;\n"
        " uint lane = threadIndex % p.threadsPerGroup;\n"
        " uint row = rowTile * 4 + rowInTile;\n"
        " if (row < p.rows) {\n"
        "  ulong base = ulong(row) * ulong(p.columns);\n"
        "  float sum = 0.0f;\n"
        "  if (p.groupSize == 128 && (p.columns & 127) == 0) {\n"
        "   ulong rowScaleBase = ulong(row) * ulong(p.columns >> 7);\n"
        "   for (uint pairIndex = lane; pairIndex < p.columns / 2; pairIndex += p.threadsPerGroup) {\n"
        "    uint column = pairIndex * 2;\n"
        "    ulong index = base + ulong(column);\n"
        "    uchar pair = packed[index >> 1];\n"
        "    float scale = scales[rowScaleBase + ulong(column >> 7)];\n"
        "    sum = fma(float(int(pair & 15) - 8) * scale, inputTile[column], sum);\n"
        "    sum = fma(float(int(pair >> 4) - 8) * scale, inputTile[column + 1], sum);\n"
        "   }\n"
        "  } else if ((p.columns & 1) == 0) {\n"
        "   for (uint pairIndex = lane; pairIndex < p.columns / 2; pairIndex += p.threadsPerGroup) {\n"
        "    uint column = pairIndex * 2;\n"
        "    ulong index = base + ulong(column);\n"
        "    uchar pair = packed[index >> 1];\n"
        "    int low = int(pair & 15) - 8;\n"
        "    int high = int(pair >> 4) - 8;\n"
        "    float lowScale = scales[index / ulong(p.groupSize)];\n"
        "    float highScale = scales[(index + 1) / ulong(p.groupSize)];\n"
        "    sum = fma(float(low) * lowScale, inputTile[column], sum);\n"
        "    sum = fma(float(high) * highScale, inputTile[column + 1], sum);\n"
        "   }\n"
        "  } else {\n"
        "   for (uint column = lane; column < p.columns; column += p.threadsPerGroup) {\n"
        "    ulong index = base + ulong(column);\n"
        "    uchar pair = packed[index >> 1];\n"
        "    int q = int((index & 1) == 0 ? (pair & 15) : (pair >> 4)) - 8;\n"
        "    float weight = float(q) * scales[index / ulong(p.groupSize)];\n"
        "    sum = fma(weight, inputTile[column], sum);\n"
        "   }\n"
        "  }\n"
        "  float total = simd_sum(sum);\n"
        "  if (lane == 0) output[row] = total;\n"
        " }\n"
        "}\n";

        NSError *compileError = nil;
        id<MTLLibrary> library = [device newLibraryWithSource:source options:nil error:&compileError];
        if (library == nil) {
            sage_write_error(error, capacity, compileError.localizedDescription ?: @"Metal Q4 source compilation failed");
            return NULL;
        }
        id<MTLFunction> function = [library newFunctionWithName:@"sage_q4_project"];
        if (function == nil) {
            sage_write_error(error, capacity, @"Sage Q4 Metal function is missing");
            return NULL;
        }
        id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:function error:&compileError];
        if (pipeline == nil) {
            sage_write_error(error, capacity, compileError.localizedDescription ?: @"Metal Q4 pipeline creation failed");
            return NULL;
        }
        id<MTLFunction> rows4Function = [library newFunctionWithName:@"sage_q4_project_rows4"];
        if (rows4Function == nil) {
            sage_write_error(error, capacity, @"Sage tiled Q4 Metal function is missing");
            return NULL;
        }
        id<MTLComputePipelineState> rows4Pipeline = [device newComputePipelineStateWithFunction:rows4Function error:&compileError];
        if (rows4Pipeline == nil) {
            sage_write_error(error, capacity, compileError.localizedDescription ?: @"Metal tiled Q4 pipeline creation failed");
            return NULL;
        }
        id<MTLCommandQueue> queue = [device newCommandQueue];
        if (queue == nil) {
            sage_write_error(error, capacity, @"Metal command queue creation failed");
            return NULL;
        }

        SageMetalContext *context = [SageMetalContext new];
        context.device = device;
        context.queue = queue;
        context.q4Pipeline = pipeline;
        context.q4Rows4Pipeline = rows4Pipeline;
        context.maxThreadgroupMemoryLength = device.maxThreadgroupMemoryLength;
        return (__bridge_retained void *)context;
    }
}

void sage_metal_context_release(void *raw) {
    if (raw != NULL) {
        CFRelease(raw);
    }
}

void *sage_metal_buffer_create(void *rawContext, const uint8_t *bytes, size_t length, char *error, size_t capacity) {
    @autoreleasepool {
        if (rawContext == NULL || bytes == NULL || length == 0) {
            sage_write_error(error, capacity, @"Metal buffer input is empty");
            return NULL;
        }
        SageMetalContext *context = (__bridge SageMetalContext *)rawContext;
        id<MTLBuffer> buffer = [context.device newBufferWithBytes:bytes length:length options:MTLResourceStorageModeShared];
        if (buffer == nil || buffer.length != length) {
            sage_write_error(error, capacity, @"Metal shared-storage weight buffer allocation failed");
            return NULL;
        }
        SageMetalBuffer *holder = [SageMetalBuffer new];
        holder.buffer = buffer;
        return (__bridge_retained void *)holder;
    }
}

void *sage_metal_buffer_create_private(void *rawContext, const uint8_t *bytes, size_t length, char *error, size_t capacity) {
    @autoreleasepool {
        if (rawContext == NULL || bytes == NULL || length == 0) {
            sage_write_error(error, capacity, @"Metal private-buffer input is empty");
            return NULL;
        }
        SageMetalContext *context = (__bridge SageMetalContext *)rawContext;
        id<MTLBuffer> staging = [context.device newBufferWithBytes:bytes length:length options:MTLResourceStorageModeShared];
        id<MTLBuffer> buffer = [context.device newBufferWithLength:length options:MTLResourceStorageModePrivate];
        if (staging == nil || staging.length != length || staging.contents == NULL || buffer == nil || buffer.length != length) {
            if (staging.contents != NULL) {
                memset(staging.contents, 0, length);
            }
            sage_write_error(error, capacity, @"Metal private weight-buffer allocation failed");
            return NULL;
        }

        id<MTLCommandBuffer> command = [context.queue commandBuffer];
        id<MTLBlitCommandEncoder> encoder = [command blitCommandEncoder];
        if (command == nil || encoder == nil) {
            memset(staging.contents, 0, length);
            sage_write_error(error, capacity, @"Metal private weight upload could not be encoded");
            return NULL;
        }
        [encoder copyFromBuffer:staging sourceOffset:0 toBuffer:buffer destinationOffset:0 size:length];
        [encoder endEncoding];
        [command commit];
        [command waitUntilCompleted];
        memset(staging.contents, 0, length);
        if (command.status != MTLCommandBufferStatusCompleted) {
            sage_write_error(error, capacity, command.error.localizedDescription ?: @"Metal private weight upload did not complete");
            return NULL;
        }

        SageMetalBuffer *holder = [SageMetalBuffer new];
        holder.buffer = buffer;
        return (__bridge_retained void *)holder;
    }
}

void *sage_metal_buffer_allocate(void *rawContext, size_t length, char *error, size_t capacity) {
    @autoreleasepool {
        if (rawContext == NULL || length == 0) {
            sage_write_error(error, capacity, @"Metal buffer allocation arguments are invalid");
            return NULL;
        }
        SageMetalContext *context = (__bridge SageMetalContext *)rawContext;
        id<MTLBuffer> buffer = [context.device newBufferWithLength:length options:MTLResourceStorageModeShared];
        if (buffer == nil || buffer.length != length || buffer.contents == NULL) {
            sage_write_error(error, capacity, @"Metal shared-storage workspace allocation failed");
            return NULL;
        }
        memset(buffer.contents, 0, length);
        SageMetalBuffer *holder = [SageMetalBuffer new];
        holder.buffer = buffer;
        return (__bridge_retained void *)holder;
    }
}

void sage_metal_buffer_release(void *raw) {
    if (raw != NULL) {
        CFRelease(raw);
    }
}

const uint8_t *sage_metal_buffer_contents(void *raw) {
    if (raw == NULL) {
        return NULL;
    }
    SageMetalBuffer *holder = (__bridge SageMetalBuffer *)raw;
    return (const uint8_t *)holder.buffer.contents;
}

uint8_t *sage_metal_buffer_contents_mut(void *raw) {
    if (raw == NULL) {
        return NULL;
    }
    SageMetalBuffer *holder = (__bridge SageMetalBuffer *)raw;
    return (uint8_t *)holder.buffer.contents;
}

size_t sage_metal_buffer_length(void *raw) {
    if (raw == NULL) {
        return 0;
    }
    SageMetalBuffer *holder = (__bridge SageMetalBuffer *)raw;
    return holder.buffer.length;
}

int sage_metal_q4_project(
    void *rawContext,
    void *rawWeights,
    void *rawScales,
    void *rawInput,
    void *rawOutput,
    uint32_t rows,
    uint32_t columns,
    uint32_t groupSize,
    uint32_t batchSize,
    uint32_t batchTileSize,
    uint32_t rowsPerThreadgroup,
    uint64_t *gpuDurationNs,
    char *error,
    size_t capacity
) {
    @autoreleasepool {
        if (rawContext == NULL || rawWeights == NULL || rawScales == NULL || rawInput == NULL || rawOutput == NULL || rows == 0 || columns == 0 || groupSize == 0 || batchSize == 0 || batchSize > 256 || (batchTileSize != 1U && batchTileSize != 2U && batchTileSize != SAGE_Q4_BATCH_TILE_MAX) || (rowsPerThreadgroup != 1 && rowsPerThreadgroup != 4)) {
            sage_write_error(error, capacity, @"Metal Q4 projection arguments are invalid");
            return -1;
        }
        SageMetalContext *context = (__bridge SageMetalContext *)rawContext;
        SageMetalBuffer *weights = (__bridge SageMetalBuffer *)rawWeights;
        SageMetalBuffer *scales = (__bridge SageMetalBuffer *)rawScales;
        SageMetalBuffer *input = (__bridge SageMetalBuffer *)rawInput;
        SageMetalBuffer *output = (__bridge SageMetalBuffer *)rawOutput;
        uint64_t elements = (uint64_t)rows * (uint64_t)columns;
        uint64_t inputElements = (uint64_t)batchSize * (uint64_t)columns;
        uint64_t outputElements = (uint64_t)batchSize * (uint64_t)rows;
        if (elements > 700000000ULL) {
            sage_write_error(error, capacity, @"Metal Q4 projection exceeds Sage's element bound");
            return -1;
        }
        if (rows > 1048576U || inputElements > 67108864ULL || outputElements > 67108864ULL) {
            sage_write_error(error, capacity, @"Metal Q4 batch exceeds Sage's bounded I/O limit");
            return -1;
        }
        uint64_t requiredWeightBytes = (elements + 1) / 2;
        uint64_t scaleCount = (elements + (uint64_t)groupSize - 1) / (uint64_t)groupSize;
        if (requiredWeightBytes != weights.buffer.length
            || scaleCount * sizeof(float) != scales.buffer.length
            || scaleCount > UINT32_MAX
            || input.buffer.length != (NSUInteger)batchSize * (NSUInteger)columns * sizeof(float)
            || output.buffer.length != (NSUInteger)batchSize * (NSUInteger)rows * sizeof(float)
            || input.buffer.contents == NULL
            || output.buffer.contents == NULL) {
            sage_write_error(error, capacity, @"Metal Q4 buffers do not match the declared matrix");
            return -1;
        }

        uint64_t inputTileBytes = (uint64_t)columns * sizeof(float);
        if (rowsPerThreadgroup == 4
            && (batchSize != 1
                || context.q4Rows4Pipeline == nil
                || inputTileBytes > context.maxThreadgroupMemoryLength)) {
            sage_write_error(error, capacity, @"Metal tiled Q4 projection exceeds the single-input threadgroup memory contract");
            return -1;
        }

        id<MTLCommandBuffer> command = [context.queue commandBuffer];
        if (command == nil) {
            sage_write_error(error, capacity, @"Metal Q4 command buffer could not be allocated");
            return -1;
        }
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        if (encoder == nil) {
            sage_write_error(error, capacity, @"Metal Q4 command resources could not be allocated");
            return -1;
        }

        id<MTLComputePipelineState> pipeline = rowsPerThreadgroup == 4
            ? context.q4Rows4Pipeline
            : context.q4Pipeline;
        NSUInteger width = pipeline.threadExecutionWidth;
        NSUInteger maximumLanes = pipeline.maxTotalThreadsPerThreadgroup / rowsPerThreadgroup;
        width = MIN(width, maximumLanes);
        width = MAX(width, 1);
        SageQ4Params params = { rows, columns, groupSize, batchSize, batchTileSize, (uint32_t)width };
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:weights.buffer offset:0 atIndex:0];
        [encoder setBuffer:scales.buffer offset:0 atIndex:1];
        [encoder setBuffer:input.buffer offset:0 atIndex:2];
        [encoder setBuffer:output.buffer offset:0 atIndex:3];
        [encoder setBytes:&params length:sizeof(params) atIndex:4];

        if (rowsPerThreadgroup == 4) {
            [encoder setThreadgroupMemoryLength:(NSUInteger)inputTileBytes atIndex:0];
        }
        NSUInteger rowTiles = ((NSUInteger)rows + rowsPerThreadgroup - 1) / rowsPerThreadgroup;
        NSUInteger batchTiles = ((NSUInteger)batchSize + batchTileSize - 1) / batchTileSize;
        [encoder dispatchThreadgroups:MTLSizeMake(rowTiles, batchTiles, 1) threadsPerThreadgroup:MTLSizeMake(width * rowsPerThreadgroup, 1, 1)];
        [encoder endEncoding];
        [command commit];
        [command waitUntilCompleted];
        if (command.status != MTLCommandBufferStatusCompleted) {
            sage_write_error(error, capacity, command.error.localizedDescription ?: @"Metal Q4 command did not complete");
            return -1;
        }
        if (gpuDurationNs != NULL) {
            double gpuSeconds = command.GPUEndTime - command.GPUStartTime;
            *gpuDurationNs = gpuSeconds > 0.0 && gpuSeconds < 18.0
                ? (uint64_t)(gpuSeconds * 1000000000.0 + 0.5)
                : 0;
        }
        return 0;
    }
}
