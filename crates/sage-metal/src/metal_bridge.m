#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <CoreFoundation/CoreFoundation.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

@interface SageMetalContext : NSObject
@property(nonatomic, strong) id<MTLDevice> device;
@property(nonatomic, strong) id<MTLCommandQueue> queue;
@property(nonatomic, strong) id<MTLComputePipelineState> q4Pipeline;
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
        "struct SageQ4Params { uint rows; uint columns; uint groupSize; uint threadsPerGroup; };\n"
        "kernel void sage_q4_project(device const uchar *packed [[buffer(0)]],\n"
        " device const float *scales [[buffer(1)]],\n"
        " device const float *input [[buffer(2)]],\n"
        " device float *output [[buffer(3)]],\n"
        " constant SageQ4Params &p [[buffer(4)]],\n"
        " uint row [[threadgroup_position_in_grid]],\n"
        " uint lane [[thread_index_in_threadgroup]]) {\n"
        " if (row >= p.rows) return;\n"
        " ulong base = ulong(row) * ulong(p.columns);\n"
        " float sum = 0.0f;\n"
        " if ((p.columns & 1) == 0) {\n"
        "  for (uint pairIndex = lane; pairIndex < p.columns / 2; pairIndex += p.threadsPerGroup) {\n"
        "   uint column = pairIndex * 2;\n"
        "   ulong index = base + ulong(column);\n"
        "   uchar pair = packed[index >> 1];\n"
        "   int low = int(pair & 15) - 8;\n"
        "   int high = int(pair >> 4) - 8;\n"
        "   float lowScale = scales[index / ulong(p.groupSize)];\n"
        "   float highScale = scales[(index + 1) / ulong(p.groupSize)];\n"
        "   sum = fma(float(low) * lowScale, input[column], sum);\n"
        "   sum = fma(float(high) * highScale, input[column + 1], sum);\n"
        "  }\n"
        " } else {\n"
        "  for (uint column = lane; column < p.columns; column += p.threadsPerGroup) {\n"
        "   ulong index = base + ulong(column);\n"
        "   uchar pair = packed[index >> 1];\n"
        "   int q = int((index & 1) == 0 ? (pair & 15) : (pair >> 4)) - 8;\n"
        "   float weight = float(q) * scales[index / ulong(p.groupSize)];\n"
        "   sum = fma(weight, input[column], sum);\n"
        "  }\n"
        " }\n"
        " float total = simd_sum(sum);\n"
        " if (lane == 0) output[row] = total;\n"
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
        id<MTLCommandQueue> queue = [device newCommandQueue];
        if (queue == nil) {
            sage_write_error(error, capacity, @"Metal command queue creation failed");
            return NULL;
        }

        SageMetalContext *context = [SageMetalContext new];
        context.device = device;
        context.queue = queue;
        context.q4Pipeline = pipeline;
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
    char *error,
    size_t capacity
) {
    @autoreleasepool {
        if (rawContext == NULL || rawWeights == NULL || rawScales == NULL || rawInput == NULL || rawOutput == NULL || rows == 0 || columns == 0 || groupSize == 0) {
            sage_write_error(error, capacity, @"Metal Q4 projection arguments are invalid");
            return -1;
        }
        SageMetalContext *context = (__bridge SageMetalContext *)rawContext;
        SageMetalBuffer *weights = (__bridge SageMetalBuffer *)rawWeights;
        SageMetalBuffer *scales = (__bridge SageMetalBuffer *)rawScales;
        SageMetalBuffer *input = (__bridge SageMetalBuffer *)rawInput;
        SageMetalBuffer *output = (__bridge SageMetalBuffer *)rawOutput;
        uint64_t elements = (uint64_t)rows * (uint64_t)columns;
        if (elements > 700000000ULL) {
            sage_write_error(error, capacity, @"Metal Q4 projection exceeds Sage's element bound");
            return -1;
        }
        uint64_t requiredWeightBytes = (elements + 1) / 2;
        uint64_t scaleCount = (elements + (uint64_t)groupSize - 1) / (uint64_t)groupSize;
        if (requiredWeightBytes != weights.buffer.length
            || scaleCount * sizeof(float) != scales.buffer.length
            || scaleCount > UINT32_MAX
            || input.buffer.length != (NSUInteger)columns * sizeof(float)
            || output.buffer.length != (NSUInteger)rows * sizeof(float)
            || input.buffer.contents == NULL
            || output.buffer.contents == NULL) {
            sage_write_error(error, capacity, @"Metal Q4 buffers do not match the declared matrix");
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

        NSUInteger width = context.q4Pipeline.threadExecutionWidth;
        width = MIN(width, context.q4Pipeline.maxTotalThreadsPerThreadgroup);
        width = MAX(width, 1);
        SageQ4Params params = { rows, columns, groupSize, (uint32_t)width };
        [encoder setComputePipelineState:context.q4Pipeline];
        [encoder setBuffer:weights.buffer offset:0 atIndex:0];
        [encoder setBuffer:scales.buffer offset:0 atIndex:1];
        [encoder setBuffer:input.buffer offset:0 atIndex:2];
        [encoder setBuffer:output.buffer offset:0 atIndex:3];
        [encoder setBytes:&params length:sizeof(params) atIndex:4];

        [encoder dispatchThreadgroups:MTLSizeMake(rows, 1, 1) threadsPerThreadgroup:MTLSizeMake(width, 1, 1)];
        [encoder endEncoding];
        [command commit];
        [command waitUntilCompleted];
        if (command.status != MTLCommandBufferStatusCompleted) {
            sage_write_error(error, capacity, command.error.localizedDescription ?: @"Metal Q4 command did not complete");
            return -1;
        }
        return 0;
    }
}
