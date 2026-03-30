#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <png.h>
#include <CL/cl.h>

// Error checking
#define CHECK_ERROR(err, msg) \
    if (err != CL_SUCCESS) { \
        fprintf(stderr, "%s (error: %d)\n", msg, err); \
        return 0; \
    }

// Save PNG image
static int save_png(const char* filename, uint8_t* image, int width, int height) {
    FILE *fp = fopen(filename, "wb");
    if (!fp) return 0;
    
    png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (!png_ptr) { fclose(fp); return 0; }
    
    png_infop info_ptr = png_create_info_struct(png_ptr);
    if (!info_ptr) { png_destroy_write_struct(&png_ptr, NULL); fclose(fp); return 0; }
    
    if (setjmp(png_jmpbuf(png_ptr))) {
        png_destroy_write_struct(&png_ptr, &info_ptr);
        fclose(fp);
        return 0;
    }
    
    png_init_io(png_ptr, fp);
    png_set_IHDR(png_ptr, info_ptr, width, height, 8,
                 PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,
                 PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
    
    png_write_info(png_ptr, info_ptr);
    
    png_bytep row_pointers[height];
    for (int y = 0; y < height; y++) {
        row_pointers[y] = (png_bytep)(image + y * width * 3);
    }
    
    png_write_image(png_ptr, row_pointers);
    png_write_end(png_ptr, NULL);
    
    png_destroy_write_struct(&png_ptr, &info_ptr);
    fclose(fp);
    return 1;
}

// Read kernel from file
static char* read_kernel_source(const char* filename, size_t* length) {
    FILE* file = fopen(filename, "r");
    if (!file) return NULL;
    
    fseek(file, 0, SEEK_END);
    *length = ftell(file);
    fseek(file, 0, SEEK_SET);
    
    char* source = malloc(*length + 1);
    fread(source, 1, *length, file);
    source[*length] = '\0';
    
    fclose(file);
    return source;
}

// Print help
static void print_help(const char* prog_name) {
    printf("GPU Ray Tracer - Final Version\n");
    printf("Usage: %s [options]\n\n", prog_name);
    printf("Options:\n");
    printf("  -w <width>      Image width (default: 800)\n");
    printf("  -h <height>     Image height (default: 600)\n");
    printf("  -s <samples>    AA samples per pixel (default: 4)\n");
    printf("  -o <file>       Output filename (default: render.png)\n");
    printf("  --no-shadow     Disable shadows\n");
    printf("  --help          Show this help\n");
    printf("\nExamples:\n");
    printf("  %s -w 1920 -h 1080 -s 1    # Fast 1080p\n", prog_name);
    printf("  %s -w 800 -h 600 -s 16     # High quality\n", prog_name);
    printf("  %s -w 400 -h 300           # Quick test\n", prog_name);
}

int main(int argc, char* argv[]) {
    // Default configuration
    int width = 800;
    int height = 600;
    int samples = 4;
    int shadows = 1;
    char output[256] = "render.png";
    
    // Parse command line arguments
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--help") == 0) {
            print_help(argv[0]);
            return 0;
        }
        else if (strcmp(argv[i], "--no-shadow") == 0) {
            shadows = 0;
        }
        else if (strcmp(argv[i], "-w") == 0 && i + 1 < argc) {
            width = atoi(argv[++i]);
            if (width <= 0) width = 800;
        }
        else if (strcmp(argv[i], "-h") == 0 && i + 1 < argc) {
            height = atoi(argv[++i]);
            if (height <= 0) height = 600;
        }
        else if (strcmp(argv[i], "-o") == 0 && i + 1 < argc) {
            strncpy(output, argv[++i], sizeof(output) - 1);
        }
        else if (strcmp(argv[i], "-s") == 0 && i + 1 < argc) {
            samples = atoi(argv[++i]);
            if (samples < 1) samples = 1;
        }
    }
    
    printf("=== GPU Ray Tracer ===\n");
    printf("Target: Adreno 610 GPU\n");
    printf("Resolution: %dx%d\n", width, height);
    printf("Samples per pixel: %d\n", samples);
    printf("Shadows: %s\n", shadows ? "Enabled" : "Disabled");
    printf("Total rays: %d million\n", (width * height * samples) / 1000000);
    
    clock_t start_time = clock();
    
    // OpenCL setup
    cl_int err;
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_kernel kernel;
    
    // Get OpenCL platform
    err = clGetPlatformIDs(1, &platform, NULL);
    CHECK_ERROR(err, "Failed to get platform");
    
    // Get GPU device (try CPU if GPU fails)
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    if (err != CL_SUCCESS) {
        printf("GPU not available, using CPU...\n");
        err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &device, NULL);
        CHECK_ERROR(err, "Failed to get any device");
    }
    
    // Create OpenCL context
    context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    CHECK_ERROR(err, "Failed to create context");
    
    // Create command queue
    queue = clCreateCommandQueueWithProperties(context, device, 0, &err);
    CHECK_ERROR(err, "Failed to create command queue");
    
    // Read and compile kernel
    size_t kernel_size;
    char* kernel_source = read_kernel_source("kernels/raytracer.cl", &kernel_size);
    if (!kernel_source) {
        fprintf(stderr, "Failed to read kernel file\n");
        return 1;
    }
    
    const char* sources[] = { kernel_source };
    program = clCreateProgramWithSource(context, 1, sources, &kernel_size, &err);
    CHECK_ERROR(err, "Failed to create program");
    
    // Build program
    err = clBuildProgram(program, 1, &device, "", NULL, NULL);
    if (err != CL_SUCCESS) {
        size_t log_size;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        char* log = malloc(log_size);
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
        fprintf(stderr, "Build failed:\n%s\n", log);
        free(log);
        free(kernel_source);
        return 1;
    }
    
    free(kernel_source);
    
    // Create kernel
    kernel = clCreateKernel(program, "raytrace_kernel", &err);
    CHECK_ERROR(err, "Failed to create kernel");
    
    // Scene data: 3 spheres (red center, green right, blue left)
    float spheres_data[] = {
        // Center sphere (red)
        0.0f, 0.0f, 0.0f,   // center x,y,z
        1.0f,               // radius
        1.0f, 0.2f, 0.2f,  // color r,g,b
        
        // Right sphere (green)
        1.5f, 0.0f, 0.0f,
        0.5f,
        0.2f, 1.0f, 0.2f,
        
        // Left sphere (blue)
        -1.5f, 0.0f, 0.0f,
        0.5f,
        0.2f, 0.2f, 1.0f
    };
    int sphere_count = 3;
    
    // Create buffers
    size_t image_pixels = width * height * 3;
    size_t float_buffer_size = image_pixels * sizeof(float);
    
    float* float_output = malloc(float_buffer_size);
    uint8_t* final_image = malloc(image_pixels);
    
    cl_mem buffer_output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, 
                                         float_buffer_size, NULL, &err);
    CHECK_ERROR(err, "Failed to create output buffer");
    
    cl_mem buffer_spheres = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                          sizeof(spheres_data), spheres_data, &err);
    CHECK_ERROR(err, "Failed to create spheres buffer");
    
    // Set kernel arguments
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &buffer_output);
    err |= clSetKernelArg(kernel, 1, sizeof(int), &width);
    err |= clSetKernelArg(kernel, 2, sizeof(int), &height);
    err |= clSetKernelArg(kernel, 3, sizeof(int), &samples);
    err |= clSetKernelArg(kernel, 4, sizeof(int), &shadows);
    err |= clSetKernelArg(kernel, 5, sizeof(cl_mem), &buffer_spheres);
    err |= clSetKernelArg(kernel, 6, sizeof(int), &sphere_count);
    CHECK_ERROR(err, "Failed to set kernel arguments");
    
    // Execute kernel
    size_t global_work_size[2] = {width, height};
    size_t local_work_size[2] = {8, 4};  // Workgroup size for mobile
    
    err = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, 
                                global_work_size, local_work_size, 
                                0, NULL, NULL);
    CHECK_ERROR(err, "Failed to execute kernel");
    
    // Read results back
    err = clEnqueueReadBuffer(queue, buffer_output, CL_TRUE, 0,
                             float_buffer_size, float_output, 0, NULL, NULL);
    CHECK_ERROR(err, "Failed to read buffer");
    
    // Convert floats to 8-bit RGB
    for (int i = 0; i < image_pixels; i++) {
        float val = float_output[i];
        val = fminf(fmaxf(val, 0.0f), 1.0f);
        final_image[i] = (uint8_t)(val * 255.0f);
    }
    
    // Save PNG
    if (save_png(output, final_image, width, height)) {
        printf("Saved: %s\n", output);
    } else {
        fprintf(stderr, "Failed to save PNG\n");
    }
    
    // Cleanup OpenCL
    clReleaseMemObject(buffer_output);
    clReleaseMemObject(buffer_spheres);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    
    // Free memory
    free(float_output);
    free(final_image);
    
    // Performance stats
    clock_t end_time = clock();
    double render_time = ((double)(end_time - start_time)) / CLOCKS_PER_SEC;
    double total_rays = (double)width * height * samples;
    
    printf("\n📊 Performance:\n");
    printf("  Render time: %.2f seconds\n", render_time);
    printf("  Rays/second: %.1f million\n", total_rays / render_time / 1000000.0);
    printf("  Pixels/second: %.1f million\n", (width * height) / render_time / 1000000.0);
    
    return 0;
}