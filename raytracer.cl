// GPU Ray Tracer Kernel - Optimized for Mobile GPUs
// Renders 3 spheres with shadows and anti-aliasing

typedef struct {
    float x, y, z;
} Vec3;

typedef struct {
    Vec3 center;
    float radius;
    Vec3 color;
} Sphere;

// Vector math functions
static Vec3 vec3_add(Vec3 a, Vec3 b) {
    return (Vec3){a.x + b.x, a.y + b.y, a.z + b.z};
}

static Vec3 vec3_sub(Vec3 a, Vec3 b) {
    return (Vec3){a.x - b.x, a.y - b.y, a.z - b.z};
}

static Vec3 vec3_mul(Vec3 a, float s) {
    return (Vec3){a.x * s, a.y * s, a.z * s};
}

static float vec3_dot(Vec3 a, Vec3 b) {
    return a.x*b.x + a.y*b.y + a.z*b.z;
}

static float vec3_length(Vec3 a) {
    return sqrt(vec3_dot(a, a));
}

static Vec3 vec3_normalize(Vec3 a) {
    float len = vec3_length(a);
    if (len > 0.0f) return vec3_mul(a, 1.0f / len);
    return a;
}

// Sphere intersection
static int sphere_intersect(Vec3 ray_origin, Vec3 ray_dir,
                            Sphere sphere, float* t) {
    Vec3 oc = vec3_sub(ray_origin, sphere.center);
    
    float a = vec3_dot(ray_dir, ray_dir);
    float b = 2.0f * vec3_dot(oc, ray_dir);
    float c = vec3_dot(oc, oc) - sphere.radius * sphere.radius;
    
    float discriminant = b*b - 4.0f*a*c;
    
    if (discriminant < 0.0f) return 0;
    
    float sqrt_disc = sqrt(discriminant);
    float t0 = (-b - sqrt_disc) / (2.0f * a);
    float t1 = (-b + sqrt_disc) / (2.0f * a);
    
    if (t0 > 0.001f) {
        *t = t0;
        return 1;
    }
    if (t1 > 0.001f) {
        *t = t1;
        return 1;
    }
    
    return 0;
}

// Random number for AA
static float random(uint2* seed) {
    seed->x = 1664525u * seed->x + 1013904223u;
    seed->y = 1664525u * seed->y + 1013904223u;
    return (float)(seed->x ^ seed->y) / (float)0xFFFFFFFFu;
}

// Main kernel
__kernel void raytrace_kernel(
    __global float* output,
    int width,
    int height,
    int samples,
    int shadows,
    __global float* spheres_data,
    int sphere_count
) {
    uint x = get_global_id(0);
    uint y = get_global_id(1);
    
    if (x >= (uint)width || y >= (uint)height) return;
    
    uint idx = (y * (uint)width + x) * 3u;
    
    // Convert packed sphere data
    Sphere spheres[3];
    for (int i = 0; i < sphere_count; i++) {
        int base = i * 7;
        spheres[i].center = (Vec3){
            spheres_data[base],
            spheres_data[base + 1],
            spheres_data[base + 2]
        };
        spheres[i].radius = spheres_data[base + 3];
        spheres[i].color = (Vec3){
            spheres_data[base + 4],
            spheres_data[base + 5],
            spheres_data[base + 6]
        };
    }
    
    Vec3 pixel_color = (Vec3){0.0f, 0.0f, 0.0f};
    uint2 seed = (uint2){x * 123456789u, y * 987654321u};
    
    // Anti-aliasing samples
    for (int s = 0; s < samples; s++) {
        // Jitter for AA
        float offset_x = (samples > 1) ? random(&seed) - 0.5f : 0.0f;
        float offset_y = (samples > 1) ? random(&seed) - 0.5f : 0.0f;
        
        // Camera setup
        float fx = (float)x + 0.5f + offset_x;
        float fy = (float)y + 0.5f + offset_y;
        float fwidth = (float)width;
        float fheight = (float)height;
        
        float aspect_ratio = fwidth / fheight;
        float px = (2.0f * (fx / fwidth) - 1.0f) * aspect_ratio;
        float py = 1.0f - 2.0f * (fy / fheight);
        
        Vec3 ray_origin = (Vec3){0.0f, 0.0f, 3.0f};
        Vec3 ray_dir = vec3_normalize((Vec3){px, py, -1.0f});
        
        // Find closest hit
        float closest_t = 1e10f;
        int hit_idx = -1;
        
        for (int i = 0; i < sphere_count; i++) {
            float t;
            if (sphere_intersect(ray_origin, ray_dir, spheres[i], &t)) {
                if (t < closest_t) {
                    closest_t = t;
                    hit_idx = i;
                }
            }
        }
        
        if (hit_idx >= 0) {
            // Hit point and normal
            Vec3 hit_point = vec3_add(ray_origin, vec3_mul(ray_dir, closest_t));
            Vec3 normal = vec3_normalize(vec3_sub(hit_point, spheres[hit_idx].center));
            
            // Light
            Vec3 light_pos = (Vec3){2.0f, 3.0f, 1.0f};
            Vec3 light_dir = vec3_normalize(vec3_sub(light_pos, hit_point));
            
            // Shadows
            float shadow_factor = 1.0f;
            if (shadows) {
                Vec3 shadow_origin = vec3_add(hit_point, vec3_mul(normal, 0.001f));
                for (int i = 0; i < sphere_count; i++) {
                    float shadow_t;
                    if (i != hit_idx && sphere_intersect(shadow_origin, light_dir, spheres[i], &shadow_t)) {
                        if (shadow_t > 0.001f) {
                            shadow_factor = 0.3f;
                            break;
                        }
                    }
                }
            }
            
            // Lighting
            float diffuse = fmax(0.0f, vec3_dot(normal, light_dir)) * shadow_factor;
            float ambient = 0.1f;
            float brightness = ambient + (1.0f - ambient) * diffuse;
            
            // Final color
            Vec3 color = (Vec3){
                spheres[hit_idx].color.x * brightness,
                spheres[hit_idx].color.y * brightness,
                spheres[hit_idx].color.z * brightness
            };
            
            pixel_color = vec3_add(pixel_color, color);
        } else {
            // Background gradient
            float t = 0.5f * (ray_dir.y + 1.0f);
            Vec3 bg_color = (Vec3){
                1.0f * (1.0f - t) + 0.5f * t,
                1.0f * (1.0f - t) + 0.7f * t,
                1.0f * (1.0f - t) + 1.0f * t
            };
            pixel_color = vec3_add(pixel_color, bg_color);
        }
    }
    
    // Average samples
    pixel_color = vec3_mul(pixel_color, 1.0f / (float)samples);
    
    // Gamma correction
    pixel_color.x = sqrt(pixel_color.x);
    pixel_color.y = sqrt(pixel_color.y);
    pixel_color.z = sqrt(pixel_color.z);
    
    // Clamp
    pixel_color.x = fmin(fmax(pixel_color.x, 0.0f), 1.0f);
    pixel_color.y = fmin(fmax(pixel_color.y, 0.0f), 1.0f);
    pixel_color.z = fmin(fmax(pixel_color.z, 0.0f), 1.0f);
    
    output[idx] = pixel_color.x;
    output[idx + 1u] = pixel_color.y;
    output[idx + 2u] = pixel_color.z;
}