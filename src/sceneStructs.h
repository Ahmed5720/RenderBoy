#pragma once

#include <cuda_runtime.h>

#include "glm/glm.hpp"

#include <string>
#include <vector>

#define BACKGROUND_COLOR (glm::vec3(0.3f))

enum GeomType
{
    SPHERE,
    CUBE
};
struct BBox
{
    glm::vec3 min = { 1000, 1000, 1000 };
    glm::vec3 max = { -1000, -1000, -1000 };
};
struct BsdfParams
{
    glm::vec3 diffuseColor;
    glm::vec3 F0;
    float alpha;
};
struct Ray
{
    glm::vec3 origin;
    glm::vec3 direction;
};

struct Geom
{
    enum GeomType type;
    int materialid;
    glm::vec3 translation;
    glm::vec3 rotation;
    glm::vec3 scale;
    glm::mat4 transform;
    glm::mat4 inverseTransform;
    glm::mat4 invTranspose;
};


struct Triangle {
    glm::vec3 v[3];   // world-space positions - after transforms are applied
    glm::vec3 n[3];   
    glm::vec2 uv[3];
    int materialid;
};


// GPU safe
struct Material
{
    glm::vec3 color = { 1,1,1 };
    float specular = 1.0;
    float roughness = 0.5;
    float metallic = 0.0f;
    cudaTextureObject_t diffuseMap = 0;
    cudaTextureObject_t specularMap = 0;
    float shininess = 0.0f;

    float hasReflective = 0.0f;
    float hasRefractive = 0.0f;
    float indexOfRefraction = 1.0f;
    float emittance = 0.0f;
};

// Host-side material with texture paths (not copied to GPU)
struct MaterialHost : Material
{
    std::string name;
    std::string diffuseTexPath;
    std::string specularTexPath;
};

struct Camera
{
    glm::ivec2 resolution;
    glm::vec3 position;
    glm::vec3 lookAt;
    glm::vec3 view;
    glm::vec3 up;
    glm::vec3 right;
    glm::vec2 fov;
    glm::vec2 pixelLength;
};

struct RenderState
{
    Camera camera;
    unsigned int iterations;
    int traceDepth;
    std::vector<glm::vec3> image;
    std::string imageName;
};

struct PathSegment
{
    Ray ray;
    glm::vec3 color;      // accumulated radiance along the path
    glm::vec3 throughput; // BSDF throughput (starts at 1)
    int pixelIndex;
    int remainingBounces;
};

// Use with a corresponding PathSegment to do:
// 1) color contribution computation
// 2) BSDF evaluation: generate a new ray
struct ShadeableIntersection
{
  float t;
  glm::vec3 surfaceNormal;
  glm::vec2 uv;
  int materialId;

  int debugVisitedNodes;
};
