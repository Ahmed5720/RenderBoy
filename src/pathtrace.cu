#include "pathtrace.h"

#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust/sort.h>

#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "intersections.h"
#include "interactions.h"
#include <thrust/partition.h>
#define ERRORCHECK 1

#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define checkCUDAError(msg) checkCUDAErrorFn(msg, FILENAME, __LINE__)
void checkCUDAErrorFn(const char* msg, const char* file, int line)
{
#if ERRORCHECK
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess == err)
    {
        return;
    }

    fprintf(stderr, "CUDA error");
    if (file)
    {
        fprintf(stderr, " (%s:%d)", file, line);
    }
    fprintf(stderr, ": %s: %s\n", msg, cudaGetErrorString(err));
#ifdef _WIN32
    getchar();
#endif // _WIN32
    exit(EXIT_FAILURE);
#endif // ERRORCHECK
}

__host__ __device__
thrust::default_random_engine makeSeededRandomEngine(int iter, int index, int depth)
{
    int h = utilhash((1 << 31) | (depth << 22) | iter) ^ utilhash(index);
    return thrust::default_random_engine(h);
}

//Kernel that writes the image to the OpenGL PBO directly.
__global__ void sendImageToPBO(uchar4* pbo, glm::ivec2 resolution, int iter, glm::vec3* image)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < resolution.x && y < resolution.y)
    {
        int index = x + (y * resolution.x);
        glm::vec3 pix = image[index];

        glm::ivec3 color;
        color.x = glm::clamp((int)(pix.x / iter * 255.0), 0, 255);
        color.y = glm::clamp((int)(pix.y / iter * 255.0), 0, 255);
        color.z = glm::clamp((int)(pix.z / iter * 255.0), 0, 255);

        // Each thread writes one pixel location in the texture (textel)
        pbo[index].w = 0;
        pbo[index].x = color.x;
        pbo[index].y = color.y;
        pbo[index].z = color.z;
    }
}

struct IsPathActive
{
    __host__ __device__ bool operator()(const PathSegment& p) const
    {
        return p.remainingBounces > 0;
    }
};

struct BVH_node
{
    BBox box;
    int left = -1;
    int right = -1;
    std::vector<int> tri_indices;
};
struct BVH_node_gpu
{
    BBox box;
    int left = -1;
    int right = -1;
    int start = 0;
    int count = 0;
};

static Scene* hst_scene = NULL;
static GuiDataContainer* guiData = NULL;
static glm::vec3* dev_image = NULL;
static Geom* dev_geoms = NULL; // now only stores primitives. better to call it primitive than geometry since meshes are stored only as triangles
static Material* dev_materials = NULL;
static Triangle* dev_tris = NULL;
static BVH_node_gpu* dev_bvh = NULL;
static int num_tris = 0;
static PathSegment* dev_paths = NULL;
static ShadeableIntersection* dev_intersections = NULL;
static int* dev_emissive_geoms = NULL;
static int num_emissive_geoms = 0;
int bvhNodeCount;

int maxDepth = 20;
std::vector<BVH_node> bvh; //  ((1 << maxDepth) - 1);
std::vector<BVH_node_gpu> bvh_gpu((1 << maxDepth) - 1);




// Returns position of cut (float) at middle of longest axis, and index of axis of cutting 0-1-2 for x,y,z
std::pair<float, int> findLongestAxis(BBox box)
{
    // Calculate extents for each axis
    float extX = box.max.x - box.min.x;
    float extY = box.max.y - box.min.y;
    float extZ = box.max.z - box.min.z;

    // Find the longest axis
    int axis = 0;
    float maxExtent = extX;

    if (extY > maxExtent) {
        maxExtent = extY;
        axis = 1;
    }
    if (extZ > maxExtent) {
        maxExtent = extZ;
        axis = 2;
    }

    // Calculate cut position at the middle of the longest axis
    float cutPos = 0.0f;
    switch (axis) {
    case 0: cutPos = (box.min.x + box.max.x) * 0.5f; break;
    case 1: cutPos = (box.min.y + box.max.y) * 0.5f; break;
    case 2: cutPos = (box.min.z + box.max.z) * 0.5f; break;
    }

   // cutPos = (box.min.y + box.max.y) * 0.5f;
    return std::make_pair(cutPos, axis);
}
// Cuts box using findLongestAxis and updates left and right boxes
void cutAlongAxis(const BBox& box, BBox& left, BBox& right)
{
    // Find the longest axis and cut position
    std::pair<float, int> cutInfo = findLongestAxis(box);
    float cutPos = cutInfo.first;
    int axis = cutInfo.second;

    // Initialize left and right boxes as copies of the original
    left = box;
    right = box;

    // Update the bounds based on the cut axis
    switch (axis) {
    case 0: // X-axis
        left.max.x = cutPos;
        right.min.x = cutPos;
        break;
    case 1: // Y-axis
        left.max.y = cutPos;
        right.min.y = cutPos;
        break;
    case 2: // Z-axis
        left.max.z = cutPos;
        right.min.z = cutPos;
        break;
    }
}


static const int   SAH_BUCKETS = 12;
static const int   MAX_LEAF_TRIS = 4;
static const int   MAX_BVH_DEPTH = 64;  // keep strictly below the GPU stack[64]

struct PrimInfo { BBox box; glm::vec3 centroid; };

static float surfaceArea(const BBox& b)
{
    glm::vec3 d = b.max - b.min;
    if (d.x < 0.0f || d.y < 0.0f || d.z < 0.0f) return 0.0f; // empty box
    return 2.0f * (d.x * d.y + d.y * d.z + d.z * d.x);
}
static BBox emptyBox()
{
    BBox b; b.min = glm::vec3(FLT_MAX); b.max = glm::vec3(-FLT_MAX); return b;
}
static void expand(BBox& b, const glm::vec3& p) { b.min = glm::min(b.min, p); b.max = glm::max(b.max, p); }
static void expand(BBox& b, const BBox& o) { b.min = glm::min(b.min, o.min); b.max = glm::max(b.max, o.max); }

void buildSAH(int node_id, int depth, const std::vector<PrimInfo>& prims)
{
    BVH_node& node = bvh[node_id];
    const int N = (int)node.tri_indices.size();

    if (N <= MAX_LEAF_TRIS || depth >= MAX_BVH_DEPTH) { node.left = node.right = -1; return; }

    // Binning is done over CENTROID bounds, not the geometric box -> better bucket spread.
    BBox cbox = emptyBox();
    for (int ti : node.tri_indices) expand(cbox, prims[ti].centroid);
    glm::vec3 cext = cbox.max - cbox.min;
    if (cext.x <= 0.0f && cext.y <= 0.0f && cext.z <= 0.0f) { node.left = node.right = -1; return; } // all centroids coincide

    int bestAxis = -1, bestSplit = -1;
    float bestCost = FLT_MAX;

    for (int axis = 0; axis < 3; axis++)
    {
        float lo = cbox.min[axis], ext = cext[axis];
        if (ext <= 0.0f) continue;
        float scale = SAH_BUCKETS / ext;

        BBox bbox[SAH_BUCKETS]; int bcnt[SAH_BUCKETS];
        for (int b = 0; b < SAH_BUCKETS; b++) { bbox[b] = emptyBox(); bcnt[b] = 0; }

        for (int ti : node.tri_indices) {
            int b = (int)((prims[ti].centroid[axis] - lo) * scale);
            b = glm::clamp(b, 0, SAH_BUCKETS - 1);
            bcnt[b]++; expand(bbox[b], prims[ti].box);
        }

        // suffix sweep: right-side box/count for each boundary
        BBox rBox[SAH_BUCKETS]; int rCnt[SAH_BUCKETS];
        rBox[SAH_BUCKETS - 1] = bbox[SAH_BUCKETS - 1];
        rCnt[SAH_BUCKETS - 1] = bcnt[SAH_BUCKETS - 1];
        for (int b = SAH_BUCKETS - 2; b >= 0; b--) {
            rBox[b] = rBox[b + 1]; expand(rBox[b], bbox[b]);
            rCnt[b] = rCnt[b + 1] + bcnt[b];
        }

        // prefix sweep: evaluate the K-1 candidate planes
        BBox lBox = emptyBox(); int lCnt = 0;
        for (int b = 0; b < SAH_BUCKETS - 1; b++) {
            expand(lBox, bbox[b]); lCnt += bcnt[b];
            if (lCnt == 0 || rCnt[b + 1] == 0) continue;
            float cost = lCnt * surfaceArea(lBox) + rCnt[b + 1] * surfaceArea(rBox[b + 1]);
            if (cost < bestCost) { bestCost = cost; bestAxis = axis; bestSplit = b; }
        }
    }

    if (bestAxis == -1) { node.left = node.right = -1; return; } // no usable split

    // partition tri_indices by the winning axis/boundary (same bucket math as above)
    float lo = cbox.min[bestAxis], scale = SAH_BUCKETS / cext[bestAxis];
    BVH_node left, right; left.box = emptyBox(); right.box = emptyBox();
    for (int ti : node.tri_indices) {
        int b = glm::clamp((int)((prims[ti].centroid[bestAxis] - lo) * scale), 0, SAH_BUCKETS - 1);
        if (b <= bestSplit) { left.tri_indices.push_back(ti);  expand(left.box, prims[ti].box); }
        else { right.tri_indices.push_back(ti); expand(right.box, prims[ti].box); }
    }
    if (left.tri_indices.empty() || right.tri_indices.empty()) { node.left = node.right = -1; return; }

    int left_idx = (int)bvh.size(), right_idx = left_idx + 1;
    bvh.emplace_back(std::move(left));
    bvh.emplace_back(std::move(right));
    bvh[node_id].left = left_idx;       // re-index via bvh[] (node ref may be stale after emplace)
    bvh[node_id].right = right_idx;
    bvh[node_id].tri_indices.clear();

    buildSAH(left_idx, depth + 1, prims);
    buildSAH(right_idx, depth + 1, prims);
}
void helper(int node_id, int depth, const std::vector<Triangle>& triangleArray)
{
    BVH_node& node = bvh[node_id];
    if (node.tri_indices.size() <= 4)
    {
        node.left = -1;
        node.right = -1;
        return;
    }


    auto [mid, ax] = findLongestAxis(node.box);

    BVH_node left;
    BVH_node right;
    
    for (int triIdx : node.tri_indices) {
        const Triangle& tri = triangleArray[triIdx];
        // Use centroid of triangle for splitting
        glm::vec3 centroid = (tri.v[0] + tri.v[1] + tri.v[2]) / 3.0f;
        float coord = (ax == 0) ? centroid.x : (ax == 1) ? centroid.y : centroid.z;

        if (coord < mid) {
            left.tri_indices.push_back(triIdx);
            for (int i = 0; i < 3; i++) {
                left.box.min = glm::min(left.box.min, tri.v[i]);
                left.box.max = glm::max(left.box.max, tri.v[i]);
            }
        }
        else {
            right.tri_indices.push_back(triIdx);
            for (int i = 0; i < 3; i++) {
                right.box.min = glm::min(right.box.min, tri.v[i]);
                right.box.max = glm::max(right.box.max, tri.v[i]);
            }
        }
    }
    if (left.tri_indices.empty()  ||  right.tri_indices.empty())
    {
        node.left = -1;
        node.right = -1;
        return;
    }
   
    // cutting the box in half and building the child boxes off of that means that we can have plenty of volume wasted that contain few triangles. 
    // consider the case where 1 triangle is on the left part and 99 are on the right part. here we are assigning boxes of equal sizes for both sides and wasting alot of traversal on the left check.
    // instead we would like to grow bounds to better fit the child nodes.
    //cutAlongAxis(node.box, left.box, right.box);

    /*int left_idx = node_id * 2 +1;
    int right_idx = node_id * 2 + 2;*/

    int left_idx = (int)bvh.size();
    int right_idx = left_idx + 1;
    bvh.emplace_back(std::move(left));
    bvh.emplace_back(std::move(right));

   
    node.left = left_idx;
    node.right = right_idx;

    bvh[node_id].left = left_idx;    
    bvh[node_id].right = right_idx;
    node.tri_indices.clear();

    helper(left_idx, depth - 1, triangleArray);
    helper(right_idx, depth - 1, triangleArray);

}
// we recursively build a bounding volume hiararchy for our mesh, we produce a triangle reordered such that each leaf's triangles are contigious in [start, start + cnt]
//void buildBVH(const std::vector<Triangle>& tris, std::vector<Triangle>& orderedOut)
//{
//
//    int index = 0;
//    glm::vec3 minB = { 1000.0f, 1000.0f, 1000.0f };
//    glm::vec3 maxB = { -1000.0f, -1000.0f, -1000.0f };
//    for (const Triangle& tri : tris) {
//        for (int v = 0; v < 3; v++) {
//            minB = glm::min(minB, tri.v[v]);
//            maxB = glm::max(maxB, tri.v[v]);
//        }
//    }
//
//    BVH_node root;
//    BBox box;
//    box.min = minB;
//    box.max = maxB;
//    root.box = box;
//    root.left = 1;
//    root.right = 2;
//
//
//    for (int i = 0; i < tris.size(); i++)
//        root.tri_indices.push_back(i);
//
//    bvh.clear();
//    bvh.reserve(2 * tris.size() + 1);   
//    bvh.push_back(root);
//
//    helper(0, maxDepth - 1, tris);
//
//    // construct bvh vector ready to be transfered to device
//    bvh_gpu.resize(bvh.size());
//    for (size_t n = 0; n < bvh.size(); n++) {
//        BVH_node_gpu ng;
//        ng.box = bvh[n].box;
//        ng.left = bvh[n].left;
//        ng.right = bvh[n].right;
//        ng.start = 0;
//        ng.count = 0;
//        if (bvh[n].left < 0 && !bvh[n].tri_indices.empty()) {   
//            ng.start = (int)orderedOut.size();
//            ng.count = (int)bvh[n].tri_indices.size();
//            for (int ti : bvh[n].tri_indices)
//                orderedOut.push_back(tris[ti]);
//        }
//        bvh_gpu[n] = ng;
//    }
//
//}


void buildBVH(const std::vector<Triangle>& tris, std::vector<Triangle>& orderedOut)
{
    std::vector<PrimInfo> prims(tris.size());
    BBox rootBox = emptyBox();
    for (size_t i = 0; i < tris.size(); i++) {
        BBox tb = emptyBox();
        for (int v = 0; v < 3; v++) expand(tb, tris[i].v[v]);
        prims[i].box = tb;
        prims[i].centroid = (tris[i].v[0] + tris[i].v[1] + tris[i].v[2]) / 3.0f;
        expand(rootBox, tb);
    }

    BVH_node root; root.box = rootBox;
    root.tri_indices.resize(tris.size());
    for (int i = 0; i < (int)tris.size(); i++) root.tri_indices[i] = i;

    bvh.clear();
    bvh.reserve(2 * tris.size() + 1);   // proper binary tree over N prims -> <= 2N-1 nodes, so no realloc
    bvh.push_back(std::move(root));
    if (!tris.empty()) buildSAH(0, 0, prims);

    // ---- flatten step is UNCHANGED ----
    bvh_gpu.resize(bvh.size());
    for (size_t n = 0; n < bvh.size(); n++) {
        BVH_node_gpu ng;
        ng.box = bvh[n].box; ng.left = bvh[n].left; ng.right = bvh[n].right;
        ng.start = 0; ng.count = 0;
        if (bvh[n].left < 0 && !bvh[n].tri_indices.empty()) {
            ng.start = (int)orderedOut.size();
            ng.count = (int)bvh[n].tri_indices.size();
            for (int ti : bvh[n].tri_indices) orderedOut.push_back(tris[ti]);
        }
        bvh_gpu[n] = ng;
    }
}

void InitDataContainer(GuiDataContainer* imGuiData)
{
    guiData = imGuiData;
}

void pathtraceInit(Scene* scene)
{
    hst_scene = scene;

    const Camera& cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    cudaMalloc(&dev_image, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_image, 0, pixelcount * sizeof(glm::vec3));

    cudaMalloc(&dev_paths, pixelcount * sizeof(PathSegment));

    cudaMalloc(&dev_geoms, scene->geoms.size() * sizeof(Geom));
    cudaMemcpy(dev_geoms, scene->geoms.data(), scene->geoms.size() * sizeof(Geom), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_intersections, pixelcount * sizeof(ShadeableIntersection));
    cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

    std::vector<Material> devMats;
    devMats.reserve(scene->materials.size());
    for (const MaterialHost& mh : scene->materials) {
        Material m = mh;
        m.diffuseMap = 0;
        m.specularMap = 0;
        if (!mh.diffuseTexPath.empty())
            m.diffuseMap = TextureLoader::loadTexture(mh.diffuseTexPath).object;
        if (!mh.specularTexPath.empty())
            m.specularMap = TextureLoader::loadTexture(mh.specularTexPath).object;
        devMats.push_back(m);
    }
    cudaMalloc(&dev_materials, devMats.size() * sizeof(Material));
    cudaMemcpy(dev_materials, devMats.data(),
        devMats.size() * sizeof(Material), cudaMemcpyHostToDevice);

    // construct bvh
    std::vector<Triangle> orderedTris;
    buildBVH(scene->triangles, orderedTris);

    bvhNodeCount = (int)bvh_gpu.size();
    cudaMalloc(&dev_bvh, bvhNodeCount * sizeof(BVH_node_gpu));
    cudaMemcpy(dev_bvh, bvh_gpu.data(),
        bvhNodeCount * sizeof(BVH_node_gpu), cudaMemcpyHostToDevice);

    num_tris = (int)orderedTris.size();
    if (num_tris > 0) {
        cudaMalloc(&dev_tris, num_tris * sizeof(Triangle));
        cudaMemcpy(dev_tris, orderedTris.data(),
            num_tris * sizeof(Triangle), cudaMemcpyHostToDevice);
    }

    std::vector<int> emissiveGeoms;
    for (int i = 0; i < (int)scene->geoms.size(); i++)
    {
        if (scene->materials[scene->geoms[i].materialid].emittance > 0.0f)
        {
            emissiveGeoms.push_back(i);
        }
    }
    num_emissive_geoms = (int)emissiveGeoms.size();
    if (num_emissive_geoms > 0)
    {
        cudaMalloc(&dev_emissive_geoms, num_emissive_geoms * sizeof(int));
        cudaMemcpy(dev_emissive_geoms, emissiveGeoms.data(),
            num_emissive_geoms * sizeof(int), cudaMemcpyHostToDevice);
    }
    else
    {
        dev_emissive_geoms = NULL;
    }

        checkCUDAError("pathtraceInit");
}

void pathtraceFree()
{
    cudaFree(dev_image);  // no-op if dev_image is null
    cudaFree(dev_paths);
    cudaFree(dev_geoms);
    cudaFree(dev_materials);
    cudaFree(dev_intersections);
    cudaFree(dev_tris);
    cudaFree(dev_bvh);
    cudaFree(dev_emissive_geoms);
    dev_tris = NULL;
    dev_bvh = NULL;
    dev_emissive_geoms = NULL;
    num_emissive_geoms = 0;
    num_tris = 0;

    checkCUDAError("pathtraceFree");
}

struct MaterialIdComparator
{
    __host__ __device__ bool operator()(const ShadeableIntersection & a, const ShadeableIntersection & b) const
    {
        return a.materialId < b.materialId;
    }
};

__device__ void intersectScene(
    const Ray& ray,
    Geom* geoms,
    int geoms_size,
    Triangle* tris,
    int num_tris,
    BVH_node_gpu* bvh,
    float& t_out,
    int& materialId_out,
    glm::vec3& normal_out,
    glm::vec2& uv_out)
{
    float t_min = FLT_MAX;
    int hit_material = -1;
    glm::vec3 normal;
    glm::vec2 uv;
    bool outside = true;

    glm::vec3 tmp_intersect;
    glm::vec3 tmp_normal;
    float t;

    for (int i = 0; i < geoms_size; i++)
    {
        Geom& geom = geoms[i];

        if (geom.type == CUBE)
        {
            t = boxIntersectionTest(geom, ray, tmp_intersect, tmp_normal, outside);
        }
        else if (geom.type == SPHERE)
        {
            t = sphereIntersectionTest(geom, ray, tmp_intersect, tmp_normal, outside);
        }
        else
        {
            continue;
        }

        if (t > 0.0f && t < t_min)
        {
            t_min = t;
            normal = tmp_normal;
            hit_material = geom.materialid;
            uv = glm::vec2(0.0f);
        }
    }

    if (num_tris > 0)
    {
        glm::vec3 invDir = glm::vec3(1.0f) / ray.direction;

        int stack[64];
        int sp = 1;
        stack[0] = 0;

        while (sp > 0)
        {
            int nodeIdx = stack[--sp];
            BVH_node_gpu node = bvh[nodeIdx];

            float boxT;
            if (!intersectAABB(node.box, ray.origin, invDir, boxT)) continue;
            if (boxT > t_min) continue;

            if (node.left < 0)
            {
                for (int i = node.start; i < node.start + node.count; i++)
                {
                    glm::vec3 tmpBary;
                    float tt;
                    Triangle& tri = tris[i];
                    Ray rayMut = ray;
                    if (triangleIntersectionTest(tri, rayMut, tt, tmpBary)
                        && tt > 0.0f && tt < t_min)
                    {
                        const Triangle& tr = tris[i];
                        t_min = tt;
                        normal = glm::normalize(tmpBary.x * tr.n[0] + tmpBary.y * tr.n[1] + tmpBary.z * tr.n[2]);
                        uv = tmpBary.x * tr.uv[0] + tmpBary.y * tr.uv[1] + tmpBary.z * tr.uv[2];
                        hit_material = tr.materialid;
                    }
                }
            }
            else
            {
                stack[sp++] = node.left;
                stack[sp++] = node.right;
            }
        }
    }

    if (hit_material == -1)
    {
        t_out = -1.0f;
        materialId_out = -1;
    }
    else
    {
        t_out = t_min;
        materialId_out = hit_material;
        normal_out = normal;
        uv_out = uv;
    }
}



__device__ glm::vec3 samplePointOnCube(const Geom& geom, glm::vec3& outNormal, thrust::default_random_engine& rng)
{
    thrust::uniform_real_distribution<float> u01(0, 1);
    int face = (int)(u01(rng) * 6.0f);
    if (face > 5) face = 5;
    float u = u01(rng) - 0.5f;
    float v = u01(rng) - 0.5f;

    glm::vec3 localPoint;
    glm::vec3 localNormal;
    switch (face)
    {
    case 0: localPoint = glm::vec3(-0.5f, u, v); localNormal = glm::vec3(-1, 0, 0); break;
    case 1: localPoint = glm::vec3(0.5f, u, v);  localNormal = glm::vec3(1, 0, 0);  break;
    case 2: localPoint = glm::vec3(u, -0.5f, v); localNormal = glm::vec3(0, -1, 0); break;
    case 3: localPoint = glm::vec3(u, 0.5f, v);  localNormal = glm::vec3(0, 1, 0);  break;
    case 4: localPoint = glm::vec3(u, v, -0.5f); localNormal = glm::vec3(0, 0, -1); break;
    default: localPoint = glm::vec3(u, v, 0.5f); localNormal = glm::vec3(0, 0, 1);  break;
    }

    outNormal = glm::normalize(multiplyMV(geom.invTranspose, glm::vec4(localNormal, 0.0f)));
    return multiplyMV(geom.transform, glm::vec4(localPoint, 1.0f));
}

__device__ float cubeWorldSurfaceArea(const Geom& geom)
{
    glm::vec3 s = geom.scale;
    return 2.0f * (s.x * s.y + s.y * s.z + s.z * s.x);
}

__device__ glm::vec3 samplePointOnSphere(const Geom& geom, glm::vec3& outNormal, thrust::default_random_engine& rng)
{
    thrust::uniform_real_distribution<float> u01(0, 1);
    float z = 1.0f - 2.0f * u01(rng);
    float r = sqrtf(fmaxf(0.0f, 1.0f - z * z));
    float phi = TWO_PI * u01(rng);
    glm::vec3 localPoint = glm::vec3(r * cosf(phi), r * sinf(phi), z) * 0.5f;
    glm::vec3 localNormal = glm::normalize(localPoint * 2.0f);
    outNormal = glm::normalize(multiplyMV(geom.invTranspose, glm::vec4(localNormal, 0.0f)));
    return multiplyMV(geom.transform, glm::vec4(localPoint, 1.0f));
}

__device__ float sphereWorldSurfaceArea(const Geom& geom)
{
    float r = 0.5f * fmaxf(geom.scale.x, fmaxf(geom.scale.y, geom.scale.z));
    return 4.0f * PI * r * r;
}

__device__ bool isVisibleToPoint(
    glm::vec3 origin,
    glm::vec3 target,
    Geom* geoms,
    int geoms_size,
    Triangle* tris,
    int num_tris,
    BVH_node_gpu* bvh)
{
    glm::vec3 wi = target - origin;
    float dist = glm::length(wi);
    if (dist < EPSILON) return false;

    Ray shadowRay;
    shadowRay.origin = origin + wi / dist * 0.001f;
    shadowRay.direction = wi / dist;

    float t;
    int matId;
    glm::vec3 n;
    glm::vec2 uv;
    intersectScene(shadowRay, geoms, geoms_size, tris, num_tris, bvh, t, matId, n, uv);
    return t < 0.0f || t >= dist - 0.01f;
}

__device__ glm::vec3 sampleDirectLight(
    glm::vec3 hitPoint,
    glm::vec3 surfaceNormal,
    glm::vec3 throughput,
    glm::vec3 albedo,
    int* emissiveGeoms,
    int numEmissiveGeoms,
    Geom* geoms,
    Material* materials,
    Triangle* tris,
    int num_tris,
    BVH_node_gpu* bvh,
    int geoms_size,
    thrust::default_random_engine& rng)
{
    if (numEmissiveGeoms <= 0) return glm::vec3(0.0f);

    thrust::uniform_real_distribution<float> u01(0, 1);
    int lightIdx = (int)(u01(rng) * numEmissiveGeoms);
    if (lightIdx >= numEmissiveGeoms) lightIdx = numEmissiveGeoms - 1;

    Geom lightGeom = geoms[emissiveGeoms[lightIdx]];
    Material lightMat = materials[lightGeom.materialid];
    if (lightMat.emittance <= 0.0f) return glm::vec3(0.0f);

    glm::vec3 lightNormal;
    glm::vec3 lightPoint;
    float lightAreaPdf;

    if (lightGeom.type == CUBE)
    {
        lightPoint = samplePointOnCube(lightGeom, lightNormal, rng);
        lightAreaPdf = 1.0f / (cubeWorldSurfaceArea(lightGeom) * (float)numEmissiveGeoms);
    }
    else if (lightGeom.type == SPHERE)
    {
        lightPoint = samplePointOnSphere(lightGeom, lightNormal, rng);
        lightAreaPdf = 1.0f / (sphereWorldSurfaceArea(lightGeom) * (float)numEmissiveGeoms);
    }
    else
    {
        return glm::vec3(0.0f);
    }

    glm::vec3 wi = lightPoint - hitPoint;
    float distSq = glm::dot(wi, wi);
    if (distSq < EPSILON) return glm::vec3(0.0f);
    float dist = sqrtf(distSq);
    wi /= dist;

    float cosThetaSurface = fmaxf(0.0f, glm::dot(surfaceNormal, wi));
    float cosThetaLight = fmaxf(0.0f, glm::dot(lightNormal, -wi));
    if (cosThetaSurface <= 0.0f || cosThetaLight <= 0.0f) return glm::vec3(0.0f);

    if (!isVisibleToPoint(hitPoint + surfaceNormal * 0.001f, lightPoint, geoms, geoms_size, tris, num_tris, bvh))
    {
        return glm::vec3(0.0f);
    }

    glm::vec3 emitted = lightMat.color * lightMat.emittance;
    glm::vec3 brdf = albedo * (1.0f / PI);
    return throughput * brdf * emitted * cosThetaSurface * cosThetaLight / (distSq * lightAreaPdf);
}

__device__ Material resolveMaterial(Material material, const ShadeableIntersection& intersection)
{
    if (material.diffuseMap != 0)
    {
        float4 tx = tex2D<float4>(material.diffuseMap, intersection.uv.x, intersection.uv.y);
        material.color = glm::vec3{ tx.x, tx.y, tx.z };
    }
    if (material.specularMap != 0)
    {
        float4 tx = tex2D<float4>(material.specularMap, intersection.uv.x, intersection.uv.y);
        float luminance = tx.x * 0.2126f + tx.y * 0.7152f + tx.z * 0.0722f;
        material.specular = luminance * 300;
    }
    return material;
}



/**
* Generate PathSegments with rays from the camera through the screen into the
* scene, which is the first bounce of rays.
*
* Antialiasing - add rays for sub-pixel sampling
* motion blur - jitter rays "in time"
* lens effect - jitter ray origin positions based on a lens
*/
__global__ void generateRayFromCamera(Camera cam, int iter, int traceDepth, PathSegment* pathSegments)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < cam.resolution.x && y < cam.resolution.y) {
        int index = x + (y * cam.resolution.x);
        PathSegment& segment = pathSegments[index];

        segment.ray.origin = cam.position;
        segment.color = glm::vec3(0.0f);
        segment.throughput = glm::vec3(1.0f);

        thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, traceDepth + 1);
        thrust::uniform_real_distribution<float> u01(0, 1);
        float jitterX = u01(rng) - 0.5f;
        float jitterY = u01(rng) - 0.5f;

        segment.ray.direction = glm::normalize(cam.view
            - cam.right * cam.pixelLength.x * ((float)x + jitterX - (float)cam.resolution.x * 0.5f)
            - cam.up * cam.pixelLength.y * ((float)y + jitterY - (float)cam.resolution.y * 0.5f)
        );

        segment.pixelIndex = index;
        segment.remainingBounces = traceDepth;
    }
}

// TODO:
// computeIntersections handles generating ray intersections ONLY.
// Generating new rays is handled in your shader(s).
// Feel free to modify the code below.
__global__ void computeIntersections(
    int depth,
    int num_paths,
    PathSegment* pathSegments,
    Geom* geoms, Triangle* tris, int num_tris, int geoms_size,
    ShadeableIntersection* intersections, BVH_node_gpu* bvh)
{
    int path_index = blockIdx.x * blockDim.x + threadIdx.x;

    if (path_index < num_paths)
    {
        PathSegment pathSegment = pathSegments[path_index];

        float t;
        glm::vec3 intersect_point;
        glm::vec3 normal;
        float t_min = FLT_MAX;
        int hit_material = -1;
        bool outside = true;

        glm::vec3 tmp_intersect;
        glm::vec3 tmp_normal;

        // naive parse through global geoms

        for (int i = 0; i < geoms_size; i++)
        {
            Geom& geom = geoms[i];

            if (geom.type == CUBE)
            {
                t = boxIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
            }
            else if (geom.type == SPHERE)
            {
                t = sphereIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
            }

            // Compute the minimum t from the intersection tests to determine what
            // scene geometry object was hit first.
            // 
            // random idea i got
            // maybe we can sort all geometry first on the axis on which the camera is aligned on the most, then we can automatically obtain the first hit object.
            // a step further would be to check if objects intersect with view frustum first (or atleast roughly a cube) and only sort / check intersections among these
            if (t > 0.0f && t_min > t)
            {
                t_min = t;
                intersect_point = tmp_intersect;
                normal = tmp_normal;
                hit_material = geom.materialid;
            }
        }

        // triangle meshes naive
        /*glm::vec3 bary;
        glm::vec2 uv;
        for (int i = 0; i < num_tris; i++) {
            if (triangleIntersectionTest(tris[i], pathSegment.ray, t, bary) && t < t_min) {
                const Triangle& tr = tris[i];
                t_min = t;
                normal = glm::normalize(bary.x * tr.n[0] + bary.y * tr.n[1] + bary.z * tr.n[2]);
                uv = bary.x * tr.uv[0] + bary.y * tr.uv[1] + bary.z * tr.uv[2];
                hit_material = tr.materialid;
            }
        }*/


        // bvh

        glm::vec3 bary;
        glm::vec2 uv;
        int visitedNodes = 0;
        if (num_tris > 0)
        {
            glm::vec3 invDir = glm::vec3(1.0f) / pathSegment.ray.direction;

            int stack[64];
            int sp = 1;
            stack[0] = 0;   // root
            

            while (sp > 0)
            {
                int nodeIdx = stack[--sp];
                BVH_node_gpu node = bvh[nodeIdx];

                float boxT;
                if (!intersectAABB(node.box, pathSegment.ray.origin, invDir, boxT)) continue;
                if (boxT > t_min) continue;   // a closer hit already exists -> prune
                visitedNodes++;

                if (node.left < 0)   // leaf
                {
                    for (int i = node.start; i < node.start + node.count; i++)
                    {
                        glm::vec3 tmpBary;
                        float tt;
                        if (triangleIntersectionTest(tris[i], pathSegment.ray, tt, tmpBary)
                            && tt > 0.0f && tt < t_min)
                        {
                            const Triangle& tr = tris[i];
                            t_min = tt;
                            normal = glm::normalize(tmpBary.x * tr.n[0] + tmpBary.y * tr.n[1] + tmpBary.z * tr.n[2]);
                            uv = tmpBary.x * tr.uv[0] + tmpBary.y * tr.uv[1] + tmpBary.z * tr.uv[2];
                            hit_material = tr.materialid;
                        }
                    }
                }
                else
                {
                    stack[sp++] = node.left;
                    stack[sp++] = node.right;
                }
            }
        }
        


        
        intersections[path_index].debugVisitedNodes = visitedNodes;

        if (hit_material == -1)
        {
            intersections[path_index].t = -1.0f;
        }
        else
        {
            // The ray hits something
            intersections[path_index].t = t_min;
            intersections[path_index].materialId = hit_material;
            intersections[path_index].surfaceNormal = normal;
            intersections[path_index].uv = uv;

        }
    }
}

// LOOK: "fake" shader demonstrating what you might do with the info in
// a ShadeableIntersection, as well as how to use thrust's random number
// generator. Observe that since the thrust random number generator basically
// adds "noise" to the iteration, the image should start off noisy and get
// cleaner as more iterations are computed.
//
// Note that this shader does NOT do a BSDF evaluation!
// Your shaders should handle that - this can allow techniques such as
// bump mapping.
__global__ void shadeDiffuseBRDFMaterial(
    int iter,
    int traceDepth,
    int num_paths,
    ShadeableIntersection* shadeableIntersections,
    PathSegment* pathSegments,
    Material* materials,
    Geom* geoms,
    int geoms_size,
    Triangle* tris,
    int num_tris,
    BVH_node_gpu* bvh,
    int* emissiveGeoms,
    int numEmissiveGeoms)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_paths)
    {   
        PathSegment& pathSegment = pathSegments[idx];
        if (pathSegment.remainingBounces <= 0) return;

        ShadeableIntersection intersection = shadeableIntersections[idx];
//#define DEBUG_BVH
#ifdef DEBUG_BVH
        float v = intersection.debugVisitedNodes / 32.0f;
        v = glm::clamp(v, 0.0f, 1.0f);
        pathSegment.color = glm::vec3(v);
        pathSegment.throughput = glm::vec3(0.0f);
        pathSegment.remainingBounces = 0;
        return;
#endif

        thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, pathSegment.remainingBounces);

        if (intersection.t > 0.0f)
        {
            Material material = resolveMaterial(materials[intersection.materialId], intersection);
            glm::vec3 hitPoint = pathSegment.ray.origin + pathSegment.ray.direction * intersection.t;
            glm::vec3 normal = glm::normalize(intersection.surfaceNormal);

            if (material.emittance > 0.0f)
            {
                pathSegment.color += pathSegment.throughput * (material.color * material.emittance);
                pathSegment.remainingBounces = 0;
                return;
            }

            glm::vec3 albedo = material.color;

            pathSegment.color += sampleDirectLight(
                hitPoint,
                normal,
                pathSegment.throughput,
                albedo,
                emissiveGeoms,
                numEmissiveGeoms,
                geoms,
                materials,
                tris,
                num_tris,
                bvh,
                geoms_size,
                rng);

            if (pathSegment.remainingBounces <= 1)
            {
                pathSegment.remainingBounces = 0;
                return;
            }

            scatterRay(pathSegment, hitPoint, normal, material, rng);
        }
        else
        {
            pathSegment.remainingBounces = 0;
      
        }
    }
}

// Add the current iteration's output to the overall image
__global__ void finalGather(int nPaths, glm::vec3* image, PathSegment* iterationPaths)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (index < nPaths)
    {
        PathSegment iterationPath = iterationPaths[index];
        image[iterationPath.pixelIndex] += iterationPath.color;
    }
}

/**
 * Wrapper for the __global__ call that sets up the kernel calls and does a ton
 * of memory management
 */
void pathtrace(uchar4* pbo, int frame, int iter)
{
    const int traceDepth = hst_scene->state.traceDepth;
    const Camera& cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    // 2D block for generating ray from camera
    const dim3 blockSize2d(8, 8);
    const dim3 blocksPerGrid2d(
        (cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
        (cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

    // 1D block for path tracing
    const int blockSize1d = 128;

    ///////////////////////////////////////////////////////////////////////////

    // Recap:
    // * Initialize array of path rays (using rays that come out of the camera)
    //   * You can pass the Camera object to that kernel.
    //   * Each path ray must carry at minimum a (ray, color) pair,
    //   * where color starts as the multiplicative identity, white = (1, 1, 1).
    //   * This has already been done for you.
    // * For each depth:
    //   * Compute an intersection in the scene for each path ray.
    //     A very naive version of this has been implemented for you, but feel
    //     free to add more primitives and/or a better algorithm.
    //     Currently, intersection distance is recorded as a parametric distance,
    //     t, or a "distance along the ray." t = -1.0 indicates no intersection.
    //     * Color is attenuated (multiplied) by reflections off of any object
    //   * TODO: Stream compact away all of the terminated paths.
    //     You may use either your implementation or `thrust::remove_if` or its
    //     cousins.
    //     * Note that you can't really use a 2D kernel launch any more - switch
    //       to 1D.
    //   * TODO: Shade the rays that intersected something or didn't bottom out.
    //     That is, color the ray by performing a color computation according
    //     to the shader, then generate a new ray to continue the ray path.
    //     We recommend just updating the ray's PathSegment in place.
    //     Note that this step may come before or after stream compaction,
    //     since some shaders you write may also cause a path to terminate.
    // * Finally, add this iteration's results to the image. This has been done
    //   for you.

    // TODO: perform one iteration of path tracing

    generateRayFromCamera<<<blocksPerGrid2d, blockSize2d>>>(cam, iter, traceDepth, dev_paths);
    checkCUDAError("generate camera ray");

    int depth = 0;
    PathSegment* dev_path_end = dev_paths + pixelcount;
    int num_paths = dev_path_end - dev_paths;

    // --- PathSegment Tracing Stage ---
    // Shoot ray into scene, bounce between objects, push shading chunks

    bool iterationComplete = false;
    while (!iterationComplete)
    {
        // clean shading chunks
        cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

        // tracing
        dim3 numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;
        computeIntersections<<<numblocksPathSegmentTracing, blockSize1d>>> (
            depth,
            num_paths,
            dev_paths,
            dev_geoms,
            dev_tris, num_tris,
            hst_scene->geoms.size(),
            dev_intersections, dev_bvh
        );
        checkCUDAError("trace one bounce");
        cudaDeviceSynchronize();
        depth++;

        // TODO:
        // --- Shading Stage ---
        // Shade path segments based on intersections and generate new rays by
        // evaluating the BSDF.
        // Start off with just a big kernel that handles all the different
        // materials you have in the scenefile.
        // TODO: compare between directly shading the path segments and shading
        // path segments that have been reshuffled to be contiguous in memory.

        // shuffle paths based on intersections'materialID
        // note performance dips to 15fps from 60 at the simple cornell scene. will likely yield better results as we get a larger num of materials.
       // thrust::sort_by_key(thrust::device, dev_intersections, dev_intersections + num_paths, dev_paths, MaterialIdComparator());
        shadeDiffuseBRDFMaterial<<<numblocksPathSegmentTracing, blockSize1d>>>(
            iter,
            traceDepth,
            num_paths,
            dev_intersections,
            dev_paths,
            dev_materials,
            dev_geoms,
            (int)hst_scene->geoms.size(),
            dev_tris,
            num_tris,
            dev_bvh,
            dev_emissive_geoms,
            num_emissive_geoms
        );

        // compact: live paths to front, dead to back
        dev_path_end = thrust::partition(
            thrust::device,
            dev_paths, dev_paths + num_paths,
            IsPathActive());
        num_paths = (int)(dev_path_end - dev_paths);


        iterationComplete = depth >= traceDepth || num_paths == 0; 

        if (guiData != NULL)
        {
            guiData->TracedDepth = depth;
        }
    }

    // Assemble this iteration and apply it to the image
    dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;
    finalGather<<<numBlocksPixels, blockSize1d>>>(pixelcount, dev_image, dev_paths);

    ///////////////////////////////////////////////////////////////////////////

    // Send results to OpenGL buffer for rendering
    sendImageToPBO<<<blocksPerGrid2d, blockSize2d>>>(pbo, cam.resolution, iter, dev_image);

    // Retrieve image from GPU
    cudaMemcpy(hst_scene->state.image.data(), dev_image,
        pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

    checkCUDAError("pathtrace");
}
