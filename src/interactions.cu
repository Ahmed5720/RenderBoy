#include "interactions.h"

#include "utilities.h"

#include <thrust/random.h>

__host__ __device__ glm::vec3 calculateRandomDirectionInHemisphere( glm::vec3 normal, thrust::default_random_engine &rng)
{
    thrust::uniform_real_distribution<float> u01(0, 1);

    float up = sqrt(u01(rng)); // cos(theta)
    float over = sqrt(1 - up * up); // sin(theta)
    float around = u01(rng) * TWO_PI;

    // Find a direction that is not the normal based off of whether or not the
    // normal's components are all equal to sqrt(1/3) or whether or not at
    // least one component is less than sqrt(1/3). Learned this trick from
    // Peter Kutz.

    glm::vec3 directionNotNormal;
    if (abs(normal.x) < SQRT_OF_ONE_THIRD)
    {
        directionNotNormal = glm::vec3(1, 0, 0);
    }
    else if (abs(normal.y) < SQRT_OF_ONE_THIRD)
    {
        directionNotNormal = glm::vec3(0, 1, 0);
    }
    else
    {
        directionNotNormal = glm::vec3(0, 0, 1);
    }

    // Use not-normal direction to generate two perpendicular directions
    glm::vec3 perpendicularDirection1 =
        glm::normalize(glm::cross(normal, directionNotNormal));
    glm::vec3 perpendicularDirection2 =
        glm::normalize(glm::cross(normal, perpendicularDirection1));

    return up * normal
        + cos(around) * over * perpendicularDirection1
        + sin(around) * over * perpendicularDirection2;
}

//__host__ __device__ void scatterRay(
//    PathSegment& pathSegment,
//    glm::vec3 intersect,
//    glm::vec3 normal,
//    const Material& m,
//    thrust::default_random_engine& rng)
//{
//    // TODO: implement this.
//    // A basic implementation of pure-diffuse shading will just call the
//    // calculateRandomDirectionInHemisphere defined above.
//    pathSegment.ray.origin = intersect + normal * 0.001f;
//    //glm::vec3 diffuseDir = glm::normalize(calculateRandomDirectionInHemisphere(normal, rng));
//    //glm::vec3 specularDir = glm::reflect(pathSegment.ray.direction, normal);
//    //float smoothness = 1 - m.specular; // m.shininess;
//
//    //pathSegment.ray.direction = glm::normalize(glm::mix(diffuseDir, specularDir, smoothness));
//
//    const bool isSpecular = m.specular > 0.0f;
//
//    if (isSpecular)
//    {
//        glm::vec3 reflectDir = glm::reflect(pathSegment.ray.direction, normal);
//        float roughness = 1.0f / (1.0f + fmaxf(m.specular, m.shininess * 100.0f) * 0.1f);
//        glm::vec3 randomOffset = calculateRandomDirectionInHemisphere(normal, rng) * roughness;
//        pathSegment.ray.direction = glm::normalize(reflectDir + randomOffset);
//        pathSegment.throughput *= m.color;
//    }
//    else
//    {
//        pathSegment.ray.direction = glm::normalize(calculateRandomDirectionInHemisphere(normal, rng));
//        pathSegment.throughput *= m.color;
//    }
//    //if(smoothness < 0.5f)
// //   pathSegment.color *= m.color;
//    pathSegment.remainingBounces--;
//}

__host__ __device__ BsdfParams makeParams(const Material& m)
{
    BsdfParams params;
    params.diffuseColor = m.color * (1.0f - m.metallic); // full metals have no diffuse // sample textures later
    params.F0 = glm::mix(glm::vec3(0.04f), m.color, m.metallic); // for none metals base Reflectiviely F0 is const 0.04, for metals its their color
    params.alpha = fmaxf(1e-3f, m.roughness * m.roughness);
    return params;

}

// how does this work?
__host__ __device__ void buildONB(const glm::vec3& n, glm::vec3& t, glm::vec3& b) {
    float s = copysignf(1.0f, n.z);
    float a = -1.0f / (s + n.z);
    t = glm::vec3(1.0f + s * n.x * n.x * a, s * n.x * n.y * a, -s * n.x);
    b = glm::vec3(n.x * n.y * a, s + n.y * n.y * a, -n.y);
}

// normal distribution function D
__host__ __device__ float D_GGX(float NoH, float alpha) {
    float a2 = alpha * alpha;
    float d = NoH * NoH * (a2 - 1.0f) + 1.0f;
    return a2 / (PI * d * d);
}

// Exact Smith-GGX G1 (no k-approximation; consistent with our sampling)
// different from Smith schlick Beckman GGX?
__host__ __device__ float smithG1(float NoX, float alpha) {
    float a2 = alpha * alpha;
    return 2.0f * NoX / (NoX + sqrtf(a2 + (1.0f - a2) * NoX * NoX));
}

// Geometry SHadowing function G
__host__ __device__ float smithG(float NoV, float NoL, float alpha) {
    return smithG1(NoV, alpha) * smithG1(NoL, alpha);
}

// Fresnel function F, cosTheta = V dot H
__host__ __device__ glm::vec3 fresnelSchlick(float cosTheta, glm::vec3 F0) {
    float m = powf(fmaxf(1.0f - cosTheta, 0.0f), 5.0f);
    return F0 + (glm::vec3(1.0f) - F0) * m;
}


__host__ __device__ float luminance(glm::vec3 c) {
    return 0.2126f * c.x + 0.7152f * c.y + 0.0722f * c.z;
}

// compute the probability of choosing the specular lobe
__host__ __device__ float specProb(const BsdfParams& p)
{
    float ws = luminance(p.F0);
    float wd = luminance(p.diffuseColor);
    return glm::clamp(ws / fmaxf(ws + wd, 1e-4f), 0.1f, 0.9f);
}
// evaluates the Physically Based BiDirectional Scattering Distribution Function
// given normal, view dir, Light Direction, and Material properties P for roughness and color
// C = kd (diffuse) + Ks (specular)
// kd + ks = 1, ks = F
// diffuse = Lambert = Color / pi
// Specular = Cook-torrence = D G / 4 (V * N) (L * N)
__host__ __device__ glm::vec3 bsdfEval(const BsdfParams& p, const glm::vec3& N, const glm::vec3& V, const glm::vec3& L)
{
    
    float NoV = glm::dot(N, V);
    float NoL = glm::dot(N, L);
    if (NoV <= 0.0f || NoL <= 0.0f) return glm::vec3(0.0f);

    glm::vec3 H = glm::normalize(V + L);
    float NoH = fmaxf(glm::dot(N, H), 0.0f);
    float VoH = fmaxf(glm::dot(V, H), 0.0f);

    // specular (Cook-Torrance)
    float D = D_GGX(NoH, p.alpha);
    float G = smithG(NoV, NoL, p.alpha);
    glm::vec3 F = fresnelSchlick(VoH, p.F0);
    glm::vec3 spec = (D * G) * F / (4.0f * NoV * NoL);

    // diffuse (Lambert), energy-weighted by (1 - F)
    glm::vec3 kd = glm::vec3(1.0f) - F;
    glm::vec3 diff = kd * p.diffuseColor / PI;

    return diff + spec;

}


// gives the final ray bounce direction
// stochasitcally choose between diffuse, and specular direction based on prob(Specular)
// diffuse dir is simply CalcRandomDirectionInHem
// specular Dir now instead of being = reflection of -V on N, its now reflection of -V on H, where H is the half vector sampled from GGX
__host__ __device__ glm::vec3 bsdfSample(const BsdfParams& p, const glm::vec3& N, const glm::vec3& V, float pSpec, thrust::default_random_engine& rng)
{
    thrust::uniform_real_distribution<float> u01(0.0f, 1.0f);
    if (u01(rng) < pSpec)
    {
        // sample H from GGX
        // whats going on here? 
        float u1 = u01(rng), u2 = u01(rng);
        float phi = 2.0f * PI * u1;
        float a2 = p.alpha * p.alpha;
        float cosT = sqrtf((1.0f - u2) / (1.0f + (a2 - 1.0f) * u2));
        float sinT = sqrtf(fmaxf(0.0f, 1.0f - cosT * cosT));
        glm::vec3 Hl(sinT * cosf(phi), sinT * sinf(phi), cosT);

        glm::vec3 T, B; buildONB(N, T, B);
        glm::vec3 H = Hl.x * T + Hl.y * B + Hl.z * N;
        H = glm::normalize(H);
        glm::vec3 L = glm::reflect(-V, H);
        if (glm::dot(N, L) <= 0.0f)
            return calculateRandomDirectionInHemisphere(N, rng);
        else
            return L;

    }
    else
        return calculateRandomDirectionInHemisphere(N, rng);
}

__host__ __device__ float bsdfPDF(const BsdfParams& p, const glm::vec3 N, const glm::vec3 V, const glm::vec3 L, float pSpec)
{   

    // again no clue whats going on here
    float NoL = glm::dot(N, L);
    float NoV = glm::dot(N, V);
    if (NoL <= 0.0f || NoV <= 0.0f) return 0.0f;

    glm::vec3 H = glm::normalize(V + L);
    float NoH = fmaxf(glm::dot(N, H), 0.0f);
    float VoH = fmaxf(glm::dot(V, H), 1e-6f);

    float pdfDiff = NoL / PI;
    float pdfSpec = D_GGX(NoH, p.alpha) * NoH / (4.0f * VoH);
    return (1.0f - pSpec) * pdfDiff + pSpec * pdfSpec;
}
__host__ __device__ void scatterRay(PathSegment& pathSegment, glm::vec3 intersect, glm::vec3 normal, const BsdfParams& p, thrust::default_random_engine& rng)
{
    // we use the BSDF to determine the scattering direction
    // The point to remember is that we use a different PDF for diffuse and for Specular
    // here "view" V direction is our incident ray and lightDir L is our outbound ray

    // float roughness = 1 - m.specular;

    glm::vec3 V = -glm::normalize(pathSegment.ray.direction);
    float pSpec = specProb(p);
    

    // stochasitcally choose between diffuse, and specular direction based on prob(Specular)
    // diffuse dir is simply CalcRandomDirectionInHem
    // specular Dir now instead of being = reflection of -V on N, its now reflection of -V on H, where H is the half vector sampled from GGX
    glm::vec3 L = bsdfSample(p, normal, V, pSpec, rng);
    float NoL = glm::dot(normal, L);

    glm::vec3 f = bsdfEval(p, normal, V, L);
    float pdf = bsdfPDF(p, normal, V, L, pSpec);
    pathSegment.throughput *= f * NoL / pdf;


    // fix nan values problem, for now this does the job
    if (!isfinite(pdf) || pdf <= 1e-7f)
    {
        pathSegment.throughput = glm::vec3(1, 0, 1);
        pathSegment.remainingBounces = 0;
        return;
    }

    //if (!isfinite(pathSegment.throughput.x) ||
    //    !isfinite(pathSegment.throughput.y) ||
    //    !isfinite(pathSegment.throughput.z))
    //{
    //    pathSegment.throughput = glm::vec3(1, 0, 1); // magenta
    //}
    pathSegment.ray.origin = intersect + normal * 0.001f;
    pathSegment.ray.direction = L;
    pathSegment.remainingBounces--;
}