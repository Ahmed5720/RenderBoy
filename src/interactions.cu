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

__host__ __device__ void scatterRay(
    PathSegment& pathSegment,
    glm::vec3 intersect,
    glm::vec3 normal,
    const Material& m,
    thrust::default_random_engine& rng)
{
    // TODO: implement this.
    // A basic implementation of pure-diffuse shading will just call the
    // calculateRandomDirectionInHemisphere defined above.
    pathSegment.ray.origin = intersect + normal * 0.001f;
    //glm::vec3 diffuseDir = glm::normalize(calculateRandomDirectionInHemisphere(normal, rng));
    //glm::vec3 specularDir = glm::reflect(pathSegment.ray.direction, normal);
    //float smoothness = 1 - m.specular; // m.shininess;

    //pathSegment.ray.direction = glm::normalize(glm::mix(diffuseDir, specularDir, smoothness));

    const bool isSpecular = m.specular > 0.0f;

    if (isSpecular)
    {
        glm::vec3 reflectDir = glm::reflect(pathSegment.ray.direction, normal);
        float roughness = 1.0f / (1.0f + fmaxf(m.specular, m.shininess * 100.0f) * 0.1f);
        glm::vec3 randomOffset = calculateRandomDirectionInHemisphere(normal, rng) * roughness;
        pathSegment.ray.direction = glm::normalize(reflectDir + randomOffset);
        pathSegment.throughput *= m.color;
    }
    else
    {
        pathSegment.ray.direction = glm::normalize(calculateRandomDirectionInHemisphere(normal, rng));
        pathSegment.throughput *= m.color;
    }
    //if(smoothness < 0.5f)
 //   pathSegment.color *= m.color;
    pathSegment.remainingBounces--;
}
