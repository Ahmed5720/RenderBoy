#pragma once

#include "sceneStructs.h"
#include <vector>

class Scene
{
private:
    void loadFromJSON(const std::string& jsonName);
    void Scene::loadObj(const std::string& filename, const glm::mat4& transform);
public:
    Scene(std::string filename);

    std::vector<Geom> geoms;
    std::vector<MaterialHost> materials;
    std::vector<Triangle> triangles;
    RenderState state;
};
